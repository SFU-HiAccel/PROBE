"""Learned deferral classifier for the LLMSQL proxy cascade.

Implements the post-hoc deferral rule from Nie et al. 2024 (OCL Eq. 5)
and Jitkrittum et al. 2023: train a small classifier to predict whether
the proxy decision is wrong, then defer to the oracle when P(wrong) > 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


DEFAULT_DEFERRAL_FEATURES: Tuple[str, ...] = (
    "z_smooth",
    "z_corrected",
    "similarity",
    "neighbour_disag",
    "rank_position",
    "boundary_dist",
)


def _as_1d_float(values: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).reshape(-1)
    if out.size == 0:
        raise ValueError(f"{name} is empty")
    return out


def _normalise_feature_names(feature_names: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if feature_names is None:
        return DEFAULT_DEFERRAL_FEATURES
    names = tuple(str(x) for x in feature_names)
    unknown = sorted(set(names) - set(DEFAULT_DEFERRAL_FEATURES))
    if unknown:
        raise ValueError(f"Unknown deferral features: {unknown}")
    if not names:
        raise ValueError("At least one deferral feature is required")
    return names


def _rank_position(sorted_idx: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    sorted_idx = np.asarray(sorted_idx, dtype=np.int64).reshape(-1)
    if n is None:
        n = int(sorted_idx.shape[0])
    ranks = np.zeros(int(n), dtype=np.float64)
    if sorted_idx.size == 0:
        return ranks
    ranks[sorted_idx] = np.arange(sorted_idx.size, dtype=np.float64) / max(
        sorted_idx.size - 1, 1
    )
    return ranks


def _neighbour_disag(W: Any, z_smooth: np.ndarray) -> np.ndarray:
    z_smooth = _as_1d_float(z_smooth, "z_smooth")
    n = z_smooth.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if W is None or not issparse(W):
        return out
    W_csr = W.tocsr()
    if W_csr.shape[0] != n:
        return out
    for i in range(n):
        start, end = W_csr.indptr[i], W_csr.indptr[i + 1]
        if end <= start:
            continue
        nbrs = W_csr.indices[start:end]
        weights = W_csr.data[start:end].astype(np.float64, copy=False)
        denom = float(weights.sum())
        if denom <= 0.0:
            continue
        vals = z_smooth[nbrs]
        # Paper §3.4: nd_i = Σ_j W_ij·|z_i − z_j| / Σ_j W_ij — the graph-weighted
        # MEAN ABSOLUTE neighbour gap between row i's smoothed score and each of
        # its neighbours (not the weighted variance of neighbour scores around
        # their own weighted mean).
        out[i] = float((weights * np.abs(vals - z_smooth[i])).sum() / denom)
    return out


@dataclass
class DeferralFeatures:
    z_smooth: np.ndarray
    z_corrected: np.ndarray
    similarity: np.ndarray
    neighbour_disag: np.ndarray
    rank_position: np.ndarray
    boundary_dist: np.ndarray

    @classmethod
    def build(
        cls,
        z_smooth: np.ndarray,
        z_corrected: Optional[np.ndarray],
        similarity: Optional[np.ndarray],
        W: Any,
        sorted_idx: Optional[np.ndarray],
    ) -> "DeferralFeatures":
        z_smooth_arr = _as_1d_float(z_smooth, "z_smooth")
        n = z_smooth_arr.shape[0]
        if z_corrected is None:
            z_corrected_arr = z_smooth_arr.copy()
        else:
            z_corrected_arr = _as_1d_float(z_corrected, "z_corrected")
            if z_corrected_arr.shape[0] != n:
                raise ValueError("z_corrected length does not match z_smooth")
        if similarity is None:
            similarity_arr = np.zeros(n, dtype=np.float64)
        else:
            similarity_arr = _as_1d_float(similarity, "similarity")
            if similarity_arr.shape[0] != n:
                raise ValueError("similarity length does not match z_smooth")
        if sorted_idx is None:
            sorted_idx_arr = np.argsort(-similarity_arr)
        else:
            sorted_idx_arr = np.asarray(sorted_idx, dtype=np.int64).reshape(-1)
            if sorted_idx_arr.shape[0] != n:
                raise ValueError("sorted_idx length does not match z_smooth")
        return cls(
            z_smooth=z_smooth_arr,
            z_corrected=z_corrected_arr,
            similarity=similarity_arr,
            neighbour_disag=_neighbour_disag(W, z_smooth_arr),
            rank_position=_rank_position(sorted_idx_arr, n=n),
            boundary_dist=np.abs(z_smooth_arr - 0.5),
        )

    def stack(self, feature_names: Optional[Sequence[str]] = None) -> np.ndarray:
        names = _normalise_feature_names(feature_names)
        cols = [np.asarray(getattr(self, name), dtype=np.float64) for name in names]
        return np.column_stack(cols)


def build_deferral_features(
    z_smooth: np.ndarray,
    z_corrected: Optional[np.ndarray],
    similarity: Optional[np.ndarray],
    W: Any,
    sorted_idx: Optional[np.ndarray],
    feature_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Build the learned-deferral feature matrix as a numpy array."""
    return DeferralFeatures.build(
        z_smooth, z_corrected, similarity, W, sorted_idx
    ).stack(feature_names)


@dataclass
class DeferralDecision:
    defer_proba: np.ndarray
    defer_mask: np.ndarray
    proxy_pred: np.ndarray
    classifier: Optional["DeferralClassifier"]
    info: Dict[str, Any]


class DeferralClassifier:
    """Small sklearn MLP with feature scaling and an online update hook."""

    def __init__(
        self,
        hidden_dims: Sequence[int] = (32, 16),
        alpha: float = 1e-3,
        max_iter: int = 400,
        random_state: int = 42,
        warm_start: bool = True,
        learning_rate_init: float = 1e-3,
    ):
        self.scaler = StandardScaler()
        self.mlp = MLPClassifier(
            hidden_layer_sizes=tuple(int(x) for x in hidden_dims),
            activation="relu",
            solver="adam",
            alpha=float(alpha),
            max_iter=int(max_iter),
            random_state=int(random_state),
            warm_start=bool(warm_start),
            learning_rate_init=float(learning_rate_init),
            early_stopping=False,
        )
        self._fitted = False
        self._scaler_fitted = False

    def fit(self, X: np.ndarray, z: np.ndarray) -> "DeferralClassifier":
        X_arr = np.asarray(X, dtype=np.float64)
        z_arr = np.asarray(z, dtype=np.int32).reshape(-1)
        self.scaler.fit(X_arr)
        self._scaler_fitted = True
        self.mlp.fit(self.scaler.transform(X_arr), z_arr)
        self._fitted = True
        return self

    def partial_fit(self, X: np.ndarray, z: np.ndarray) -> "DeferralClassifier":
        X_arr = np.asarray(X, dtype=np.float64)
        z_arr = np.asarray(z, dtype=np.int32).reshape(-1)
        if X_arr.size == 0 or z_arr.size == 0:
            return self
        if not self._scaler_fitted:
            self.scaler.fit(X_arr)
            self._scaler_fitted = True
        else:
            self.scaler.partial_fit(X_arr)
        Xs = self.scaler.transform(X_arr)
        if not hasattr(self.mlp, "classes_"):
            self.mlp.partial_fit(Xs, z_arr, classes=np.array([0, 1], dtype=np.int32))
        else:
            self.mlp.partial_fit(Xs, z_arr)
        self._fitted = True
        return self

    def predict_defer_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("DeferralClassifier is not fitted")
        X_arr = np.asarray(X, dtype=np.float64)
        proba = self.mlp.predict_proba(self.scaler.transform(X_arr))
        if proba.shape[1] == 1:
            return np.zeros(X_arr.shape[0], dtype=np.float64)
        pos_col = list(self.mlp.classes_).index(1) if 1 in self.mlp.classes_ else 0
        return proba[:, pos_col].astype(np.float64, copy=False)


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_arr = np.asarray(y_true, dtype=np.int32).reshape(-1)
    s_arr = np.asarray(score, dtype=np.float64).reshape(-1)
    if y_arr.size == 0 or np.unique(y_arr).size < 2:
        return float("nan")
    return float(roc_auc_score(y_arr, s_arr))


def _candidate_grid(
    hidden_dims_grid: Optional[Sequence[Sequence[int]]] = None,
    alpha_grid: Optional[Sequence[float]] = None,
    classifier_kind: str = "mlp",
) -> List[Dict[str, Any]]:
    if classifier_kind == "lr":
        # Logistic-regression router (paper §3.4): grid over inverse-reg C.
        return [{"C": float(c)} for c in (0.25, 1.0, 4.0)]
    hidden = hidden_dims_grid or ((16,), (32, 16), (64, 32))
    alphas = alpha_grid or (1e-3, 1e-2)
    return [
        {"hidden_dims": tuple(int(x) for x in dims), "alpha": float(alpha)}
        for dims in hidden
        for alpha in alphas
    ]


def _make_deferral_classifier(
    cfg: Dict[str, Any],
    classifier_kind: str,
    *,
    random_state: int,
    warm_start: bool = False,
    max_iter: int = 300,
):
    """Build the deferral/router classifier. The paper specifies logistic
    regression; ``classifier_kind="mlp"`` keeps the legacy MLP available."""
    if classifier_kind == "lr":
        return LRDeferralClassifier(
            C=float(cfg.get("C", 1.0)), random_state=int(random_state),
        )
    return DeferralClassifier(
        hidden_dims=cfg["hidden_dims"],
        alpha=cfg["alpha"],
        random_state=int(random_state),
        warm_start=bool(warm_start),
        max_iter=int(max_iter),
    )


def _inner_cv_select(
    X: np.ndarray,
    z: np.ndarray,
    *,
    candidates: Sequence[Dict[str, Any]],
    n_splits: int,
    random_state: int,
    classifier_kind: str = "mlp",
) -> Tuple[Dict[str, Any], float, float]:
    z_arr = np.asarray(z, dtype=np.int32).reshape(-1)
    counts = np.bincount(z_arr, minlength=2)
    min_count = int(counts.min())
    if min_count < 2 or z_arr.shape[0] < 6:
        return dict(candidates[0]), float("nan"), float("nan")
    k = max(2, min(int(n_splits), min_count))
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    best_cfg = dict(candidates[0])
    best_mean = -float("inf")
    best_std = float("nan")
    for cfg in candidates:
        scores: List[float] = []
        for train_idx, val_idx in splitter.split(X, z_arr):
            clf = _make_deferral_classifier(
                cfg, classifier_kind,
                random_state=random_state, warm_start=False, max_iter=300,
            ).fit(X[train_idx], z_arr[train_idx])
            proba = clf.predict_defer_proba(X[val_idx])
            auc = _safe_auc(z_arr[val_idx], proba)
            if not np.isnan(auc):
                scores.append(float(auc))
        if not scores:
            continue
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        if mean > best_mean:
            best_mean = mean
            best_std = std
            best_cfg = dict(cfg)
    if best_mean == -float("inf"):
        best_mean = float("nan")
    return best_cfg, float(best_mean), float(best_std)


def _heuristic_defer(
    z_smooth: np.ndarray,
    threshold_proxy: float,
    *,
    fallback_band: float = 0.10,
    reason: str,
) -> DeferralDecision:
    z_arr = _as_1d_float(z_smooth, "z_smooth")
    dist = np.abs(z_arr - float(threshold_proxy))
    band = max(float(fallback_band), 1e-6)
    defer_proba = np.clip(1.0 - dist / band, 0.0, 1.0)
    defer_mask = dist < band
    proxy_pred = z_arr >= float(threshold_proxy)
    info = {
        "method": "learned_deferral_heuristic",
        "fallback_reason": reason,
        "fallback_band": float(band),
        "defer_rate": float(defer_mask.mean()) if defer_mask.size else 0.0,
        "feature_dim": 1,
    }
    return DeferralDecision(
        defer_proba=defer_proba,
        defer_mask=defer_mask,
        proxy_pred=proxy_pred,
        classifier=None,
        info=info,
    )


def fit_deferral_classifier(
    z_smooth_full: np.ndarray,
    z_corrected_full: Optional[np.ndarray],
    similarity_full: Optional[np.ndarray],
    W_full: Any,
    sorted_idx_full: Optional[np.ndarray],
    z_smooth_oof: np.ndarray,
    z_corrected_oof: Optional[np.ndarray],
    labeled_indices: np.ndarray,
    labeled_values: np.ndarray,
    *,
    threshold_proxy: float = 0.5,
    feature_names: Optional[Sequence[str]] = None,
    hidden_dims_grid: Optional[Sequence[Sequence[int]]] = None,
    alpha_grid: Optional[Sequence[float]] = None,
    inner_cv_k: int = 3,
    min_inner_cv_auc: float = 0.60,
    probe_indices: Optional[np.ndarray] = None,
    probe_labels: Optional[np.ndarray] = None,
    min_probe_auc: float = 0.60,
    fallback_band: float = 0.10,
    random_state: int = 42,
    classifier_kind: str = "lr",
) -> DeferralDecision:
    """Train the learned deferral rule on OOF labelled-row proxy outputs.

    Training rows use OOF scores to avoid Bias Diffusion + Homogeneity Propagation pinning bias. Deployment rows
    use the full-run scores because those are the scores seen at inference.
    """
    z_full = _as_1d_float(z_smooth_full, "z_smooth_full")
    n_total = z_full.shape[0]
    idx = np.asarray(labeled_indices, dtype=np.int64).reshape(-1)
    y = np.asarray(labeled_values, dtype=np.int32).reshape(-1)
    if idx.shape[0] != y.shape[0]:
        raise ValueError("labeled_indices and labeled_values length mismatch")
    z_oof = _as_1d_float(z_smooth_oof, "z_smooth_oof")
    if z_oof.shape[0] != idx.shape[0]:
        raise ValueError("z_smooth_oof must have one entry per labelled row")
    if z_corrected_oof is None:
        z_corr_oof = z_oof.copy()
    else:
        z_corr_oof = _as_1d_float(z_corrected_oof, "z_corrected_oof")
        if z_corr_oof.shape[0] != idx.shape[0]:
            raise ValueError("z_corrected_oof must have one entry per labelled row")

    names = _normalise_feature_names(feature_names)
    full_features = DeferralFeatures.build(
        z_full, z_corrected_full, similarity_full, W_full, sorted_idx_full
    )
    full_stack = full_features.stack(names)
    train_stack = full_stack[idx].copy()
    feature_pos = {name: pos for pos, name in enumerate(names)}
    if "z_smooth" in feature_pos:
        train_stack[:, feature_pos["z_smooth"]] = z_oof
    if "z_corrected" in feature_pos:
        train_stack[:, feature_pos["z_corrected"]] = z_corr_oof
    if "boundary_dist" in feature_pos:
        train_stack[:, feature_pos["boundary_dist"]] = np.abs(
            z_oof - float(threshold_proxy)
        )

    proxy_pred_oof = (z_oof >= float(threshold_proxy)).astype(np.int32)
    z_train = (proxy_pred_oof != y.astype(np.int32)).astype(np.int32)
    n_wrong = int(z_train.sum())
    n_right = int((1 - z_train).sum())
    if n_wrong < 4 or n_right < 4:
        decision = _heuristic_defer(
            z_full,
            threshold_proxy,
            fallback_band=fallback_band,
            reason="cold_start_deferral_target",
        )
        decision.info.update(
            {
                "n_oof_train": int(idx.shape[0]),
                "n_train_wrong": n_wrong,
                "n_train_right": n_right,
                "features": list(names),
            }
        )
        return decision

    candidates = _candidate_grid(
        hidden_dims_grid, alpha_grid, classifier_kind=classifier_kind,
    )
    chosen_cfg, inner_auc, inner_auc_std = _inner_cv_select(
        train_stack,
        z_train,
        candidates=candidates,
        n_splits=int(inner_cv_k),
        random_state=int(random_state),
        classifier_kind=classifier_kind,
    )
    if not np.isnan(inner_auc) and inner_auc < float(min_inner_cv_auc):
        decision = _heuristic_defer(
            z_full,
            threshold_proxy,
            fallback_band=fallback_band,
            reason="inner_cv_auc_below_min",
        )
        decision.info.update(
            {
                "n_oof_train": int(idx.shape[0]),
                "n_train_wrong": n_wrong,
                "n_train_right": n_right,
                "inner_cv_auc": float(inner_auc),
                "inner_cv_auc_std": float(inner_auc_std),
                "min_inner_cv_auc": float(min_inner_cv_auc),
                "features": list(names),
            }
        )
        return decision

    clf = _make_deferral_classifier(
        chosen_cfg, classifier_kind,
        random_state=int(random_state), warm_start=True, max_iter=400,
    ).fit(train_stack, z_train)
    defer_proba = clf.predict_defer_proba(full_stack)
    defer_mask = defer_proba > 0.5
    proxy_pred = z_full >= float(threshold_proxy)

    info: Dict[str, Any] = {
        "method": "learned_deferral_oof",
        "n_oof_train": int(idx.shape[0]),
        "n_train_wrong": n_wrong,
        "n_train_right": n_right,
        "defer_rate": float(defer_mask.mean()) if defer_mask.size else 0.0,
        "feature_dim": int(train_stack.shape[1]),
        "features": list(names),
        "classifier_kind": classifier_kind,
        "chosen_config": dict(chosen_cfg),
        "chosen_hidden_dims": (
            tuple(chosen_cfg["hidden_dims"]) if "hidden_dims" in chosen_cfg else None
        ),
        "chosen_alpha": (
            float(chosen_cfg["alpha"]) if "alpha" in chosen_cfg else None
        ),
        "inner_cv_auc": float(inner_auc),
        "inner_cv_auc_std": float(inner_auc_std),
        "min_inner_cv_auc": float(min_inner_cv_auc),
        "feature_min": {
            name: float(np.min(full_stack[:, pos])) for pos, name in enumerate(names)
        },
        "feature_max": {
            name: float(np.max(full_stack[:, pos])) for pos, name in enumerate(names)
        },
    }

    if probe_indices is not None and probe_labels is not None:
        pidx = np.asarray(probe_indices, dtype=np.int64).reshape(-1)
        py = np.asarray(probe_labels, dtype=np.int32).reshape(-1)
        if pidx.shape[0] == py.shape[0] and pidx.shape[0] > 0:
            p_proba = defer_proba[pidx]
            p_proxy_correct = proxy_pred[pidx].astype(np.int32) == py.astype(np.int32)
            p_target = (~p_proxy_correct).astype(np.int32)
            p_auc = _safe_auc(p_target, p_proba)
            confidence_baseline = -np.abs(
                z_full[pidx] - float(threshold_proxy)
            )
            baseline_auc = _safe_auc(p_target, confidence_baseline)
            no_defer = p_proba <= 0.5
            p_selective_acc = (
                float(p_proxy_correct[no_defer].mean())
                if np.any(no_defer)
                else float("nan")
            )
            info.update(
                {
                    "probe_n": int(pidx.shape[0]),
                    "probe_auc": float(p_auc),
                    "probe_confidence_baseline_auc": float(baseline_auc),
                    "probe_defer_rate": float((p_proba > 0.5).mean()),
                    "probe_selective_acc": p_selective_acc,
                    "min_probe_auc": float(min_probe_auc),
                }
            )
            if not np.isnan(p_auc) and p_auc < float(min_probe_auc):
                fallback = _heuristic_defer(
                    z_full,
                    threshold_proxy,
                    fallback_band=fallback_band,
                    reason="probe_auc_below_min",
                )
                fallback.info.update(info)
                fallback.info["method"] = "learned_deferral_heuristic"
                fallback.info["fallback_reason"] = "probe_auc_below_min"
                return fallback

    return DeferralDecision(
        defer_proba=defer_proba,
        defer_mask=defer_mask,
        proxy_pred=proxy_pred,
        classifier=clf,
        info=info,
    )


def stratified_proxy_router_split(
    labeled_values: np.ndarray,
    *,
    router_ratio: float = 0.4,
    min_router_pos: int = 2,
    min_router_neg: int = 1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Split the actively-sampled labels once into L_proxy and L_router.

    Paper §3.4 (online router): the labels are stratified by expert-model
    label and split once into a proxy partition ``L_proxy`` (trains the base
    proxy and pins label propagation) and a held-out router partition
    ``L_router`` (trains the router). We allocate ``router_ratio`` of each
    class to ``L_router`` (the paper uses a fixed 6:4 split, i.e.
    ``router_ratio=0.4``), preserving the positive:negative ratio, with a
    floor of ``min_router_pos`` positives (and ``min_router_neg`` negatives)
    in ``L_router``.

    Returns boolean masks ``(proxy_mask, router_mask)`` over the labeled rows
    plus an info dict. When the label counts cannot satisfy the floors while
    leaving at least one positive and one negative for ``L_proxy``, the split
    is infeasible: ``proxy_mask`` is all-True, ``router_mask`` all-False, and
    ``info["feasible"]`` is False (the caller should fall back to the OOF
    path that reuses the full labeled set for both proxy and router).
    """
    y = np.asarray(labeled_values, dtype=np.int32).reshape(-1)
    n = int(y.shape[0])
    proxy_mask = np.ones(n, dtype=bool)
    router_mask = np.zeros(n, dtype=bool)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos, n_neg = int(pos_idx.shape[0]), int(neg_idx.shape[0])
    info: Dict[str, Any] = {
        "router_ratio": float(router_ratio),
        "n_total_labeled": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "min_router_pos": int(min_router_pos),
        "min_router_neg": int(min_router_neg),
    }

    n_router_pos = max(int(min_router_pos), int(round(float(router_ratio) * n_pos)))
    n_router_neg = max(int(min_router_neg), int(round(float(router_ratio) * n_neg)))
    if (
        n_router_pos < int(min_router_pos)
        or n_router_neg < int(min_router_neg)
        or n_pos - n_router_pos < 1
        or n_neg - n_router_neg < 1
    ):
        info["feasible"] = False
        info["fallback_reason"] = (
            f"insufficient labels for stratified {1 - router_ratio:.0%}:"
            f"{router_ratio:.0%} split (pos={n_pos}, neg={n_neg})"
        )
        info["n_proxy"] = n
        info["n_router"] = 0
        return proxy_mask, router_mask, info

    rng = np.random.default_rng(int(random_state))
    router_sel = np.concatenate(
        [
            rng.choice(pos_idx, size=n_router_pos, replace=False),
            rng.choice(neg_idx, size=n_router_neg, replace=False),
        ]
    )
    router_mask[router_sel] = True
    proxy_mask[router_sel] = False
    info.update(
        {
            "feasible": True,
            "n_proxy": int(proxy_mask.sum()),
            "n_router": int(router_mask.sum()),
            "n_router_pos": n_router_pos,
            "n_router_neg": n_router_neg,
            "n_proxy_pos": n_pos - n_router_pos,
            "n_proxy_neg": n_neg - n_router_neg,
        }
    )
    return proxy_mask, router_mask, info


def build_online_deferral_features(
    positive_proba: np.ndarray,
    *,
    indices: Optional[np.ndarray] = None,
    total: Optional[int] = None,
) -> np.ndarray:
    """Build six-column deferral features for the LR-only online path."""
    p = _as_1d_float(positive_proba, "positive_proba")
    n = p.shape[0]
    if indices is None or total is None:
        rank = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    else:
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        rank = idx.astype(np.float64) / max(int(total) - 1, 1)
    zeros = np.zeros(n, dtype=np.float64)
    return np.column_stack([p, p, zeros, zeros, rank, np.abs(p - 0.5)])


class LRDeferralClassifier:
    """Logistic Regression deferral classifier with feature scaling.

    Drop-in alternative to ``DeferralClassifier`` (MLP). Used by the DASP
    path because the classifier is fit on a small disjoint partition
    (B_defer) where a linear model is more stable than an MLP, and
    because LR's class-weighting handles the rare-positive class without
    sample reweighting.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        max_iter: int = 200,
    ):
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(
            C=float(C),
            class_weight=class_weight,
            random_state=int(random_state),
            max_iter=int(max_iter),
            solver="lbfgs",
        )
        self._fitted = False

    def fit(self, X: np.ndarray, z: np.ndarray) -> "LRDeferralClassifier":
        X_arr = np.asarray(X, dtype=np.float64)
        z_arr = np.asarray(z, dtype=np.int32).reshape(-1)
        self.scaler.fit(X_arr)
        self.lr.fit(self.scaler.transform(X_arr), z_arr)
        self._fitted = True
        return self

    def predict_defer_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LRDeferralClassifier is not fitted")
        X_arr = np.asarray(X, dtype=np.float64)
        proba = self.lr.predict_proba(self.scaler.transform(X_arr))
        if proba.shape[1] == 1:
            return np.zeros(X_arr.shape[0], dtype=np.float64)
        pos_col = list(self.lr.classes_).index(1) if 1 in self.lr.classes_ else 0
        return proba[:, pos_col].astype(np.float64, copy=False)



__all__ = [
    "DEFAULT_DEFERRAL_FEATURES",
    "DeferralClassifier",
    "DeferralDecision",
    "DeferralFeatures",
    "LRDeferralClassifier",
    "build_deferral_features",
    "build_online_deferral_features",
    "fit_deferral_classifier",
    "stratified_proxy_router_split",
]
