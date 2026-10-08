"""Evaluation-only oracle audit for the learned-calibration stage.

After routing, an audit sample of the still-unverified rows is labelled by the same oracle (same
request factory, prompts and chat template as the run) on a fresh, unlimited ledger. The labels
never reach the run's decisions or its call accounting; they only let us draw reliability diagrams
and compute ECE of the raw and calibrated scores against oracle labels (GAMCAL, arXiv:2604.00660,
Fig. 3 / Table 1). Runs with the audit on are for calibration plots only: their wall clock includes
the audit calls.

Enabled by LASO_LC_AUDIT_DIR (output directory) and LASO_LC_AUDIT_N (sample size, default 3000).
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Tuple

import numpy as np

from oracle_budget import BudgetedOracle, OracleLedger

__all__ = ["stratified_audit_sample", "audit_labels", "weighted_ece", "weighted_brier", "write_audit"]


def stratified_audit_sample(pool: np.ndarray, x: np.ndarray, n: int, *, n_bins: int = 20,
                            seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Rows of ``pool`` to audit and their inverse-inclusion weights.

    The whole pool when it has at most ``n`` rows (weight 1). Otherwise equal allocation over
    ``n_bins`` quantile strata of the proxy logit ``x``, so the rare high-score region, where the
    positives are, is represented; each row's weight is stratum size / rows drawn from it.
    """
    pool = np.asarray(pool, dtype=np.int64).reshape(-1)
    if pool.size <= n:
        return pool.copy(), np.ones(pool.size, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    order = pool[np.argsort(np.asarray(x, dtype=np.float64)[pool], kind="stable")]
    strata = [s for s in np.array_split(order, min(n_bins, n)) if s.size]
    quota = np.full(len(strata), n // len(strata))
    quota[: n - int(quota.sum())] += 1
    idx, w = [], []
    for rows, k in zip(strata, quota):
        k = int(min(k, rows.size))
        idx.append(rng.choice(rows, size=k, replace=False))
        w.append(np.full(k, rows.size / k))
    return np.concatenate(idx), np.concatenate(w)


def audit_labels(session: Any, ids: np.ndarray) -> np.ndarray:
    """Oracle labels for ``ids`` through the run's own request factory, on a separate ledger.

    Unresolved answers come back as -1 and are dropped by the metrics.
    """
    ids = [int(i) for i in np.asarray(ids).reshape(-1)]
    ledger = OracleLedger(limit=len(ids))

    async def execute():
        async with session.request_factory() as request:
            oracle = BudgetedOracle(ledger, request, concurrency=int(getattr(session, "concurrency", 32)))
            return await oracle.label(ids, stage="audit")

    values = asyncio.run(execute())
    return np.array([-1 if v is None else int(v) for v in values], dtype=np.int32)


def _valid(p, y, w):
    p, y, w = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (p, y, w))
    keep = (y == 0) | (y == 1)
    return p[keep], y[keep], w[keep]


def weighted_ece(p, y, w, *, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width probability bins and sampling weights."""
    p, y, w = _valid(p, y, w)
    if w.sum() <= 0:
        return float("nan")
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    err = 0.0
    for k in np.unique(b):
        m = b == k
        wk = w[m].sum()
        err += wk * abs(np.dot(w[m], p[m]) / wk - np.dot(w[m], y[m]) / wk)
    return float(err / w.sum())


def weighted_brier(p, y, w) -> float:
    p, y, w = _valid(p, y, w)
    return float(np.dot(w, (p - y) ** 2) / w.sum()) if w.sum() > 0 else float("nan")


def write_audit(out_dir: str, tag: str, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> str:
    """One compressed .npz per run; ``meta`` values are stored as 0-d arrays."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}.npz")
    np.savez_compressed(path, **arrays, **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})
    return path


def run_audit(*, result: Any, x: np.ndarray, h: np.ndarray, labeled_indices: np.ndarray,
              labeled_values: np.ndarray, session: Any, config: Any, out_dir: str, n: int,
              x_cal: Any = None) -> Dict[str, Any]:
    """Audit one learned-calibration run and write its .npz; returns the lc_audit_* diagnostics.

    ``x`` is the LR proxy logit (the audit sample is stratified on it); ``x_cal`` is the calibrator input when
    it differs (calib_input="refined"), stored as ``x_cal`` and used for ``p_cal``."""
    from weighted_calibrator import predict_proba

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    xc = x if x_cal is None else np.asarray(x_cal, dtype=np.float64).reshape(-1)
    l_idx = np.asarray(labeled_indices, dtype=np.int64).reshape(-1)
    verified = np.zeros(x.size, dtype=bool)
    verified[l_idx] = True
    verified[result.calib_idx] = True
    verified[result.verify_idx] = True
    idx, w = stratified_audit_sample(np.flatnonzero(~verified), x, int(n), seed=int(config.seed))
    t0 = time.perf_counter()
    y = audit_labels(session, idx)
    elapsed = time.perf_counter() - t0
    params = result.params
    if params is None:
        p_cal = np.full(x.size, np.nan)
    elif hasattr(params, "predict"):              # GAM arm
        p_cal = params.predict(xc, h)
    else:                                         # Platt arm (identity for calibrator="none")
        p_cal = predict_proba(params, xc, h)
    arrays = dict(
        x=x, x_cal=xc, h=h, p_cal=p_cal, final_pos=np.asarray(result.final_pos, dtype=bool),
        L_idx=l_idx, L_y=np.asarray(labeled_values, dtype=np.int32),
        A_idx=result.calib_idx, A_y=result.final_pos[result.calib_idx].astype(np.int32),
        D_idx=result.verify_idx, D_y=result.final_pos[result.verify_idx].astype(np.int32),
        audit_idx=idx, audit_y=y, audit_w=w,
    )
    meta = dict(calibrator=str(config.calibrator), calib_input=str(getattr(config, "calib_input", "proxy")), alpha=float(config.alpha), seed=int(config.seed),
                n_rows=int(x.size), status=str(result.status))
    for key in ("a", "b", "c", "delta", "t"):
        if params is not None and hasattr(params, key):
            meta[f"platt_{key}"] = float(getattr(params, key))
    path = write_audit(out_dir, f"n{x.size}", arrays, meta)
    return {"lc_audit_path": path, "lc_audit_n": int((y >= 0).sum()), "lc_audit_time_s": float(elapsed)}
