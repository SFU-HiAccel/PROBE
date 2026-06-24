import hashlib
import logging
import math
import os
import pickle
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

LabelFnType = Callable[[List[Union[str, list]]], np.ndarray]
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg as conjugate_gradient
from scipy.stats import beta as beta_dist
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors


logger = logging.getLogger("proxy_model")

# ---------------------------------------------------------------------------
# KNN graph disk cache
#
# Same recipe as proxy_model's embedding cache: a sampled content-hash of the
# input embeddings + (k, sigma) forms the cache key, pickled sparse matrices
# live under LLMSQL_KNN_CACHE_DIR. Survives process restarts so the grid
# re-uses the (slow) neighbour search across ablation cells that share the
# same pair-embedding matrix.
# ---------------------------------------------------------------------------
_KNN_CACHE_VERSION = os.environ.get("LLMSQL_KNN_CACHE_VERSION", "v1")
_KNN_CACHE_ENABLED = os.environ.get(
    "LLMSQL_KNN_CACHE", "1"
).strip() not in ("0", "false", "False")
_KNN_CACHE_DIR = os.environ.get(
    "LLMSQL_KNN_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "llmsql", "knn_graphs"),
)


def _embeddings_fingerprint(embeddings: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(embeddings.shape).encode())
    h.update(b"|")
    h.update(str(embeddings.dtype).encode())
    h.update(b"|")
    n = int(embeddings.shape[0]) if embeddings.ndim > 0 else 0
    if n > 0:
        idxs = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
        for i in idxs:
            h.update(np.ascontiguousarray(embeddings[i]).tobytes())
    return h.hexdigest()


def _knn_cache_path(fingerprint: str, knn_k: int, sigma: float, viz_tag: str) -> str:
    key_str = f"{_KNN_CACHE_VERSION}|{fingerprint}|k={int(knn_k)}|sigma={float(sigma):.6g}"
    digest = hashlib.sha256(key_str.encode()).hexdigest()
    safe_tag = re.sub(r"[^\w\-.]", "_", str(viz_tag))[:64] if viz_tag else ""
    name = f"{safe_tag}_{digest}" if safe_tag else digest
    return os.path.join(_KNN_CACHE_DIR, name + ".pkl")


def _knn_cache_get(path: str) -> Optional[csr_matrix]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            W = pickle.load(f)
        if not isinstance(W, csr_matrix):
            return None
        return W
    except Exception as e:
        logger.warning(f"[knn_cache] load failed for {path}: {e}")
        return None


def _knn_cache_put(path: str, W: csr_matrix) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(W, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"[knn_cache] save failed: {e}")


# ---------------------------------------------------------------------------
# KNN search backend selection
#
# Backends (in preference order when LLMSQL_KNN_BACKEND=auto):
#   faiss_gpu  — faiss.IndexFlatL2 on GPU (fastest for 100k+ high-dim vectors)
#   faiss_cpu  — faiss.IndexFlatL2 on CPU (multithreaded, SIMD; 50-100x sklearn)
#   sklearn    — sklearn.neighbors.NearestNeighbors (fallback, always present)
#
# The NN search dominated ecomm_q7's 521s knn_build on sklearn brute force in
# 768-dim. FAISS uses blocked BLAS/cuBLAS GEMM to compute all pairwise L2
# distances and a top-k reduction, which parallelises trivially.
#
# Env vars:
#   LLMSQL_KNN_BACKEND   one of {auto, faiss_gpu, faiss_cpu, sklearn} (default auto)
#   LLMSQL_FAISS_GPU_ID  GPU device id for faiss_gpu (default 0)
# ---------------------------------------------------------------------------
# Wrap in a broad try/except — a broken faiss install (e.g. mismatched
# CPU/GPU swig wrappers that raise AttributeError during module init) must
# not take down cluster_proxy.py. We degrade to sklearn and surface a
# warning so the user can fix the faiss env without losing the pipeline.
_FAISS_IMPORT_ERROR: Optional[BaseException] = None
try:
    import faiss as _faiss  # type: ignore
    _FAISS_CPU_AVAILABLE = True
except BaseException as _e:  # ImportError, AttributeError, RuntimeError, ...
    _faiss = None  # type: ignore
    _FAISS_CPU_AVAILABLE = False
    _FAISS_IMPORT_ERROR = _e

_FAISS_GPU_AVAILABLE = False
if _FAISS_CPU_AVAILABLE:
    try:
        _FAISS_GPU_AVAILABLE = (
            hasattr(_faiss, "StandardGpuResources")
            and hasattr(_faiss, "get_num_gpus")
            and int(_faiss.get_num_gpus()) > 0
        )
    except Exception:
        _FAISS_GPU_AVAILABLE = False

if _FAISS_IMPORT_ERROR is not None:
    logger.warning(
        "[knn] faiss import failed (%s: %s) — falling back to sklearn. "
        "If you meant to use the GPU backend, repair the faiss install: "
        "uninstall every faiss variant (faiss / faiss-cpu / faiss-gpu / "
        "faiss-gpu-cu12) via BOTH pip and conda, then reinstall exactly one "
        "— e.g. `conda install -c pytorch -c nvidia faiss-gpu=1.8.0` or "
        "`pip install faiss-gpu-cu12`. Workaround: `export "
        "FAISS_DISABLE_CPU_FEATURES=AVX512` to skip the broken AVX512 dispatcher.",
        type(_FAISS_IMPORT_ERROR).__name__, _FAISS_IMPORT_ERROR,
    )

_KNN_BACKEND = os.environ.get("LLMSQL_KNN_BACKEND", "auto").strip().lower()
_FAISS_GPU_ID = int(os.environ.get("LLMSQL_FAISS_GPU_ID", "0"))


def _select_knn_backend(requested: str = _KNN_BACKEND) -> str:
    """Resolve the effective backend name from the user/env request."""
    r = (requested or "auto").strip().lower()
    if r == "auto":
        if _FAISS_GPU_AVAILABLE:
            return "faiss_gpu"
        if _FAISS_CPU_AVAILABLE:
            return "faiss_cpu"
        return "sklearn"
    if r == "faiss_gpu":
        if _FAISS_GPU_AVAILABLE:
            return "faiss_gpu"
        logger.warning(
            "[knn] LLMSQL_KNN_BACKEND=faiss_gpu requested but not available "
            "(faiss_cpu=%s, gpu=%s); falling back.",
            _FAISS_CPU_AVAILABLE, _FAISS_GPU_AVAILABLE,
        )
        return "faiss_cpu" if _FAISS_CPU_AVAILABLE else "sklearn"
    if r == "faiss_cpu":
        if _FAISS_CPU_AVAILABLE:
            return "faiss_cpu"
        logger.warning("[knn] LLMSQL_KNN_BACKEND=faiss_cpu requested but faiss is not installed; falling back to sklearn.")
        return "sklearn"
    if r == "sklearn":
        return "sklearn"
    logger.warning("[knn] unknown LLMSQL_KNN_BACKEND=%r; using auto.", requested)
    return _select_knn_backend("auto")


def _faiss_knn_search(
    embeddings: np.ndarray, k_plus_1: int, use_gpu: bool
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Run a FAISS IndexFlatL2 brute-force kNN search.

    Returns (distances, indices, fit_time, query_time) with distances in
    euclidean (not squared) L2 to match sklearn's convention — the caller
    downstream squares them again for the Gaussian kernel, but the bandwidth
    sigma = median(kth euclidean distance) needs non-squared values.
    """
    x = np.ascontiguousarray(embeddings, dtype=np.float32)
    d = int(x.shape[1])

    t0 = time.time()
    index = _faiss.IndexFlatL2(d)
    if use_gpu:
        res = _faiss.StandardGpuResources()
        index = _faiss.index_cpu_to_gpu(res, _FAISS_GPU_ID, index)
    index.add(x)
    fit_time = time.time() - t0

    t0 = time.time()
    dist_sq, indices = index.search(x, int(k_plus_1))
    query_time = time.time() - t0

    distances = np.sqrt(np.clip(dist_sq, 0.0, None))
    return distances, indices.astype(np.int64, copy=False), fit_time, query_time


def _sklearn_knn_search(
    embeddings: np.ndarray, k_plus_1: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    # n_jobs=-1: parallelize the brute-force kneighbors query across all
    # cores. Single-threaded sklearn on N>50k×768 dims takes minutes;
    # this drops it to ~cores/1.5x faster. Real fix is faiss-gpu, but
    # this saves the user when faiss is missing.
    t0 = time.time()
    nn = NearestNeighbors(
        n_neighbors=int(k_plus_1), metric="euclidean",
        algorithm="auto", n_jobs=-1,
    )
    nn.fit(embeddings)
    fit_time = time.time() - t0
    t0 = time.time()
    distances, indices = nn.kneighbors(embeddings)
    query_time = time.time() - t0
    return distances, indices, fit_time, query_time


# Backend availability is logged lazily on the first _build_knn_graph
# call (see cluster_proxy.py around the build site). Emitting it at
# import time was misleading: it would appear during active-learning sampling iterative
# sampling — which doesn't touch KNN at all — and look like KNN work
# was starting prematurely. Use logger.debug here for env-debugging:
logger.debug(
    "[knn] backend availability (import-time): faiss_cpu=%s, faiss_gpu=%s "
    "(gpu_id=%s), requested=%s, effective=%s",
    _FAISS_CPU_AVAILABLE, _FAISS_GPU_AVAILABLE, _FAISS_GPU_ID,
    _KNN_BACKEND, _select_knn_backend(),
)
_KNN_BACKEND_LOGGED = False

CLUSTER_PURE_POS = "pure_positive"
CLUSTER_PURE_NEG = "pure_negative"
CLUSTER_BOUNDARY = "boundary"


# ---------------------------------------------------------------------------
# Phase A: Core utilities
# ---------------------------------------------------------------------------


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster embeddings using KMeans.

    Args:
        embeddings: (N, d) array.
        n_clusters: Number of clusters. 0 = auto heuristic.
        random_state: Reproducibility seed.

    Returns:
        cluster_labels: (N,) cluster assignment per row.
        centroids: (K, d) cluster centroids.
    """
    n = embeddings.shape[0]
    if n_clusters <= 0:
        k = auto_kmeans_k(n)
    else:
        k = n_clusters
    k = min(k, n)

    if n > 10_000:
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=1024)
    else:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)

    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_

    sizes = np.bincount(labels, minlength=k)
    logger.info(
        f"KMeans: K={k}, cluster sizes min={int(sizes.min())} "
        f"max={int(sizes.max())} mean={float(sizes.mean()):.1f}"
    )
    return labels.astype(np.int32), centroids.astype(np.float32)




# ---------------------------------------------------------------------------
# Phase D: Label Propagation (CutSSL-inspired)
# ---------------------------------------------------------------------------


def _build_knn_graph(
    embeddings: np.ndarray,
    knn_k: int = 15,
    sigma: float = 0.0,
    viz_tag: str = "",
    cache: bool = True,
    sub_timings: Optional[Dict[str, float]] = None,
) -> csr_matrix:
    """Build a symmetric KNN weight matrix.

    Args:
        embeddings: (N, d) array.
        knn_k: Number of nearest neighbors.
        sigma: Gaussian kernel bandwidth. 0 = auto (median of k-th NN distances).
        viz_tag: Optional query/dataset tag used to make the disk-cache filename
            grep-able (e.g. "ecomm_q7_...").
        cache: If True (default) and LLMSQL_KNN_CACHE is enabled, look up /
            populate the on-disk cache. Set False to bypass (e.g. for benchmarks).
        sub_timings: Optional out-dict. When provided, the function writes fine-
            grained timings ( cache_lookup_time, nn_fit_time, nn_query_time,
            csr_build_time, symmetrize_time, cache_save_time, cache_hit, nnz,
            n_samples, knn_k, sigma_used ) into it so callers can surface
            them in profiling / logs.

    Returns:
        Sparse symmetric weight matrix W of shape (N, N).
    """
    if sub_timings is None:
        sub_timings = {}

    n = int(embeddings.shape[0])
    d = int(embeddings.shape[1]) if embeddings.ndim > 1 else 1
    k = min(int(knn_k), n - 1)

    sub_timings["n_samples"] = n
    sub_timings["dim"] = d
    sub_timings["knn_k"] = k
    sub_timings.setdefault("cache_hit", False)
    for _key in (
        "cache_lookup_time", "nn_fit_time", "nn_query_time",
        "csr_build_time", "symmetrize_time", "cache_save_time",
    ):
        sub_timings.setdefault(_key, 0.0)

    if k <= 0:
        sub_timings["nnz"] = 0
        return csr_matrix((n, n), dtype=np.float64)

    # --- cache lookup ---------------------------------------------------
    cache_path: Optional[str] = None
    if cache and _KNN_CACHE_ENABLED:
        t0 = time.time()
        fingerprint = _embeddings_fingerprint(embeddings)
        cache_path = _knn_cache_path(fingerprint, k, float(sigma), viz_tag)
        W_cached = _knn_cache_get(cache_path)
        sub_timings["cache_lookup_time"] = time.time() - t0
        if W_cached is not None and W_cached.shape == (n, n):
            sub_timings["cache_hit"] = True
            sub_timings["nnz"] = int(W_cached.nnz)
            fname = os.path.basename(cache_path)
            logger.info(
                f"[knn_cache] DISK HIT {fname[:80]} "
                f"(N={n}, d={d}, k={k}, nnz={W_cached.nnz}, "
                f"lookup={sub_timings['cache_lookup_time']:.3f}s)"
            )
            return W_cached

    backend = _select_knn_backend()
    # Emit backend-availability summary the first time we actually build a
    # KNN graph. Keeps the active-learning sampling iterative sampling log clean (sampling
    # doesn't touch KNN) and makes the diagnostic appear right next to
    # the "KNN graph build: ..." line where it's actionable.
    global _KNN_BACKEND_LOGGED
    if not _KNN_BACKEND_LOGGED:
        logger.info(
            "[knn] backend availability: faiss_cpu=%s, faiss_gpu=%s "
            "(gpu_id=%s), requested=%s, effective=%s",
            _FAISS_CPU_AVAILABLE, _FAISS_GPU_AVAILABLE, _FAISS_GPU_ID,
            _KNN_BACKEND, backend,
        )
        _KNN_BACKEND_LOGGED = True
    logger.info(
        f"KNN graph build: N={n}, d={d}, k={k}, backend={backend} "
        f"(cache {'miss' if cache and _KNN_CACHE_ENABLED else 'disabled'}, computing)..."
    )
    sub_timings["backend"] = backend

    # --- NN fit + query (via selected backend, with graceful fallback) --
    def _run_backend(name: str):
        if name == "faiss_gpu":
            return _faiss_knn_search(embeddings, k + 1, use_gpu=True)
        if name == "faiss_cpu":
            return _faiss_knn_search(embeddings, k + 1, use_gpu=False)
        return _sklearn_knn_search(embeddings, k + 1)

    try:
        distances, indices, fit_time, query_time = _run_backend(backend)
    except Exception as e:
        # Typical failure modes: GPU OOM, faiss lib mismatch. Fall back
        # down the chain (faiss_gpu -> faiss_cpu -> sklearn).
        fallback = "sklearn"
        if backend == "faiss_gpu" and _FAISS_CPU_AVAILABLE:
            fallback = "faiss_cpu"
        logger.warning(
            "[knn] backend=%s failed (%s); falling back to %s",
            backend, type(e).__name__, fallback,
        )
        distances, indices, fit_time, query_time = _run_backend(fallback)
        sub_timings["backend"] = fallback
        sub_timings["backend_fallback_from"] = backend

    sub_timings["nn_fit_time"] = float(fit_time)
    sub_timings["nn_query_time"] = float(query_time)

    if sigma <= 0.0:
        med = float(np.median(distances[:, -1]))
        sigma_used = med if med > 1e-10 else 1.0
    else:
        sigma_used = float(sigma)
    sub_timings["sigma_used"] = sigma_used
    sigma_sq_2 = 2.0 * sigma_used * sigma_used

    # --- CSR build (vectorized; replaces ~N*k Python iterations) --------
    t0 = time.time()
    nbr_idx = indices[:, 1:k + 1]
    nbr_dist = distances[:, 1:k + 1]
    row_ind = np.repeat(np.arange(n, dtype=np.int64), k)
    col_ind = np.ascontiguousarray(nbr_idx, dtype=np.int64).ravel()
    d2 = np.square(np.ascontiguousarray(nbr_dist, dtype=np.float64).ravel())
    weights = np.exp(-d2 / sigma_sq_2)
    W = csr_matrix((weights, (row_ind, col_ind)), shape=(n, n))
    sub_timings["csr_build_time"] = time.time() - t0

    # --- Symmetrize -----------------------------------------------------
    t0 = time.time()
    W = (W + W.T) / 2.0
    if not isinstance(W, csr_matrix):
        W = W.tocsr()
    sub_timings["symmetrize_time"] = time.time() - t0
    sub_timings["nnz"] = int(W.nnz)

    # --- Cache save -----------------------------------------------------
    if cache_path is not None:
        t0 = time.time()
        _knn_cache_put(cache_path, W)
        sub_timings["cache_save_time"] = time.time() - t0

    logger.info(
        f"KNN graph build detail (backend={sub_timings.get('backend', 'sklearn')}, "
        f"N={n}, d={d}, k={k}, sigma={sigma_used:.4g}, nnz={W.nnz}): "
        f"cache_lookup={sub_timings['cache_lookup_time']:.3f}s  "
        f"nn_fit={sub_timings['nn_fit_time']:.3f}s  "
        f"nn_query={sub_timings['nn_query_time']:.3f}s  "
        f"csr_build={sub_timings['csr_build_time']:.3f}s  "
        f"symmetrize={sub_timings['symmetrize_time']:.3f}s  "
        f"cache_save={sub_timings['cache_save_time']:.3f}s"
    )

    return W


def _compute_graph_laplacian(W: csr_matrix) -> csr_matrix:
    """Compute combinatorial graph Laplacian L = D - W."""
    degree = np.array(W.sum(axis=1)).flatten()
    D = diags(degree, format="csr")
    return D - W


def _solve_constrained_propagation(
    L: csr_matrix,
    labeled_indices: np.ndarray,
    labeled_values: np.ndarray,
    class_prior: float,
    n_total: int,
    s_values: List[float],
    max_iter: int = 100,
) -> np.ndarray:
    """Solve cardinality-constrained label propagation.

    Uses a two-stage approach:
      1. Harmonic label propagation with labeled node clamping
      2. Cardinality-aware threshold adjustment

    Returns soft labels f in [0, 1]^N.
    """
    label_weight = 100.0
    labeled_mask = np.zeros(n_total, dtype=np.float64)
    labeled_mask[labeled_indices] = label_weight

    y = np.full(n_total, class_prior, dtype=np.float64)
    y[labeled_indices] = labeled_values.astype(np.float64)

    f = y.copy()

    for s in s_values:
        reg = max(s, 1e-6)
        A = L + diags(labeled_mask + reg, format="csr")
        rhs = labeled_mask * y + reg * class_prior

        f_new, info = conjugate_gradient(A, rhs, x0=f, maxiter=max_iter, rtol=1e-8)

        f_new = np.clip(f_new, 0.0, 1.0)
        f_new[labeled_indices] = labeled_values.astype(np.float64)

        target_sum = class_prior * n_total

        unlabeled_mask_bool = np.ones(n_total, dtype=bool)
        unlabeled_mask_bool[labeled_indices] = False
        unlabeled_f = f_new[unlabeled_mask_bool]

        if len(unlabeled_f) > 0:
            current_pos = float(np.sum(labeled_values)) + float(np.sum(unlabeled_f >= 0.5))
            if abs(current_pos - target_sum) > 1.0 and len(unlabeled_f) > 0:
                target_unlabeled_pos = max(
                    0, int(target_sum - float(np.sum(labeled_values)))
                )
                target_unlabeled_pos = min(target_unlabeled_pos, len(unlabeled_f))
                if target_unlabeled_pos > 0:
                    sorted_vals = np.sort(unlabeled_f)[::-1]
                    threshold = float(sorted_vals[min(target_unlabeled_pos, len(sorted_vals) - 1)])
                    threshold = max(threshold, 0.01)
                    scale = 0.5 / max(threshold, 1e-8)
                    f_new[unlabeled_mask_bool] = np.clip(unlabeled_f * scale, 0.0, 1.0)
                    f_new[labeled_indices] = labeled_values.astype(np.float64)

        f = f_new

    return f


# ---------------------------------------------------------------------------
# BaS-style similarity-stratified sampling
# ---------------------------------------------------------------------------


def _stratify_by_similarity(
    similarities: np.ndarray,
    n_strata: int = 10,
) -> List[np.ndarray]:
    """Sort candidates by similarity descending, split into K equal strata.

    strata[0] = highest similarity, strata[-1] = lowest.
    """
    n = len(similarities)
    n_strata = max(1, min(n_strata, n))
    sorted_idx = np.argsort(-similarities)
    return [chunk.copy() for chunk in np.array_split(sorted_idx, n_strata)]


def _importance_weighted_sample(
    candidates: np.ndarray,
    similarities: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_samples from candidates with P(x) proportional to similarity."""
    if len(candidates) == 0 or n_samples <= 0:
        return np.array([], dtype=np.int64)
    n_samples = min(n_samples, len(candidates))
    sims = similarities[candidates].copy()
    sims = sims - sims.min() + 1e-6
    probs = sims / sims.sum()
    chosen = rng.choice(len(candidates), size=n_samples, replace=False, p=probs)
    return candidates[chosen]


def _pilot_sample_strata(
    strata: List[np.ndarray],
    similarities: np.ndarray,
    texts: List[Union[str, list]],
    label_fn: LabelFnType,
    pilot_budget: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Importance-weighted pilot sampling across all strata.

    Budget per stratum is proportional to total similarity weight.
    """
    rng = np.random.default_rng(42)
    n_strata = len(strata)

    stratum_weights = []
    for members in strata:
        if len(members) == 0:
            stratum_weights.append(0.0)
        else:
            stratum_weights.append(float(np.sum(np.maximum(similarities[members], 0.0) + 1e-6)))
    total_weight = sum(stratum_weights) or 1.0

    allocations = []
    remaining = pilot_budget
    for s_id, members in enumerate(strata):
        if len(members) == 0:
            allocations.append(0)
            continue
        raw = max(1, round(pilot_budget * stratum_weights[s_id] / total_weight))
        raw = min(raw, len(members), remaining)
        allocations.append(raw)
        remaining -= raw

    all_indices: List[int] = []
    all_labels: List[int] = []
    stratum_stats: List[Dict[str, Any]] = []

    for s_id, members in enumerate(strata):
        n_alloc = allocations[s_id]
        if n_alloc == 0 or len(members) == 0:
            stratum_stats.append({
                "size": len(members), "n_sampled": 0, "n_pos": 0,
                "pos_rate": 0.0, "variance": 0.0, "weight": stratum_weights[s_id],
                "sim_min": 0.0, "sim_max": 0.0, "sim_mean": 0.0,
            })
            continue

        picked = _importance_weighted_sample(members, similarities, n_alloc, rng)
        if len(picked) == 0:
            stratum_stats.append({
                "size": len(members), "n_sampled": 0, "n_pos": 0,
                "pos_rate": 0.0, "variance": 0.0, "weight": stratum_weights[s_id],
                "sim_min": 0.0, "sim_max": 0.0, "sim_mean": 0.0,
            })
            continue

        labels = label_fn([texts[i] for i in picked])
        for idx, lbl in zip(picked.tolist(), labels.tolist()):
            all_indices.append(idx)
            all_labels.append(int(lbl))

        sims_picked = np.maximum(similarities[picked], 1e-6)
        labels_f = labels.astype(np.float64)
        x_i = labels_f / sims_picked
        var_x = float(np.var(x_i)) if len(x_i) > 1 else 0.0
        stratum_var = float(len(members)) ** 2 * var_x

        member_sims = similarities[members]
        n_pos = int(np.sum(labels == 1))
        stratum_stats.append({
            "size": len(members), "n_sampled": len(picked), "n_pos": n_pos,
            "pos_rate": n_pos / len(picked) if len(picked) > 0 else 0.0,
            "variance": stratum_var, "weight": stratum_weights[s_id],
            "sim_min": float(member_sims.min()),
            "sim_max": float(member_sims.max()),
            "sim_mean": float(member_sims.mean()),
        })

    pilot_indices = np.array(all_indices, dtype=np.int64)
    pilot_labels = np.array(all_labels, dtype=np.int32)

    for s_id, stats in enumerate(stratum_stats):
        logger.info(
            f"  [Stratum {s_id}] size={stats['size']}, sampled={stats['n_sampled']}, "
            f"pos={stats['n_pos']}, pos_rate={stats['pos_rate']:.3f}, "
            f"sim=[{stats['sim_min']:.4f}, {stats['sim_max']:.4f}], "
            f"mean_sim={stats['sim_mean']:.4f}, variance={stats['variance']:.2f}"
        )
    logger.info(
        f"Pilot sampling: {len(pilot_indices)} samples across {n_strata} strata, "
        f"{int(np.sum(pilot_labels == 1))} positives"
    )
    return pilot_indices, pilot_labels, stratum_stats


def _allocate_main_budget(
    stratum_stats: List[Dict[str, Any]],
    strata: List[np.ndarray],
    main_budget: int,
    already_sampled: int,
    method: str = "neyman",
) -> List[int]:
    """Allocate remaining budget to strata.

    "neyman": budget_i proportional to sqrt(variance) * size
    "proportional": budget_i proportional to similarity weight
    """
    n_strata = len(strata)
    scores = np.zeros(n_strata)

    for s_id in range(n_strata):
        stats = stratum_stats[s_id]
        remaining_in_stratum = stats["size"] - stats["n_sampled"]
        if remaining_in_stratum <= 0:
            continue
        if method == "neyman":
            scores[s_id] = math.sqrt(max(stats["variance"], 0.0)) * remaining_in_stratum
        else:
            scores[s_id] = stats["weight"]

    total_score = scores.sum()
    if total_score < 1e-12:
        per = max(1, main_budget // max(n_strata, 1))
        return [min(per, stratum_stats[s]["size"] - stratum_stats[s]["n_sampled"])
                for s in range(n_strata)]

    allocations = []
    budget_left = main_budget
    for s_id in range(n_strata):
        remaining_in_stratum = stratum_stats[s_id]["size"] - stratum_stats[s_id]["n_sampled"]
        raw = round(main_budget * scores[s_id] / total_score)
        raw = min(raw, remaining_in_stratum, budget_left)
        raw = max(0, raw)
        allocations.append(raw)
        budget_left -= raw

    return allocations


def _execute_main_sampling(
    strata: List[np.ndarray],
    allocations: List[int],
    similarities: np.ndarray,
    texts: List[Union[str, list]],
    label_fn: LabelFnType,
    already_labeled: set,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Importance-weighted sampling per stratum for the main budget."""
    rng = np.random.default_rng(99)
    new_indices: List[int] = []
    new_labels: List[int] = []
    main_stratum_stats: List[Dict[str, Any]] = []

    for s_id, (members, n_alloc) in enumerate(zip(strata, allocations)):
        if n_alloc <= 0 or len(members) == 0:
            main_stratum_stats.append({"n_sampled": 0, "n_pos": 0, "pos_rate": 0.0})
            continue
        available = np.array([i for i in members if i not in already_labeled])
        if len(available) == 0:
            main_stratum_stats.append({"n_sampled": 0, "n_pos": 0, "pos_rate": 0.0})
            continue

        picked = _importance_weighted_sample(available, similarities, n_alloc, rng)
        if len(picked) == 0:
            main_stratum_stats.append({"n_sampled": 0, "n_pos": 0, "pos_rate": 0.0})
            continue

        labels = label_fn([texts[i] for i in picked])
        n_pos = int(np.sum(labels == 1))
        for idx, lbl in zip(picked.tolist(), labels.tolist()):
            new_indices.append(idx)
            new_labels.append(int(lbl))

        main_stratum_stats.append({
            "n_sampled": len(picked), "n_pos": n_pos,
            "pos_rate": n_pos / len(picked) if len(picked) > 0 else 0.0,
        })
        logger.info(
            f"  [Stratum {s_id} main] sampled={len(picked)}, "
            f"pos={n_pos}, pos_rate={n_pos / len(picked):.3f}"
        )

    logger.info(
        f"Main sampling: {len(new_indices)} samples, "
        f"{sum(new_labels)} positives"
    )
    return (
        np.array(new_indices, dtype=np.int64),
        np.array(new_labels, dtype=np.int32),
        main_stratum_stats,
    )


def sample_stratified_by_similarity(
    similarities: np.ndarray,
    texts: List[Union[str, list]],
    label_fn: LabelFnType,
    sample_size: int,
    n_strata: int = 10,
    pilot_ratio: float = 0.15,
    allocation_method: str = "neyman",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """BaS-style similarity-stratified sampling.

    Returns (labeled_indices, labeled_values, profiling_stats).
    """
    n_total = len(similarities)
    n_strata = max(1, min(n_strata, n_total))

    logger.info(
        f"Stratified sampling: N={n_total}, budget={sample_size}, "
        f"K={n_strata}, pilot_ratio={pilot_ratio:.2f}, method={allocation_method}"
    )

    t_stratum = time.time()
    strata = _stratify_by_similarity(similarities, n_strata)
    stratum_time = time.time() - t_stratum

    pilot_budget = max(n_strata, int(sample_size * pilot_ratio))
    pilot_budget = min(pilot_budget, sample_size)
    main_budget = sample_size - pilot_budget

    logger.info(f"  pilot_budget={pilot_budget}, main_budget={main_budget}")

    pilot_indices, pilot_labels, stratum_stats = _pilot_sample_strata(
        strata, similarities, texts, label_fn, pilot_budget,
    )

    allocations = _allocate_main_budget(
        stratum_stats, strata, main_budget,
        already_sampled=len(pilot_indices),
        method=allocation_method,
    )

    for s_id, alloc in enumerate(allocations):
        if alloc > 0:
            logger.info(
                f"  [Stratum {s_id}] main_alloc={alloc} "
                f"(pilot_var={stratum_stats[s_id]['variance']:.2f})"
            )

    already_set = set(pilot_indices.tolist())
    main_indices, main_labels, main_stats = _execute_main_sampling(
        strata, allocations, similarities, texts, label_fn, already_set,
    )

    if len(main_indices) > 0:
        labeled_indices = np.concatenate([pilot_indices, main_indices])
        labeled_values = np.concatenate([pilot_labels, main_labels])
    else:
        labeled_indices = pilot_indices
        labeled_values = pilot_labels

    # Combined per-stratum summary
    for s_id in range(n_strata):
        ps = stratum_stats[s_id]
        ms = main_stats[s_id] if s_id < len(main_stats) else {"n_sampled": 0, "n_pos": 0}
        total_sampled = ps["n_sampled"] + ms["n_sampled"]
        total_pos = ps["n_pos"] + ms["n_pos"]
        logger.info(
            f"  [Stratum {s_id} TOTAL] size={ps['size']}, sampled={total_sampled}, "
            f"pos={total_pos}, pos_rate={total_pos / total_sampled:.3f}" if total_sampled > 0
            else f"  [Stratum {s_id} TOTAL] size={ps['size']}, sampled=0"
        )

    profiling = {
        "n_strata": n_strata,
        "pilot_budget": pilot_budget,
        "main_budget": main_budget,
        "pilot_samples": len(pilot_indices),
        "main_samples": len(main_indices),
        "allocation_method": allocation_method,
        "stratum_time": stratum_time,
        "stratum_stats": stratum_stats,
        "main_stratum_stats": main_stats,
    }
    return labeled_indices, labeled_values, profiling


# ---------------------------------------------------------------------------
# KMeans-stratified sampling with regularized Neyman allocation
# (Cai & Rafi 2024, "On the Performance of the Neyman Allocation
#  with Small Pilots")
# ---------------------------------------------------------------------------


def _regularized_neyman_allocation(
    stratum_stats: List[Dict[str, Any]],
    strata: List[np.ndarray],
    main_budget: int,
    reg_method: str = "exponential",
    reg_strength: float = 0.5,
) -> List[int]:
    """Regularized Neyman allocation (Cai & Rafi 2024).

    Shrinks raw Neyman shares toward proportional allocation to handle
    small-pilot variance instability.

    reg_method:
        "additive":    S_h^reg = (1-λ)·S_h + λ·S̄
        "exponential": p_h^reg ∝ p_neyman^μ · p_prop^(1-μ)  (geometric interp)
    reg_strength:
        For additive:    λ ∈ [0,1], 0=raw Neyman, 1=all-equal SD
        For exponential: μ ∈ [0,1], 1=raw Neyman, 0=proportional
    """
    n_strata = len(strata)
    remaining = np.zeros(n_strata)
    raw_sd = np.zeros(n_strata)

    for s_id in range(n_strata):
        stats = stratum_stats[s_id]
        rem = stats["size"] - stats["n_sampled"]
        remaining[s_id] = max(rem, 0)
        raw_sd[s_id] = math.sqrt(max(stats["variance"], 0.0))

    total_remaining = remaining.sum()
    if total_remaining < 1 or main_budget <= 0:
        return [0] * n_strata

    if reg_method == "additive":
        lam = reg_strength
        mean_sd = raw_sd.mean()
        reg_sd = (1.0 - lam) * raw_sd + lam * mean_sd
        scores = reg_sd * remaining
    else:
        mu = reg_strength
        raw_scores = raw_sd * remaining
        total_raw = raw_scores.sum()
        prop_shares = remaining / total_remaining

        eps_floor = 1e-8
        if total_raw < eps_floor:
            neyman_shares = prop_shares.copy()
        else:
            neyman_shares = raw_scores / total_raw

        neyman_shares = np.maximum(neyman_shares, eps_floor)
        prop_shares = np.maximum(prop_shares, eps_floor)

        log_reg = mu * np.log(neyman_shares) + (1.0 - mu) * np.log(prop_shares)
        scores = np.exp(log_reg - log_reg.max())

    total_score = scores.sum()
    if total_score < 1e-12:
        per = max(1, main_budget // max(n_strata, 1))
        return [min(per, int(remaining[s])) for s in range(n_strata)]

    allocations = []
    budget_left = main_budget
    for s_id in range(n_strata):
        raw = round(main_budget * scores[s_id] / total_score)
        raw = min(raw, int(remaining[s_id]), budget_left)
        raw = max(0, raw)
        allocations.append(raw)
        budget_left -= raw

    if budget_left > 0:
        for s_id in sorted(range(n_strata), key=lambda i: remaining[i], reverse=True):
            can_add = int(remaining[s_id]) - allocations[s_id]
            add = min(can_add, budget_left)
            if add > 0:
                allocations[s_id] += add
                budget_left -= add
            if budget_left <= 0:
                break

    return allocations


def auto_k_for_stratified(
    n: int,
    budget: int,
    pilot_ratio: float = 0.15,
    c_pilot: int = 5,
    c_row: int = 30,
    k_min: int = 5,
    k_cap: int = 20,
) -> int:
    """Budget-adaptive K selection for kmeans_stratified sampling.

    Closed-form rule (BAKPR Phase A):
        K_ideal = floor(pilot_ratio * budget / c_pilot)
        K_max   = min(k_cap, floor(sqrt(n) / 2), floor(n / c_row))
        K*      = clamp(K_ideal, k_min, max(k_min, K_max))

    Bounds rationale:
      - k_min=5: Cochran (1977) recommends >=3 strata; 5 gives headroom
      - c_pilot=5: min samples per stratum for valid variance estimate;
        Cai & Rafi (2024) show regularized Neyman compensates for small pilots
      - sqrt(n)/2: Rivest (2002) marginal variance reduction decays past sqrt(N)
      - n/c_row: min rows per stratum (classical rule of 30)
      - k_cap=20: Cochran-inspired diminishing-returns cap
    """
    if budget <= 0 or n <= 0:
        return k_min
    pilot_budget = max(1, int(pilot_ratio * budget))
    k_ideal = pilot_budget // max(c_pilot, 1)

    k_sqrt = int(math.sqrt(n) / 2)
    k_rows = n // max(c_row, 1)
    k_max = min(k_cap, k_sqrt, k_rows)
    k_max = max(k_max, k_min)

    return int(min(max(k_ideal, k_min), k_max))


def _split_merge_refine(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    strata: List[np.ndarray],
    stratum_stats: List[Dict[str, Any]],
    k_min: int,
    k_max: int,
    c_row: int = 30,
    split_ratio: float = 3.0,
    merge_ratio: float = 0.1,
) -> Tuple[np.ndarray, List[np.ndarray], List[Dict[str, Any]]]:
    """Variance-driven split-merge refinement (BAKPR Phase C).

    SPLIT cluster h iff:
      - score_h > split_ratio * mean(scores)  (dominates allocation)
      - N_h >= 2 * c_row                      (room for two sub-clusters)
      - current K + 1 <= k_max

    MERGE cluster h into nearest neighbor iff:
      - score_h < merge_ratio * mean(scores)  (negligible contribution)
      - N_h < 2 * c_row                       (too small to matter)
      - current K - 1 >= k_min

    Returns updated (cluster_labels, strata, stratum_stats).
    Note: pilot stats for split/merged clusters are marked as stale
    (n_sampled=0, variance=0) so they are re-sampled in the main phase.
    """
    k_cur = len(strata)
    scores = np.array([
        math.sqrt(max(stratum_stats[h]["variance"], 0.0))
        for h in range(k_cur)
    ])
    mean_score = float(scores.mean()) if scores.size > 0 else 0.0

    if mean_score < 1e-12:
        return cluster_labels, strata, stratum_stats

    new_labels = cluster_labels.copy()
    new_strata = [s.copy() for s in strata]
    new_stats = [dict(s) for s in stratum_stats]

    split_candidates = [
        h for h in range(k_cur)
        if scores[h] > split_ratio * mean_score
        and new_stats[h]["size"] >= 2 * c_row
    ]
    split_candidates.sort(key=lambda h: -scores[h])

    for h in split_candidates:
        if len(new_strata) >= k_max:
            break
        members = new_strata[h]
        if len(members) < 2 * c_row:
            continue
        sub_emb = embeddings[members]
        try:
            sub_km = KMeans(n_clusters=2, random_state=42, n_init=5)
            sub_labels = sub_km.fit_predict(sub_emb)
        except Exception:
            continue
        new_cluster_id = len(new_strata)
        mask_new = sub_labels == 1
        new_labels[members[mask_new]] = new_cluster_id
        new_strata[h] = members[~mask_new]
        new_strata.append(members[mask_new])

        orig_stats = new_stats[h]
        for idx, sub_members in [(h, new_strata[h]), (new_cluster_id, new_strata[-1])]:
            entry = {
                "size": int(len(sub_members)),
                "n_sampled": 0,
                "n_pos": 0,
                "pos_rate": 0.0,
                "variance": 0.0,
                "weight": orig_stats.get("weight", 0.0) * len(sub_members) / max(orig_stats["size"], 1),
                "sim_min": orig_stats.get("sim_min", 0.0),
                "sim_max": orig_stats.get("sim_max", 0.0),
                "sim_mean": orig_stats.get("sim_mean", 0.0),
                "refined": True,
            }
            if idx == h:
                new_stats[h] = entry
            else:
                new_stats.append(entry)
        logger.info(f"  [Refine] SPLIT cluster {h} (size={orig_stats['size']}) -> clusters {h}, {new_cluster_id}")

    if len(new_strata) > k_min:
        scores_post = np.array([
            math.sqrt(max(new_stats[h]["variance"], 0.0))
            for h in range(len(new_strata))
        ])
        mean_post = float(scores_post.mean()) if scores_post.size > 0 else 0.0
        merge_candidates = [
            h for h in range(len(new_strata))
            if (mean_post > 1e-12 and scores_post[h] < merge_ratio * mean_post
                and new_stats[h]["size"] < 2 * c_row
                and new_stats[h]["size"] > 0)
        ]
        merge_candidates.sort(key=lambda h: new_stats[h]["size"])

        for h in merge_candidates:
            if len(new_strata) <= k_min:
                break
            if new_stats[h]["size"] == 0:
                continue
            members_h = new_strata[h]
            centroid_h = embeddings[members_h].mean(axis=0)
            best_nn = -1
            best_dist = float("inf")
            for k_id in range(len(new_strata)):
                if k_id == h or new_stats[k_id]["size"] == 0:
                    continue
                centroid_k = embeddings[new_strata[k_id]].mean(axis=0)
                d = float(np.linalg.norm(centroid_h - centroid_k))
                if d < best_dist:
                    best_dist = d
                    best_nn = k_id
            if best_nn < 0:
                continue
            new_labels[members_h] = best_nn
            merged = np.concatenate([new_strata[best_nn], members_h])
            new_strata[best_nn] = merged
            new_stats[best_nn] = {
                "size": int(len(merged)),
                "n_sampled": 0,
                "n_pos": 0,
                "pos_rate": 0.0,
                "variance": 0.0,
                "weight": new_stats[best_nn].get("weight", 0.0) + new_stats[h].get("weight", 0.0),
                "sim_min": 0.0, "sim_max": 0.0, "sim_mean": 0.0,
                "refined": True,
            }
            new_strata[h] = np.array([], dtype=np.int64)
            new_stats[h] = {
                "size": 0, "n_sampled": 0, "n_pos": 0, "pos_rate": 0.0,
                "variance": 0.0, "weight": 0.0,
                "sim_min": 0.0, "sim_max": 0.0, "sim_mean": 0.0,
                "refined": True,
            }
            logger.info(f"  [Refine] MERGE cluster {h} -> {best_nn}")

    kept = [i for i in range(len(new_strata)) if new_stats[i]["size"] > 0]
    if len(kept) < len(new_strata):
        remap = {old: new for new, old in enumerate(kept)}
        compact_labels = np.zeros_like(new_labels)
        for old_id, new_id in remap.items():
            compact_labels[new_labels == old_id] = new_id
        new_labels = compact_labels
        new_strata = [new_strata[i] for i in kept]
        new_stats = [new_stats[i] for i in kept]

    return new_labels, new_strata, new_stats


def sample_kmeans_stratified(
    embeddings: np.ndarray,
    similarities: np.ndarray,
    texts: List[Union[str, list]],
    label_fn: LabelFnType,
    sample_size: int,
    n_clusters: int = 0,
    pilot_ratio: float = 0.15,
    reg_method: str = "exponential",
    reg_strength: float = 0.5,
    auto_k: bool = False,
    refine: bool = False,
    c_pilot: int = 5,
    c_row: int = 30,
    k_min: int = 5,
    k_cap: int = 20,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """KMeans-stratified sampling with regularized Neyman allocation.

    Same BaS pipeline but with KMeans clusters as strata and regularized
    Neyman for main-budget allocation (Cai & Rafi 2024).

    auto_k: If True, compute K via auto_k_for_stratified() ignoring n_clusters.
    refine: If True, apply variance-driven split-merge refinement after pilot.

    Returns (labeled_indices, labeled_values, profiling_stats).
    """
    n_total = len(similarities)
    if auto_k:
        n_clusters = auto_k_for_stratified(
            n_total, sample_size,
            pilot_ratio=pilot_ratio, c_pilot=c_pilot, c_row=c_row,
            k_min=k_min, k_cap=k_cap,
        )
        logger.info(f"  auto_k: selected K={n_clusters} for N={n_total}, budget={sample_size}")
    elif n_clusters <= 0:
        n_clusters = auto_kmeans_k(n_total)
    n_clusters = min(n_clusters, n_total)

    k_sqrt = int(math.sqrt(n_total) / 2)
    k_rows = n_total // max(c_row, 1)
    k_max = max(k_min, min(k_cap, k_sqrt, k_rows))

    logger.info(
        f"KMeans-stratified sampling: N={n_total}, budget={sample_size}, "
        f"K={n_clusters}, pilot_ratio={pilot_ratio:.2f}, "
        f"reg={reg_method}({reg_strength:.2f}), refine={refine}"
    )

    t_stratum = time.time()
    cluster_labels, _centroids = cluster_embeddings(embeddings, n_clusters=n_clusters)
    actual_k = int(_centroids.shape[0])
    strata = [np.where(cluster_labels == k)[0] for k in range(actual_k)]
    stratum_time = time.time() - t_stratum

    pilot_budget = max(actual_k, int(sample_size * pilot_ratio))
    pilot_budget = min(pilot_budget, sample_size)
    main_budget = sample_size - pilot_budget

    logger.info(f"  pilot_budget={pilot_budget}, main_budget={main_budget}")

    pilot_indices, pilot_labels, stratum_stats = _pilot_sample_strata(
        strata, similarities, texts, label_fn, pilot_budget,
    )

    refine_info: Dict[str, Any] = {"refined": False, "k_before": actual_k, "k_after": actual_k}
    if refine and actual_k < k_max:
        t_refine = time.time()
        new_labels, new_strata, new_stats = _split_merge_refine(
            embeddings, cluster_labels, strata, stratum_stats,
            k_min=k_min, k_max=k_max, c_row=c_row,
        )
        stratum_time += time.time() - t_refine
        if len(new_strata) != actual_k:
            cluster_labels = new_labels
            strata = new_strata
            stratum_stats = new_stats
            actual_k = len(strata)
            refine_info["refined"] = True
            refine_info["k_after"] = actual_k
            logger.info(f"  [Refine] K changed: {refine_info['k_before']} -> {actual_k}")

    allocations = _regularized_neyman_allocation(
        stratum_stats, strata, main_budget,
        reg_method=reg_method,
        reg_strength=reg_strength,
    )

    for s_id, alloc in enumerate(allocations):
        if alloc > 0:
            logger.info(
                f"  [Cluster {s_id}] main_alloc={alloc} "
                f"(pilot_var={stratum_stats[s_id]['variance']:.2f})"
            )

    already_set = set(pilot_indices.tolist())
    main_indices, main_labels, main_stats = _execute_main_sampling(
        strata, allocations, similarities, texts, label_fn, already_set,
    )

    if len(main_indices) > 0:
        labeled_indices = np.concatenate([pilot_indices, main_indices])
        labeled_values = np.concatenate([pilot_labels, main_labels])
    else:
        labeled_indices = pilot_indices
        labeled_values = pilot_labels

    for s_id in range(actual_k):
        ps = stratum_stats[s_id]
        ms = main_stats[s_id] if s_id < len(main_stats) else {"n_sampled": 0, "n_pos": 0}
        total_sampled = ps["n_sampled"] + ms["n_sampled"]
        total_pos = ps["n_pos"] + ms["n_pos"]
        logger.info(
            f"  [Cluster {s_id} TOTAL] size={ps['size']}, sampled={total_sampled}, "
            f"pos={total_pos}, pos_rate={total_pos / total_sampled:.3f}" if total_sampled > 0
            else f"  [Cluster {s_id} TOTAL] size={ps['size']}, sampled=0"
        )

    profiling = {
        "n_clusters": actual_k,
        "auto_k": auto_k,
        "refine": refine,
        "refine_info": refine_info,
        "pilot_budget": pilot_budget,
        "main_budget": main_budget,
        "pilot_samples": len(pilot_indices),
        "main_samples": len(main_indices),
        "reg_method": reg_method,
        "reg_strength": reg_strength,
        "stratum_time": stratum_time,
        "stratum_stats": stratum_stats,
        "main_stratum_stats": main_stats,
    }
    return labeled_indices, labeled_values, profiling

# ---------------------------------------------------------------------------
# active-learning sampling (Nuggehalli et al., arXiv 2312.09196)
# ---------------------------------------------------------------------------


_ACTIVESAMPLING_MIN_LR_FIT = 20
_ACTIVESAMPLING_LAMBDA_CAP = 1.0


def _activeSampling_split_budget(total: int, k: int, mode: str) -> List[int]:
    if k <= 1 or total <= 0:
        return [max(0, int(total))]
    k = int(k)
    if mode == "front_heavy":
        weights = np.array([0.5 ** i for i in range(k)], dtype=np.float64)
    elif mode == "back_heavy":
        weights = np.array([0.5 ** (k - 1 - i) for i in range(k)], dtype=np.float64)
    else:
        weights = np.ones(k, dtype=np.float64)
    weights = weights / float(weights.sum())
    raw = np.maximum(1, np.floor(weights * total).astype(np.int64))
    deficit = int(total) - int(raw.sum())
    if deficit > 0:
        order = np.argsort(-weights)
        for j in range(deficit):
            raw[order[j % k]] += 1
    elif deficit < 0:
        surplus = -deficit
        order = np.argsort(-weights)
        idx = 0
        while surplus > 0:
            target = order[idx % k]
            if raw[target] > 1:
                raw[target] -= 1
                surplus -= 1
            idx += 1
            if idx > 10 * k:
                break
    return [int(v) for v in raw.tolist()]


def _activeSampling_blend_lambda(
    phase_idx: int,
    n_phases: int,
    schedule: str,
    n_lab: int,
    cold_start_phases: int,
) -> float:
    if phase_idx < max(0, int(cold_start_phases)):
        return 0.0
    if schedule == "sim_only":
        return 0.0
    if schedule == "lr_only":
        return _ACTIVESAMPLING_LAMBDA_CAP
    if schedule == "step":
        return _ACTIVESAMPLING_LAMBDA_CAP if phase_idx >= n_phases // 2 else 0.0
    if schedule == "label_count":
        return min(_ACTIVESAMPLING_LAMBDA_CAP, max(0.0, (n_lab - 30) / 200.0))
    # "linear" (default) and any unrecognized schedule: paper Alg. 1's ramp
    # lambda_t = (t-1)/(T-1), increasing linearly 0 -> 1 across the T rounds.
    # phase_idx is 0-based, so the first non-cold round contributes
    # proportionally and the final round reaches 1.0.
    denom = max(1, n_phases - 1)
    return min(_ACTIVESAMPLING_LAMBDA_CAP, phase_idx / denom)


def _activeSampling_zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-8:
        return np.zeros_like(x, dtype=np.float64)
    return ((x - mu) / sd).astype(np.float64)


def _activeSampling_compute_phase_axis(
    similarities: np.ndarray,
    lr_score: Optional[np.ndarray],
    lam: float,
) -> np.ndarray:
    if lr_score is None or lam <= 0.0:
        return similarities.astype(np.float64)
    z_sim = _activeSampling_zscore(similarities)
    z_lr = _activeSampling_zscore(lr_score)
    return ((1.0 - lam) * z_sim + lam * z_lr).astype(np.float64)


def _activeSampling_inverse_permutation(order: np.ndarray) -> np.ndarray:
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order), dtype=order.dtype)
    return inv


def _activeSampling_get_acc(w: np.ndarray) -> np.ndarray:
    cumsum = np.insert(np.cumsum(w), 0, 0)
    reverse_cumsum = np.insert(np.cumsum(w[::-1]), 0, 0)[::-1]
    return cumsum - reverse_cumsum


def _activeSampling_find_best_hyp(w: np.ndarray) -> int:
    acc = _activeSampling_get_acc(w)
    return int(np.argmax(acc))


def _activeSampling_find_top_k_hyps(
    w: np.ndarray, k: int, min_separation: int,
) -> List[Tuple[int, float]]:
    """Find up to k local maxima of the accuracy vector.

    Returns a list of (position, score) sorted by score descending. Adjacent
    plateaus are collapsed to their midpoint. Peaks are filtered to enforce
    a minimum separation (in sorted-axis positions) so we don't pick K
    near-duplicates of the same peak.

    For multi-modal positive distributions, each local maximum corresponds
    to a candidate label-transition zone — i.e. a separate boundary worth
    querying around in the QA phase.
    """
    if k <= 1:
        acc_single = _activeSampling_get_acc(w)
        if len(acc_single) == 0:
            return []
        argmax_pos = int(np.argmax(acc_single))
        return [(argmax_pos, float(acc_single[argmax_pos]))]

    acc = _activeSampling_get_acc(w)
    n = len(acc)
    if n == 0:
        return []

    raw_peaks: List[Tuple[int, float]] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and acc[j + 1] == acc[i]:
            j += 1
        left_ok = (i == 0) or (acc[i] > acc[i - 1])
        right_ok = (j == n - 1) or (acc[j] > acc[j + 1])
        if left_ok and right_ok:
            raw_peaks.append(((i + j) // 2, float(acc[i])))
        i = j + 1

    if not raw_peaks:
        return [(int(np.argmax(acc)), float(np.max(acc)))]

    raw_peaks.sort(key=lambda p: -p[1])
    selected: List[Tuple[int, float]] = []
    for pos, score in raw_peaks:
        if all(abs(pos - s_pos) >= min_separation for s_pos, _ in selected):
            selected.append((pos, score))
        if len(selected) >= k:
            break
    return selected



def _activeSampling_compute_pos_weight(w: np.ndarray) -> float:
    """Inverse-frequency weight for positive labels in the current w.

    Returns ``N_neg_labeled / N_pos_labeled`` (clamped to >= 1.0). Used
    to rebalance the k-interval DP's sum-of-w objective so net-positive
    intervals can still form even when positives are heavily
    outnumbered (e.g. animals_q1 has ~6% selectivity — a bi-modal
    positive distribution where the minor mode has a 10:50 pos/neg
    ratio is net-negative under unweighted accuracy and gets dropped
    by the DP, so the secondary boundary is invisible).

    Falls back to 1.0 (= no rebalancing) when no positive labels have
    been collected yet, so the first VS round before any +1 has been
    seen behaves as before.
    """
    n_pos = int(np.sum(w == 1.0))
    n_neg = int(np.sum(w == -1.0))
    if n_pos == 0:
        return 1.0
    return max(1.0, float(n_neg) / float(n_pos))


def _activeSampling_kinterval_dp(
    w: np.ndarray, k: int, pos_weight: float = 1.0,
) -> Tuple[List[Tuple[int, int]], float]:
    """Find up to k disjoint subarrays maximizing Σ w[i] over their union.

    Solves the classical "maximum sum of at most k non-overlapping
    subarrays" problem in O(N*k) time and memory. Returns the optimal
    intervals as a list of (a, b) inclusive index pairs in increasing
    order with b_m < a_{m+1}, plus the total sum achieved.

    When w encodes labeled positions as {+1=positive, -1=negative, 0=unlabeled},
    the optimal k-interval union approximates the best k-interval separator
    of positives from negatives on the sorted-similarity axis. Unlabeled
    positions contribute 0, neither helping nor hurting any interval.

    ``pos_weight`` (>= 1.0) up-weights positive entries before the DP
    runs — the standard inverse-frequency trick for class-imbalanced
    data. With pos_weight = N_neg/N_pos, an interval needs only a
    proportional share of positives to be net-positive in the DP
    objective, so secondary positive modes that drown in negatives
    under unweighted accuracy can still be detected. Default 1.0
    keeps the original behavior.

    Empty list returned when the best total is <= 0 (no interval beats the
    "classify nothing as positive" baseline).
    """
    n = int(len(w))
    if n == 0 or k <= 0:
        return [], 0.0
    k = min(k, n)
    NEG_INF = -1e18

    # f[i, j, s] = best sum using the first i elements, with exactly j
    # intervals "used" (started or fully closed), in state s ∈ {0: outside,
    # 1: inside the j-th interval, which must be non-empty).
    f = np.full((n + 1, k + 1, 2), NEG_INF, dtype=np.float64)
    f[0, 0, 0] = 0.0

    if pos_weight != 1.0:
        # Up-weight the +1 entries; negatives unchanged. Stays float64.
        w64 = np.where(np.asarray(w) > 0,
                       np.asarray(w, dtype=np.float64) * float(pos_weight),
                       np.asarray(w, dtype=np.float64))
    else:
        w64 = np.asarray(w, dtype=np.float64)
    for i in range(1, n + 1):
        wi = float(w64[i - 1])
        # state 0: element i is NOT in any interval. Two ways to arrive:
        #   (a) previous position was outside   -> f[i-1, j, 0]
        #   (b) previous position CLOSED the j-th interval -> f[i-1, j, 1]
        f[i, :, 0] = np.maximum(f[i - 1, :, 0], f[i - 1, :, 1])
        # state 1: element i IS in the j-th interval. Two ways to arrive:
        #   (a) continuing: previous position was inside -> f[i-1, j, 1] + wi
        #   (b) starting:   previous position was outside with j-1 intervals
        #                   completed            -> f[i-1, j-1, 0] + wi
        cont = f[i - 1, :, 1] + wi
        start = np.full(k + 1, NEG_INF, dtype=np.float64)
        start[1:] = f[i - 1, :-1, 0] + wi
        f[i, :, 1] = np.maximum(cont, start)

    # Best terminal state.
    flat = f[n].reshape(-1)
    idx = int(np.argmax(flat))
    best = float(flat[idx])
    best_j, best_s = divmod(idx, 2)
    if best <= 0.0:
        return [], 0.0

    # Backtrack through the table to recover intervals.
    intervals: List[Tuple[int, int]] = []
    i = n
    j = int(best_j)
    s = int(best_s)
    cur_right: Optional[int] = None
    while i > 0:
        wi = float(w64[i - 1])
        if s == 1:
            if cur_right is None:
                cur_right = i - 1
            # Did we arrive at (i, j, 1) by continuing or by starting?
            cont_val = f[i - 1, j, 1] + wi
            start_val = (f[i - 1, j - 1, 0] + wi) if j >= 1 else NEG_INF
            if start_val >= cont_val:
                # started here — [i-1, cur_right] is a complete interval
                intervals.append((i - 1, cur_right))
                cur_right = None
                j -= 1
                s = 0
            else:
                s = 1
            i -= 1
        else:
            a = f[i - 1, j, 0]
            b = f[i - 1, j, 1]
            s = 0 if a >= b else 1
            i -= 1

    intervals.reverse()
    return intervals, best


def _activeSampling_kinterval_peaks(
    intervals: List[Tuple[int, int]], w: np.ndarray,
) -> List[Tuple[int, float]]:
    """Interval endpoints as QA peak positions.

    Returns (position, score) tuples for each endpoint, ready to feed into
    the existing QA expansion loop. Score is the sum of w over a 1-unit
    neighborhood inside the interval (proxy for "positive density near the
    boundary"), so endpoints on strongly-positive intervals receive more
    QA budget. Left and right endpoints of the same interval count as
    separate peaks so both boundaries are refined.
    """
    peaks: List[Tuple[int, float]] = []
    if not intervals:
        return peaks
    w64 = np.asarray(w, dtype=np.float64)
    n = int(len(w64))
    for (a, b) in intervals:
        length = max(1, b - a + 1)
        # Peak score ≈ sum of w inside the interval, normalized by length.
        # Bounded above by 1 (pure positive interval), equals -1 for pure
        # negative (which the DP wouldn't have picked anyway, so safe).
        interval_score = float(np.sum(w64[a : b + 1])) / length
        peaks.append((int(a), interval_score))
        if b != a:
            peaks.append((int(b), interval_score))
    # Deduplicate positions (k=1 with a==b would otherwise produce a
    # duplicate) while keeping insertion order.
    seen: set = set()
    out: List[Tuple[int, float]] = []
    for pos, score in peaks:
        if pos in seen:
            continue
        seen.add(pos)
        if 0 <= pos < n:
            out.append((pos, score))
    return out


def _activeSampling_auto_select_k(
    w: np.ndarray, k_max: int, gain_tol: float, pos_weight: float = 1.0,
) -> Tuple[int, List[Tuple[int, int]], float]:
    """Elbow selection of k by relative-gain threshold.

    Runs the k-interval DP for k in {1, ..., k_max}; picks the smallest k
    whose marginal relative gain (acc_k - acc_{k-1}) / max(1, acc_{k-1})
    is below `gain_tol`. Falls back to k_max if every k keeps adding
    non-negligible accuracy. Returns (chosen_k, intervals_at_chosen_k,
    acc_at_chosen_k).

    ``pos_weight`` is forwarded to every internal DP call so the elbow
    is computed against the same balanced objective the caller expects.
    """
    k_max = max(1, int(k_max))
    prev_acc = 0.0
    chosen_k = 1
    chosen_intervals: List[Tuple[int, int]] = []
    chosen_acc = 0.0
    for k in range(1, k_max + 1):
        intervals, acc = _activeSampling_kinterval_dp(w, k, pos_weight=pos_weight)
        if k == 1:
            chosen_k, chosen_intervals, chosen_acc = 1, intervals, acc
            prev_acc = acc
            continue
        gain = acc - prev_acc
        rel = gain / max(1.0, abs(prev_acc))
        if rel < gain_tol:
            # Not enough marginal gain — stop at previous k.
            break
        chosen_k, chosen_intervals, chosen_acc = k, intervals, acc
        prev_acc = acc
    return chosen_k, chosen_intervals, chosen_acc


def _activeSampling_multi_interval_vs_reduction(
    sorted_indices: np.ndarray,
    texts: List[Union[str, list]],
    label_fn: LabelFnType,
    labeled_mask: np.ndarray,
    w: np.ndarray,
    windows: List[Tuple[int, int]],
    B1: int,
    B2: int,
    k: int,
    class_balance: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, List[Tuple[int, int]]]:
    spend = 0
    n = int(len(w))

    if B2 <= 0 or not windows:
        return labeled_mask, w, spend, windows

    budget = min(B1, B2)

    # Index-based uniform sampling over the UNION of windows, excluding
    # already-labeled positions. Equivalent to a uniform probability
    # vector over the same support but skips the O(N) weighted-choice
    # path — see _activeSampling_version_space_reduction for the rationale.
    _t_choice = time.time()
    window_mask = np.zeros(n, dtype=bool)
    for (I, J) in windows:
        if 0 <= I <= J < n:
            window_mask[I : J + 1] = True
    window_mask &= ~labeled_mask
    candidate_positions = np.flatnonzero(window_mask)
    n_available = candidate_positions.size
    num_sample = min(budget, n_available)
    if num_sample <= 0:
        return labeled_mask, w, spend, windows
    samp_positions = np.random.choice(
        candidate_positions, size=num_sample, replace=False
    )
    _dt_choice = time.time() - _t_choice
    query_global_indices = sorted_indices[samp_positions]

    query_texts = [texts[gi] for gi in query_global_indices]
    _t_llm = time.time()
    query_labels = np.asarray(label_fn(query_texts), dtype=np.int32)
    _dt_llm = time.time() - _t_llm
    spend += num_sample
    logger.info(
        f"    [multi-VS round] N={n} k_windows={len(windows)} "
        f"avail={n_available} sampled={num_sample} | "
        f"choice={_dt_choice*1000:.1f}ms llm={_dt_llm:.2f}s"
    )

    labeled_mask[samp_positions] = True
    w[samp_positions] = np.where(query_labels == 1, 1.0, -1.0)

    # Re-run the DP to get updated optimal intervals. With
    # class_balance, recompute pos_weight from the just-updated w so
    # the rebalancing tracks the running pos/neg ratio as more labels
    # come in (early rounds with few +1 labels degrade gracefully —
    # _activeSampling_compute_pos_weight returns 1.0 when no positives yet).
    pw = _activeSampling_compute_pos_weight(w) if class_balance else 1.0
    intervals, _ = _activeSampling_kinterval_dp(w, k, pos_weight=pw)

    # Shrink each window. If the DP returned fewer intervals than we have
    # windows (e.g. some windows are purely negative so far), unmatched
    # windows just keep their current extent and will be re-probed next
    # round; this preserves exploration without collapsing too aggressively.
    remaining_rounds = max(1, B2 // max(1, B1))
    new_windows: List[Tuple[int, int]] = []
    used_dp = [False] * len(intervals)
    for (I, J) in windows:
        if I >= J:
            continue
        # Match this window to its closest DP interval by midpoint distance,
        # preferring unused DP intervals so we don't collapse two windows
        # onto the same DP interval.
        win_mid = (I + J) / 2.0
        best_idx = -1
        best_dist = float("inf")
        for idx, (da, db) in enumerate(intervals):
            if used_dp[idx]:
                continue
            if db < I or da > J:
                # DP interval does not overlap this window at all; still
                # allow matching but with a distance penalty.
                dist = abs((da + db) / 2.0 - win_mid) + n
            else:
                dist = abs((da + db) / 2.0 - win_mid)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx >= 0:
            used_dp[best_idx] = True
            dp_a, dp_b = intervals[best_idx]
            dp_a = max(I, dp_a)
            dp_b = min(J, dp_b)
        else:
            # No DP interval available — retain current window.
            new_windows.append((I, J))
            continue

        cur_diam = int(np.count_nonzero(~labeled_mask[I : J + 1]))
        if cur_diam <= 1:
            new_windows.append((I, J))
            continue
        c2 = cur_diam ** (1.0 / remaining_rounds) if remaining_rounds > 1 else 1.0
        target = max(1, math.ceil(cur_diam / c2))
        diam = cur_diam
        ni, nj = I, J
        while diam > target and ni < nj:
            # Trim from whichever side lies further from [dp_a, dp_b];
            # fall back to accuracy-based tie-breaking (lower-acc side
            # trimmed first) mirroring the 1-D algorithm.
            left_dist = max(0, dp_a - ni)
            right_dist = max(0, nj - dp_b)
            if right_dist > left_dist:
                if not labeled_mask[nj]:
                    diam -= 1
                nj -= 1
            elif left_dist > right_dist:
                if not labeled_mask[ni]:
                    diam -= 1
                ni += 1
            else:
                # Symmetric: trim from the side with the lower local
                # accuracy under the current hypothesis; ties broken
                # randomly (matches 1-D behavior).
                if np.random.randint(0, 2) == 0:
                    if not labeled_mask[ni]:
                        diam -= 1
                    ni += 1
                else:
                    if not labeled_mask[nj]:
                        diam -= 1
                    nj -= 1
            if ni >= nj:
                break
        new_windows.append((ni, nj))

    # Merge any windows that became adjacent or overlapping after shrinking,
    # so the next round's uniform sampling doesn't double-count positions.
    new_windows.sort()
    merged: List[Tuple[int, int]] = []
    for (I, J) in new_windows:
        if merged and I <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], J))
        else:
            merged.append((I, J))

    sub_labeled, w, sub_spend, final_windows = _activeSampling_multi_interval_vs_reduction(
        sorted_indices, texts, label_fn, labeled_mask, w,
        merged, B1, B2 - budget, k, class_balance=class_balance,
    )
    return sub_labeled, w, spend + sub_spend, final_windows


def _activeSampling_version_space_reduction(
    sorted_indices: np.ndarray,
    texts: List[Union[str, list]],
    label_fn: LabelFnType,
    labeled_mask: np.ndarray,
    w: np.ndarray,
    I: int,
    J: int,
    B1: int,
    B2: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """active-learning sampling version-space reduction."""
    n = len(w)
    spend = 0

    if B2 <= 0 or I >= J:
        return labeled_mask, w, spend

    budget = min(B1, B2)

    # Index-based uniform sampling over unlabeled positions in [I, J].
    # Equivalent to building a probability vector lmbda[i] = 1/span on
    # [I, J] and zeroing out labeled rows, then calling
    # np.random.choice(n, ..., p=lmbda/total). The probability vector is
    # uniform on its support, so we can skip building it: just enumerate
    # the candidate indices and Fisher-Yates over them. ~50–200× faster
    # for large N (numpy's weighted-without-replacement path is O(N) with
    # a high constant; index-based is O(k) for k=B1).
    _t_choice = time.time()
    if I < 0:
        I = 0
    if J >= n:
        J = n - 1
    if I > J:
        return labeled_mask, w, spend
    window_mask = np.zeros(n, dtype=bool)
    window_mask[I : J + 1] = True
    window_mask &= ~labeled_mask
    candidate_positions = np.flatnonzero(window_mask)
    n_available = candidate_positions.size
    num_sample = min(budget, n_available)
    if num_sample <= 0:
        return labeled_mask, w, spend
    samp_positions = np.random.choice(
        candidate_positions, size=num_sample, replace=False
    )
    _dt_choice = time.time() - _t_choice

    query_global_indices = sorted_indices[samp_positions]
    query_texts = [texts[gi] for gi in query_global_indices]
    _t_llm = time.time()
    query_labels = label_fn(query_texts)
    _dt_llm = time.time() - _t_llm
    spend += num_sample
    logger.info(
        f"    [VS round] N={n} I={I} J={J} span={J-I+1} "
        f"avail={n_available} sampled={num_sample} | "
        f"choice={_dt_choice*1000:.1f}ms llm={_dt_llm:.2f}s"
    )

    query_labels_arr = np.asarray(query_labels, dtype=np.int32)
    labeled_mask[samp_positions] = True
    w[samp_positions] = np.where(query_labels_arr == 1, 1.0, -1.0)

    # --- Interval contraction (paper Alg. 2, lines 5-9) ---
    # Geometric base is the RAW interval width (J - I); shrink factor
    # c = (J - I)^(1/m) with m the remaining rounds, giving target width
    # h = (J - I) / c. We then contract to the width-h sub-interval that
    # minimizes the paper's max-endpoint threshold loss:
    #     (I, J) <- argmin_{i,j in [I,J], j-i=h} max{ L(i), L(j) },
    # where L(s) = |{matches <= s}| + |{non-matches > s}| is the empirical
    # threshold loss. We recover L from the signed labels w via the exact
    # affine identity L(s) = (sum|w| - acc(s)) / 2, where
    # acc(s) = sum_{k<s} w_k - sum_{k>=s} w_k (so argmin L == argmax acc).
    width = J - I
    if width > 1:
        remaining_rounds = B2 // B1 if B1 > 0 else 1
        if remaining_rounds <= 1:
            c2 = float(width)
        else:
            c2 = float(width) ** (1.0 / remaining_rounds)
        h = max(1, int(round(width / c2))) if c2 > 0 else 1
        if h < width:
            total_labeled = float(np.sum(np.abs(w)))
            loss = (total_labeled - _activeSampling_get_acc(w)) / 2.0  # L(s), 0<=s<=n
            # max{L(i), L(j)} over every width-h sub-interval [i, i+h] of [I, J]
            left = loss[I : J - h + 1]
            right = loss[I + h : J + 1]
            best_off = int(np.argmin(np.maximum(left, right)))
            I, J = I + best_off, I + best_off + h

    sub_labeled, w, sub_spend = _activeSampling_version_space_reduction(
        sorted_indices, texts, label_fn, labeled_mask, w,
        I, J, B1, B2 - budget,
    )
    return sub_labeled, w, spend + sub_spend


def sample_random_pure(
    embeddings: np.ndarray,
    texts: List[Union[str, list]],
    label_fn: LabelFnType,
    sample_size: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Optional[csr_matrix], Dict[str, Any]]:
    n_total = embeddings.shape[0]
    logger.info(f"Random sampling: N={n_total}, budget={sample_size}")

    t_sample = time.time()
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(
        n_total, size=min(sample_size, n_total), replace=False
    )
    sample_texts = [texts[i] for i in sample_indices]
    sample_labels = np.asarray(label_fn(sample_texts), dtype=np.int32)
    sample_time = time.time() - t_sample

    n_pos = int(np.sum(sample_labels == 1))
    logger.info(
        f"  Sampled: {len(sample_indices)} nodes, {n_pos} positive "
        f"({100.0 * n_pos / max(len(sample_indices), 1):.1f}%) in {sample_time:.2f}s"
    )

    profiling: Dict[str, Any] = {
        "graph_time": 0.0,
        "sample_size": sample_size,
        "n_sampled": len(sample_indices),
        "n_pos": n_pos,
        "sample_time": sample_time,
        "observed_selectivity": n_pos / max(len(sample_indices), 1),
    }
    return sample_indices, sample_labels, None, profiling


def sample_top_k_similarity(
    embeddings: np.ndarray,
    similarities: np.ndarray,
    texts: List[Union[str, list]],
    label_fn: LabelFnType,
    sample_size: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[csr_matrix], Dict[str, Any]]:
    if similarities is None:
        raise ValueError("similarities array is required for sampling_method='top_k'")

    n_total = int(embeddings.shape[0])
    K = int(min(max(0, sample_size), n_total))

    logger.info(f"top_k similarity sampling: N={n_total}, K={K}")

    t_sample = time.time()
    if K <= 0:
        sample_time = time.time() - t_sample
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int32),
            None,
            {
                "graph_time": 0.0,
                "sample_size": int(sample_size),
                "n_sampled": 0,
                "n_pos": 0,
                "sample_time": sample_time,
                "observed_selectivity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
            },
        )

    # Sort descending by similarity; stable sort so ties break by index.
    order = np.argsort(-similarities.astype(np.float64, copy=False), kind="stable")
    top_idx = order[:K].astype(np.int64)
    # Sort ascending so labeled_indices is monotonic (downstream pipeline
    # convention; matches every other sampler in this module).
    sample_indices = np.sort(top_idx)

    sample_texts = [texts[int(i)] for i in sample_indices.tolist()]
    t_llm = time.time()
    sample_labels = np.asarray(label_fn(sample_texts), dtype=np.int32)
    llm_time = time.time() - t_llm
    sample_time = time.time() - t_sample

    n_pos = int(np.sum(sample_labels == 1))
    sims_sel = similarities[sample_indices]
    logger.info(
        f"  top_k: K={K} labeled, {n_pos} positive "
        f"({100.0 * n_pos / max(K, 1):.1f}%), "
        f"sim_range=[{float(sims_sel.min()):.4f}, {float(sims_sel.max()):.4f}], "
        f"llm={llm_time:.2f}s"
    )

    profiling: Dict[str, Any] = {
        "graph_time": 0.0,
        "sample_size": int(sample_size),
        "n_sampled": int(K),
        "n_pos": n_pos,
        "sample_time": sample_time,
        "observed_selectivity": n_pos / max(K, 1),
        "vllm_invocation_time": float(llm_time),
        "min_similarity": float(sims_sel.min()),
        "max_similarity": float(sims_sel.max()),
    }
    return sample_indices, sample_labels, None, profiling


# ============================================================================
# UQE Algorithm 2 — online active-learning sampler.
#
# Literal implementation of Liu et al., NeurIPS 2024 (arXiv:2407.09522),
# §4.1.2 + §C.2:
#   1. Embed each row T_i as e_i (done upstream — frozen embeddings).
#   2. Maintain ĝ(i) approximating f(T_i, cond); init ĝ(i) ~ U(0, 1).
#   3. S = ∅ at t = 0.
#   4. At step t: S_t = argmax over unsampled rows of ĝ(i) + ε_{t,i};
#      observe f(T_i, cond) for i ∈ S_t; S ← S ∪ S_t.
#   5. Refit ĝ on S; goto 4 while |S| < B.
#
# Per §C.2 the paper:
#   - sets ε_{t,i} = 0 in all experiments (configurable here),
#   - trains g only after at least one positive AND one negative are
#     observed ("cold-start guard"),
#   - uses sklearn LogisticRegression with default hyperparameters.
#
# Spec: docs/uqe_active_implementation.md
# ============================================================================

_UQE_MIN_LR_FIT = 4  # minimum |S| before attempting an LR fit


def sample_uqe_active(
    embeddings: np.ndarray,
    similarities: Optional[np.ndarray],
    texts: List[Union[str, list]],
    label_fn: LabelFnType,
    sample_size: int,
    *,
    batch_size: int = 16,
    exploration_sigma_0: float = 0.0,
    exploration_decay: float = 0.9,
    min_per_class_for_fit: int = 1,
    lr_C: float = 1.0,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Optional[csr_matrix], Dict[str, Any]]:
    n_total = int(embeddings.shape[0])
    rng = np.random.default_rng(int(random_seed))

    target_B = int(min(max(0, sample_size), n_total))

    profiling_empty: Dict[str, Any] = {
        "graph_time": 0.0,
        "sample_size": int(sample_size),
        "n_sampled": 0,
        "n_pos": 0,
        "sample_time": 0.0,
        "observed_selectivity": 0.0,
        "n_rounds": 0,
        "n_lr_refits": 0,
        "lr_fit_time": 0.0,
        "vllm_invocation_time": 0.0,
        "round_n_pos": [],
        "round_n_neg": [],
        "exploration_sigma": [],
    }
    if target_B <= 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int32),
            None,
            profiling_empty,
        )

    b = max(1, int(batch_size))

    logger.info(
        f"uqe_active sampling: N={n_total}, B={target_B}, b={b}, "
        f"sigma0={exploration_sigma_0}, lr_C={lr_C}"
    )

    # UQE Alg.2 step 2: ĝ(i) ~ U(0, 1). Used as the acquisition score in
    # the cold-start phase (before LR is trainable). Because rows are
    # selected by argmax over these scores, this is equivalent to
    # iterating through a fixed random permutation.
    scores_init = rng.uniform(0.0, 1.0, size=n_total)

    sampled_mask = np.zeros(n_total, dtype=bool)
    labels_arr = np.full(n_total, -1, dtype=np.int32)

    t0_wall = time.time()
    lr_fit_time_acc = 0.0
    llm_time_acc = 0.0
    n_lr_refits = 0

    round_n_pos: List[int] = []
    round_n_neg: List[int] = []
    exploration_sigma: List[float] = []

    lr_model: Optional[LogisticRegression] = None
    p1_full: Optional[np.ndarray] = None

    t = 0
    while int(sampled_mask.sum()) < target_B:
        # UQE Alg.2 step 4: pick batch by argmax (ĝ + ε) over unlabeled.
        sigma_t = float(exploration_sigma_0) * (
            float(exploration_decay) ** t
        )
        exploration_sigma.append(sigma_t)

        if p1_full is not None:
            scores_now = p1_full.astype(np.float64, copy=True)
        else:
            scores_now = scores_init.astype(np.float64, copy=True)

        if sigma_t > 0.0:
            scores_now = scores_now + sigma_t * rng.standard_normal(
                size=n_total
            )
        scores_now[sampled_mask] = -np.inf

        n_left = target_B - int(sampled_mask.sum())
        b_t = min(b, int(n_left))
        if b_t <= 0:
            break
        batch = np.argsort(-scores_now, kind="stable")[:b_t].astype(np.int64)

        # ---- oracle
        batch_texts = [texts[int(i)] for i in batch.tolist()]
        t_llm = time.time()
        batch_labels = np.asarray(label_fn(batch_texts), dtype=np.int32)
        llm_time_acc += time.time() - t_llm

        sampled_mask[batch] = True
        labels_arr[batch] = batch_labels

        n_pos_total = int((labels_arr[sampled_mask] == 1).sum())
        n_neg_total = int((labels_arr[sampled_mask] == 0).sum())
        round_n_pos.append(n_pos_total)
        round_n_neg.append(n_neg_total)

        # UQE Alg.2 step 5: refit ĝ with the observations seen so far.
        # Cold-start guard (paper §C.2): require both classes to be
        # present before fitting.
        has_both = (
            n_pos_total >= int(min_per_class_for_fit)
            and n_neg_total >= int(min_per_class_for_fit)
        )
        if has_both and int(sampled_mask.sum()) >= _UQE_MIN_LR_FIT:
            labeled_idx_so_far = np.where(sampled_mask)[0]
            X_train = embeddings[labeled_idx_so_far]
            y_train = labels_arr[labeled_idx_so_far].astype(np.int32, copy=False)
            try:
                t_lr = time.time()
                # Paper §C.2: "simply leverage sklearn for that" — use
                # default LR hyperparameters (no class_weight, no
                # resampling). `lr_C` and `max_iter` are exposed so the
                # caller can override regularisation if needed.
                lr_model = LogisticRegression(
                    C=float(lr_C),
                    max_iter=1000,
                ).fit(X_train, y_train)
                lr_fit_time_acc += time.time() - t_lr
                p1_full = lr_model.predict_proba(embeddings)[:, 1]
                n_lr_refits += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[uqe_active] LR refit failed: {exc}; keeping previous"
                )

        t += 1

    # ----------------------------------------------------------------- out
    labeled_indices = np.sort(np.where(sampled_mask)[0].astype(np.int64))
    labeled_values = labels_arr[labeled_indices].astype(np.int32, copy=False)
    n_pos = int((labeled_values == 1).sum())
    total_wall = time.time() - t0_wall

    logger.info(
        f"  uqe_active total: {len(labeled_indices)} labeled, {n_pos} positive "
        f"({100.0 * n_pos / max(len(labeled_indices), 1):.1f}%), "
        f"rounds={t}, lr_refits={n_lr_refits}, "
        f"wall={total_wall:.2f}s (lr_fit={lr_fit_time_acc:.2f}s, "
        f"llm={llm_time_acc:.2f}s)"
    )

    profiling: Dict[str, Any] = {
        "graph_time": 0.0,
        "sample_size": int(sample_size),
        "n_sampled": int(len(labeled_indices)),
        "n_pos": n_pos,
        "sample_time": max(0.0, total_wall - lr_fit_time_acc),
        "observed_selectivity": n_pos / max(len(labeled_indices), 1),
        "n_rounds": int(t),
        "n_lr_refits": int(n_lr_refits),
        "lr_fit_time": float(lr_fit_time_acc),
        "vllm_invocation_time": float(llm_time_acc),
        "round_n_pos": round_n_pos,
        "round_n_neg": round_n_neg,
        "exploration_sigma": exploration_sigma,
    }
    return labeled_indices, labeled_values, None, profiling


def _run_activeSampling_single_phase(
    sort_order: np.ndarray,
    texts: List[Union[str, list]],
    label_fn: LabelFnType,
    labeled_mask: np.ndarray,
    w: np.ndarray,
    *,
    B1: int,
    vs_budget: int,
    qa_budget: int,
    use_multi_interval: bool,
    vs_k: int,
    effective_k: int,
    n_modes: int,
    k_auto: bool,
    k_max: int,
    k_gain_tol: float,
    class_balance: bool,
) -> Dict[str, Any]:
    n_total = len(w)

    t_vs = time.time()
    final_windows: List[Tuple[int, int]] = []
    chosen_intervals: List[Tuple[int, int]] = []
    chosen_acc = 0.0
    dp_time = 0.0
    if not use_multi_interval:
        labeled_mask, w, vs_spend = _activeSampling_version_space_reduction(
            sort_order, texts, label_fn, labeled_mask, w,
            I=0, J=n_total - 1, B1=B1, B2=vs_budget,
        )
        chosen_k = 1
    else:
        init_windows: List[Tuple[int, int]] = []
        step = n_total / float(max(1, vs_k))
        for m in range(max(1, vs_k)):
            lo = int(math.floor(m * step))
            hi = int(math.floor((m + 1) * step)) - 1
            if m == max(1, vs_k) - 1:
                hi = n_total - 1
            if hi >= lo:
                init_windows.append((lo, hi))
        labeled_mask, w, vs_spend, final_windows = _activeSampling_multi_interval_vs_reduction(
            sort_order, texts, label_fn, labeled_mask, w,
            init_windows, B1=B1, B2=vs_budget, k=max(1, vs_k),
            class_balance=bool(class_balance),
        )
        chosen_k = effective_k
    vs_time = time.time() - t_vs
    logger.info(f"  Version space reduction: spent {vs_spend} labels in {vs_time:.2f}s")

    final_pos_weight = 1.0
    if use_multi_interval:
        t_dp = time.time()
        if class_balance:
            final_pos_weight = _activeSampling_compute_pos_weight(w)
        if k_auto:
            _k_upper, chosen_intervals, chosen_acc = _activeSampling_auto_select_k(
                w, k_max=int(k_max), gain_tol=float(k_gain_tol),
                pos_weight=final_pos_weight,
            )
        else:
            chosen_intervals, chosen_acc = _activeSampling_kinterval_dp(
                w, effective_k, pos_weight=final_pos_weight,
            )
        chosen_k = len(chosen_intervals)
        dp_time = time.time() - t_dp

    t_qa = time.time()
    if use_multi_interval and chosen_intervals:
        peaks = _activeSampling_kinterval_peaks(chosen_intervals, w)
    else:
        effective_n_modes = max(1, int(n_modes))
        min_separation = max(1, n_total // (effective_n_modes * 10))
        peaks = _activeSampling_find_top_k_hyps(
            w, k=effective_n_modes, min_separation=min_separation,
        )
    if not peaks:
        acc_fb = _activeSampling_get_acc(w)
        if len(acc_fb) > 0:
            argmax_pos_fb = int(np.argmax(acc_fb))
            peaks = [(argmax_pos_fb, float(acc_fb[argmax_pos_fb]))]
        else:
            peaks = [(0, 0.0)]

    score_floor = 1.0
    weights = np.array([max(s, 0.0) + score_floor for _, s in peaks], dtype=np.float64)
    weights = weights / weights.sum()
    per_peak_budget = [max(2, int(qa_budget * w_)) for w_ in weights]

    overflow = sum(per_peak_budget) - qa_budget
    if overflow > 0:
        idx_max = int(np.argmax(per_peak_budget))
        per_peak_budget[idx_max] = max(2, per_peak_budget[idx_max] - overflow)

    logger.info(
        f"  QA peaks (k_intervals={effective_k}, chosen_k={chosen_k}, "
        f"vs_k={vs_k}): "
        + ", ".join(f"pos={p} score={s:.3f} budget={b}"
                    for (p, s), b in zip(peaks, per_peak_budget))
    )

    pending_positions: List[int] = []
    pending_set: set = set()
    for (peak_pos, _peak_score), peak_budget in zip(peaks, per_peak_budget):
        remaining = int(peak_budget)
        pos = int(peak_pos) - 1  # match side: strictly left of the threshold
        while remaining > 0 and pos >= 0:
            if not labeled_mask[pos] and pos not in pending_set:
                pending_positions.append(pos)
                pending_set.add(pos)
                remaining -= 1
            pos -= 1

    total_qa = int(sum(int(b) for b in per_peak_budget))
    leftover = total_qa - len(pending_positions)
    if leftover > 0:
        t_star = int(peaks[0][0]) if peaks else 0
        for p in range(t_star, n_total):
            if leftover <= 0:
                break
            if not labeled_mask[p] and p not in pending_set:
                pending_positions.append(int(p))
                pending_set.add(int(p))
                leftover -= 1

    qa_spend = 0
    if pending_positions:
        pending_arr = np.asarray(pending_positions, dtype=np.int64)
        global_indices = sort_order[pending_arr]

        query_texts = [texts[gi] for gi in global_indices]
        query_labels = np.asarray(label_fn(query_texts), dtype=np.int32)
        qa_spend = len(pending_positions)
        labeled_mask[pending_arr] = True
        w[pending_arr] = np.where(query_labels == 1, 1.0, -1.0)

    qa_time = time.time() - t_qa
    logger.info(f"  Query around hypothesis: spent {qa_spend} labels in {qa_time:.2f}s")

    if use_multi_interval and qa_spend > 0:
        t_dp_post = time.time()
        if class_balance:
            final_pos_weight = _activeSampling_compute_pos_weight(w)
        if k_auto:
            _k_upper, chosen_intervals, chosen_acc = _activeSampling_auto_select_k(
                w, k_max=int(k_max), gain_tol=float(k_gain_tol),
                pos_weight=final_pos_weight,
            )
        else:
            chosen_intervals, chosen_acc = _activeSampling_kinterval_dp(
                w, effective_k, pos_weight=final_pos_weight,
            )
        chosen_k = len(chosen_intervals)
        dp_time += time.time() - t_dp_post

    return {
        "labeled_mask": labeled_mask,
        "w": w,
        "vs_spend": int(vs_spend),
        "qa_spend": int(qa_spend),
        "vs_time": float(vs_time),
        "qa_time": float(qa_time),
        "dp_time": float(dp_time),
        "peaks": peaks,
        "per_peak_budget": per_peak_budget,
        "chosen_intervals": chosen_intervals,
        "chosen_acc": float(chosen_acc),
        "final_windows": final_windows,
        "final_pos_weight": float(final_pos_weight),
        "chosen_k": int(chosen_k),
    }


def sample_activeSampling(
    embeddings: np.ndarray,
    similarities: np.ndarray,
    texts: List[Union[str, list]],
    label_fn: LabelFnType,
    sample_size: int,
    knn_k: int = 0,
    B1: int = 5,
    vs_ratio: float = 0.5,
    n_modes: int = 1,
    k_intervals: Optional[int] = None,
    k_auto: bool = False,
    k_max: int = 4,
    k_gain_tol: float = 0.02,
    class_balance: bool = False,
    n_phases: int = 1,
    lr_blend_schedule: str = "linear",
    cold_start_phases: int = 1,
    lr_explore_frac: float = 0.0,
    phase_budget_split: str = "equal",
    lr_C: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[csr_matrix], Dict[str, Any]]:
    n_total = len(similarities)
    if knn_k <= 0:
        knn_k = auto_knn_k(n_total)

    use_multi_interval = (
        k_intervals is not None or bool(k_auto) or int(n_modes) > 1
    )
    if k_intervals is None:
        effective_k = max(1, int(n_modes))
    else:
        effective_k = max(1, int(k_intervals))
    vs_k = max(effective_k, int(k_max)) if k_auto else effective_k

    effective_n_phases = max(1, int(n_phases))
    if use_multi_interval and effective_n_phases > 1:
        logger.warning(
            f"active-learning sampling: n_phases={effective_n_phases} requested but "
            f"multi-interval path is active (k_intervals/k_auto/n_modes>1); "
            f"falling back to n_phases=1 (multi-interval × dynamic-axis "
            f"is out of scope for v1)"
        )
        effective_n_phases = 1

    logger.info(
        f"active-learning sampling: N={n_total}, budget={sample_size}, "
        f"knn_k={knn_k}, B1={B1}, vs_ratio={vs_ratio}, "
        f"k_intervals={effective_k}, k_auto={k_auto}, k_max={k_max}, "
        f"multi_interval_path={use_multi_interval}, "
        f"class_balance={class_balance}, n_phases={effective_n_phases}, "
        f"lr_blend_schedule={lr_blend_schedule}, "
        f"cold_start_phases={cold_start_phases}, "
        f"lr_explore_frac={lr_explore_frac}, "
        f"phase_budget_split={phase_budget_split}, lr_C={lr_C}"
    )

    W: Optional[csr_matrix] = None
    graph_time = 0.0

    effective_B1 = max(1, B1)
    B2_total = max(1, int(sample_size * vs_ratio))
    if B2_total > 0:
        HTTP_CHUNK = 32
        target_rounds = 3
        per_round = max(HTTP_CHUNK, B2_total // target_rounds)
        auto_min = min(per_round, max(1, B2_total // 2))
        if effective_B1 < auto_min:
            effective_B1 = auto_min
    if effective_B1 != B1:
        logger.info(
            f"  Scaling B1 from {B1} to {effective_B1} to fill LLM HTTP batches"
        )

    phase_budgets = _activeSampling_split_budget(
        int(sample_size), effective_n_phases, phase_budget_split,
    )
    logger.info(f"  Phase budgets: {phase_budgets}")

    labeled_global: Dict[int, int] = {}
    rng = np.random.default_rng()

    last_peaks: List[Tuple[int, float]] = []
    last_per_peak_budget: List[int] = []
    last_chosen_intervals: List[Tuple[int, int]] = []
    last_chosen_acc = 0.0
    last_final_windows: List[Tuple[int, int]] = []
    last_final_pos_weight = 1.0
    last_chosen_k = 1
    last_sort_order: np.ndarray = np.argsort(-similarities)
    # Final blended ordering axis σ (paper Eq. a_i^(t)); exported so the router
    # can compute π_i as a rank on this axis rather than on raw similarity.
    last_axis: np.ndarray = similarities.astype(np.float64)

    phase_axis_source: List[str] = []
    phase_vs_spend: List[int] = []
    phase_qa_spend: List[int] = []
    phase_explore_spend: List[int] = []
    phase_lambda: List[float] = []
    phase_n_labeled_after: List[int] = []
    axis_lr_corr: List[float] = []
    total_vs_time = 0.0
    total_qa_time = 0.0
    total_dp_time = 0.0
    total_lr_refit_time = 0.0
    n_lr_refits = 0

    lr_model: Optional[LogisticRegression] = None

    def _refit_lr() -> Tuple[Optional[LogisticRegression], float]:
        if len(labeled_global) < _ACTIVESAMPLING_MIN_LR_FIT:
            return None, 0.0
        keys = np.fromiter(labeled_global.keys(), dtype=np.int64)
        vals = np.fromiter(
            (labeled_global[int(k)] for k in keys.tolist()), dtype=np.int32,
        )
        if len(np.unique(vals)) < 2:
            return None, 0.0
        t_lr = time.time()
        try:
            model = LogisticRegression(C=float(lr_C), max_iter=200)
            model.fit(embeddings[keys], vals)
        except Exception as exc:
            logger.warning(f"  LR refit failed: {exc}; keeping previous LR")
            return None, time.time() - t_lr
        return model, time.time() - t_lr

    for phase_idx in range(effective_n_phases):
        phase_total_budget = int(phase_budgets[phase_idx])
        if phase_total_budget <= 0:
            phase_axis_source.append("skipped")
            phase_vs_spend.append(0)
            phase_qa_spend.append(0)
            phase_explore_spend.append(0)
            phase_lambda.append(0.0)
            phase_n_labeled_after.append(len(labeled_global))
            axis_lr_corr.append(float("nan"))
            continue

        explore_n = int(phase_total_budget * float(lr_explore_frac))
        explore_n = max(0, min(explore_n, phase_total_budget))
        vs_qa_budget = phase_total_budget - explore_n
        phase_vs_budget = max(1, int(vs_qa_budget * vs_ratio)) if vs_qa_budget > 0 else 0
        phase_qa_budget = max(0, vs_qa_budget - phase_vs_budget)

        lam = _activeSampling_blend_lambda(
            phase_idx, effective_n_phases, lr_blend_schedule,
            len(labeled_global), int(cold_start_phases),
        )
        if lr_model is not None and lam > 0.0:
            lr_score = lr_model.decision_function(embeddings)
            axis = _activeSampling_compute_phase_axis(similarities, lr_score, lam)
            corr_mat = np.corrcoef(similarities.astype(np.float64), lr_score.astype(np.float64))
            corr = float(corr_mat[0, 1]) if corr_mat.shape == (2, 2) else float("nan")
            axis_source = f"blend(lam={lam:.3f})"
        else:
            axis = similarities.astype(np.float64)
            corr = float("nan")
            axis_source = "siglip"

        last_axis = axis
        sort_order_p = np.argsort(-axis)
        pos_of_global = _activeSampling_inverse_permutation(sort_order_p)

        w_p = np.zeros(n_total, dtype=np.float64)
        labeled_mask_p = np.zeros(n_total, dtype=bool)
        if labeled_global:
            keys_arr = np.fromiter(labeled_global.keys(), dtype=np.int64)
            vals_arr = np.fromiter(
                (labeled_global[int(k)] for k in keys_arr.tolist()),
                dtype=np.int32,
            )
            positions = pos_of_global[keys_arr]
            labeled_mask_p[positions] = True
            w_p[positions] = np.where(vals_arr == 1, 1.0, -1.0)

        phase_B1 = max(1, min(effective_B1, max(1, phase_vs_budget)))

        result = _run_activeSampling_single_phase(
            sort_order_p, texts, label_fn, labeled_mask_p, w_p,
            B1=phase_B1, vs_budget=phase_vs_budget, qa_budget=phase_qa_budget,
            use_multi_interval=use_multi_interval, vs_k=vs_k,
            effective_k=effective_k, n_modes=n_modes,
            k_auto=k_auto, k_max=k_max, k_gain_tol=k_gain_tol,
            class_balance=class_balance,
        )

        labeled_mask_p = result["labeled_mask"]
        w_p = result["w"]
        new_positions = np.where(labeled_mask_p)[0]
        for pos in new_positions.tolist():
            g = int(sort_order_p[int(pos)])
            labeled_global[g] = int(1 if w_p[int(pos)] == 1.0 else 0)

        explore_spend = 0
        if explore_n > 0:
            unlabeled = np.array(
                [g for g in range(n_total) if g not in labeled_global],
                dtype=np.int64,
            )
            if len(unlabeled) > 0:
                pick_n = min(explore_n, len(unlabeled))
                picks = rng.choice(unlabeled, size=pick_n, replace=False)
                pick_texts = [texts[int(g)] for g in picks.tolist()]
                pick_labels = np.asarray(label_fn(pick_texts), dtype=np.int32)
                for g, lab in zip(picks.tolist(), pick_labels.tolist()):
                    labeled_global[int(g)] = int(lab)
                explore_spend = pick_n

        last_peaks = result["peaks"]
        last_per_peak_budget = result["per_peak_budget"]
        last_chosen_intervals = result["chosen_intervals"]
        last_chosen_acc = result["chosen_acc"]
        last_final_windows = result["final_windows"]
        last_final_pos_weight = result["final_pos_weight"]
        last_chosen_k = result["chosen_k"]
        last_sort_order = sort_order_p

        total_vs_time += result["vs_time"]
        total_qa_time += result["qa_time"]
        total_dp_time += result["dp_time"]
        phase_axis_source.append(axis_source)
        phase_vs_spend.append(int(result["vs_spend"]))
        phase_qa_spend.append(int(result["qa_spend"]))
        phase_explore_spend.append(int(explore_spend))
        phase_lambda.append(float(lam))
        phase_n_labeled_after.append(len(labeled_global))
        axis_lr_corr.append(corr)

        if phase_idx < effective_n_phases - 1:
            new_lr, refit_dt = _refit_lr()
            total_lr_refit_time += refit_dt
            if new_lr is not None:
                lr_model = new_lr
                n_lr_refits += 1

        logger.info(
            f"  phase {phase_idx}: axis={axis_source} "
            f"vs={result['vs_spend']} qa={result['qa_spend']} "
            f"explore={explore_spend} |labeled|={len(labeled_global)} "
            f"corr(sim,lr)={corr:.3f}"
        )

    if labeled_global:
        pos_of_last = _activeSampling_inverse_permutation(last_sort_order)
        keys_arr = np.fromiter(labeled_global.keys(), dtype=np.int64)
        positions_in_last = pos_of_last[keys_arr]
        order = np.argsort(positions_in_last, kind="stable")
        keys_final = keys_arr[order]
        vals_final = np.array(
            [labeled_global[int(k)] for k in keys_final.tolist()],
            dtype=np.int32,
        )
    else:
        keys_final = np.zeros(0, dtype=np.int64)
        vals_final = np.zeros(0, dtype=np.int32)
    labeled_indices = keys_final
    labeled_values = vals_final

    n_pos = int(np.sum(labeled_values == 1))
    logger.info(
        f"  active-learning sampling total: {len(labeled_indices)} labeled, {n_pos} positive "
        f"({100.0 * n_pos / max(len(labeled_indices), 1):.1f}%)"
    )

    profiling: Dict[str, Any] = {
        "graph_time": graph_time,
        "sample_size": sample_size,
        "n_sampled": len(labeled_indices),
        "n_pos": n_pos,
        "vs_time": float(total_vs_time),
        "vs_spend": int(sum(phase_vs_spend)),
        "qa_time": float(total_qa_time),
        "qa_spend": int(sum(phase_qa_spend)),
        "qa_peak_positions": [int(p) for p, _ in last_peaks],
        "qa_peak_scores": [float(s) for _, s in last_peaks],
        "qa_per_peak_budget": list(last_per_peak_budget),
        "n_modes": int(effective_k),
        "best_hyp_position": int(last_peaks[0][0]) if last_peaks else 0,
        "knn_k": knn_k,
        "observed_selectivity": n_pos / max(len(labeled_indices), 1),
        "k_intervals": int(effective_k),
        "vs_k": int(vs_k),
        "k_auto": bool(k_auto),
        "chosen_k": int(last_chosen_k),
        "dp_time": float(total_dp_time),
        "dp_acc": float(last_chosen_acc),
        "intervals_sorted_space": [
            [int(a), int(b)] for (a, b) in last_chosen_intervals
        ],
        "vs_final_windows_sorted_space": [
            [int(a), int(b)] for (a, b) in last_final_windows
        ],
        "class_balance": bool(class_balance),
        "pos_weight": float(last_final_pos_weight),
        "n_phases": int(effective_n_phases),
        "phase_axis_source": phase_axis_source,
        "phase_vs_spend": phase_vs_spend,
        "phase_qa_spend": phase_qa_spend,
        "phase_explore_spend": phase_explore_spend,
        "phase_lambda": phase_lambda,
        "phase_n_labeled_after": phase_n_labeled_after,
        "axis_lr_corr": axis_lr_corr,
        "lr_refit_time": float(total_lr_refit_time),
        "lr_n_refits": int(n_lr_refits),
        "lr_blend_schedule": lr_blend_schedule,
        "cold_start_phases": int(cold_start_phases),
        "lr_explore_frac": float(lr_explore_frac),
        "phase_budget_split": phase_budget_split,
        "phase_budgets": list(phase_budgets),
        "lr_C": float(lr_C),
        "last_phase_sort_order_first10": [
            int(x) for x in last_sort_order[:10].tolist()
        ],
        # Full final ordering axis σ (over the sampler's input space). The
        # caller extracts and removes this before profiling is serialized
        # (it is one float per candidate, too large for logs/CSV).
        "_final_axis": np.asarray(last_axis, dtype=np.float64).reshape(-1),
    }

    return labeled_indices, labeled_values, W, profiling


def _diffuse_with_early_exit(
    S: csr_matrix,
    init_values: np.ndarray,
    pinned_indices: np.ndarray,
    pinned_values: np.ndarray,
    alpha: float,
    num_iter: int,
    tol: float,
    patience: int,
    clip01: bool = False,
) -> Tuple[np.ndarray, int]:
    """Iterative diffusion on normalized graph with early exit.

    Updates: x_new = (1 - alpha) * x_init + alpha * S @ x
    When `clip01` is set, the affine update is clipped to [0, 1] every
    iteration BEFORE pinning (paper Alg. 3, line 53: the homogeneity-propagation
    step wraps the whole update in clip(.,0,1)). Bias diffusion (Phase A) leaves
    `clip01=False` because its residual update is unclipped in the paper.
    Pins `pinned_indices` to `pinned_values` after each update.
    Exits when max |x_new - x| < tol for `patience` consecutive iterations.

    Returns (final_values, iterations_run).
    """
    x = init_values.copy()
    x_init = init_values.copy()
    no_improvement = 0
    for i in range(num_iter):
        x_new = (1.0 - alpha) * x_init + alpha * (S @ x)
        if clip01:
            np.clip(x_new, 0.0, 1.0, out=x_new)
        if len(pinned_indices) > 0:
            x_new[pinned_indices] = pinned_values
        delta = float(np.max(np.abs(x_new - x))) if len(x) > 0 else 0.0
        x = x_new
        if delta < tol:
            no_improvement += 1
            if no_improvement >= patience:
                return x, i + 1
        else:
            no_improvement = 0
    return x, num_iter

def predict_with_score_refinement(
    embeddings: np.ndarray,
    W: csr_matrix,
    labeled_indices: np.ndarray,
    labeled_values: np.ndarray,
    confidence_threshold: float,
    alpha: float = 0.8,
    beta: float = 0.8,
    num_iter: int = 50,
    tol: float = 1e-4,
    patience: int = 5,
    imbalance_strategy: str = "balanced",
    lr_C: float = 1.0,
    base_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Bias Diffusion + Homogeneity Propagation: LR base + residual propagation + prediction smoothing.

    Faithful to the FDiff-scale variant of Huang et al. (ICLR 2021), Section 2.2:

    Stage A: train LR on (embeddings[labeled], labels) → base_probas for all rows.
    Stage B (Correct): propagate residuals R₀ = Y − Z on the kNN graph, with
        R[L] pinned to R₀[L] every iteration; write the corrected prediction
        only on unlabeled rows (z_corrected[U] = clip(Z[U] + R*[U], 0, 1));
        labeled rows keep Z[L].
    Stage C (Smooth): initialize G[L] = Y, G[U] = z_corrected[U] (Eq. (4) of
        the paper), then iterate G ← (1−β)G_init + β·S·G with G[L] pinned to Y.

    Returns (positive_mask, final_scores, profiling).
    """
    n = embeddings.shape[0]
    prof: Dict[str, Any] = {}

    # --- Stage A: LR base prediction ---
    t0 = time.time()
    unique_classes = np.unique(labeled_values)
    if len(unique_classes) < 2:
        single_class = int(unique_classes[0])
        logger.warning(
            f"Bias Diffusion + Homogeneity Propagation: only one class ({single_class}) in labeled set — "
            f"returning constant prediction"
        )
        base_probas = np.full(n, float(single_class), dtype=np.float64)
        prof["lr_fit_time"] = 0.0
        prof["lr_predict_time"] = time.time() - t0
        prof["lr_training_time"] = prof["lr_fit_time"] + prof["lr_predict_time"]
        prof["base_single_class"] = True
    else:
        _cw = None if imbalance_strategy == "not_balanced" else imbalance_strategy
        lr = LogisticRegression(
            class_weight=_cw, C=lr_C, max_iter=1000, solver="lbfgs",
        )
        t_fit = time.time()
        lr.fit(embeddings[labeled_indices], labeled_values)
        prof["lr_fit_time"] = time.time() - t_fit
        # Continuous proxy score Z = P(y=1 | embedding) for the bias-diffusion
        # residual R₀ = Y − Z (paper §3.3, Z ∈ [0,1]) — NOT the hard-label
        # argmax. predict_proba keeps the Bias Diffusion step propagating a calibrated
        # soft bias rather than 0/1 label flips.
        t_pred = time.time()
        _proba = lr.predict_proba(embeddings)
        if _proba.shape[1] == 2:
            _pos_col = list(lr.classes_).index(1) if 1 in lr.classes_ else 1
            base_probas = _proba[:, _pos_col].astype(np.float64)
        else:
            base_probas = _proba[:, 0].astype(np.float64)
        prof["lr_predict_time"] = time.time() - t_pred
        prof["lr_training_time"] = prof["lr_fit_time"] + prof["lr_predict_time"]
        prof["base_single_class"] = False
    # base_probas is now a continuous score in [0, 1] (predict_proba); count
    # predicted positives at the 0.5 cutoff for diagnostics only. The
    # base_threshold percentile knob only affects the FINAL cutoff on z_smooth
    # (and z_corrected) — see Stage B / Stage C below.
    n_base_positive = int(np.sum(base_probas > 0.5))
    prof["n_base_positive"] = n_base_positive
    prof["lr_C"] = float(lr_C)
    logger.info(
        f"Bias Diffusion + Homogeneity Propagation: LR base trained on {len(labeled_indices)} samples (C={lr_C}), "
        f"n_base_positive={n_base_positive}/{n} (predict_proba >= 0.5)"
    )

    # --- Graph normalization: S = D^(-1/2) W D^(-1/2) ---
    t0 = time.time()
    deg = np.array(W.sum(axis=1)).flatten()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-10))
    D_inv_sqrt = diags(d_inv_sqrt, format="csr")
    S = D_inv_sqrt @ W @ D_inv_sqrt
    prof["graph_norm_time"] = time.time() - t0

    pinned_idx = np.asarray(labeled_indices, dtype=np.int64)
    pinned_labels = labeled_values.astype(np.float64)

    # --- Stage B: Bias Diffusion (propagate residuals) ---
    t0 = time.time()
    E_init = np.zeros(n, dtype=np.float64)
    E_init[pinned_idx] = pinned_labels - base_probas[pinned_idx]
    pinned_residuals = E_init[pinned_idx].copy()
    E_final, bias_diffusion_iters = _diffuse_with_early_exit(
        S, E_init, pinned_idx, pinned_residuals,
        alpha=alpha, num_iter=num_iter, tol=tol, patience=patience,
    )
    # FDiff-scale (Huang et al. 2021, Sec. 2.2): write the corrected prediction
    # only on unlabeled rows; labeled rows keep their base prediction Z[L].
    # The labels are subsequently rewritten to Y at the start of the Smooth
    # step (Eq. (4) of the original paper); see Stage C below.
    z_corrected = base_probas.copy().astype(np.float64)
    if len(pinned_idx) > 0:
        unlabeled_mask = np.ones(n, dtype=bool)
        unlabeled_mask[pinned_idx] = False
        z_corrected[unlabeled_mask] = np.clip(
            base_probas[unlabeled_mask] + E_final[unlabeled_mask], 0.0, 1.0,
        )
    else:
        z_corrected = np.clip(base_probas + E_final, 0.0, 1.0)
    prof["bias_diffusion_time"] = time.time() - t0
    prof["bias_diffusion_iters"] = bias_diffusion_iters
    # When base_threshold is a percentile, also apply it to z_corrected for
    # a consistent top (100-P)% diagnostic.  Otherwise fall back to the
    # absolute confidence_threshold.
    if base_threshold is not None:
        _pct = float(base_threshold)
        _pct = 0.0 if _pct < 0.0 else (100.0 if _pct > 100.0 else _pct)
        corrected_threshold = float(np.quantile(z_corrected, _pct / 100.0))
    else:
        corrected_threshold = float(confidence_threshold)
    n_diffusion_positive = int(np.sum(z_corrected >= corrected_threshold))
    prof["n_diffusion_positive"] = n_diffusion_positive
    prof["n_diffusion_added"] = n_diffusion_positive - n_base_positive
    prof["corrected_threshold_used"] = corrected_threshold
    logger.info(
        f"Bias Diffusion done in {bias_diffusion_iters} iters, "
        f"z_corrected range [{float(z_corrected.min()):.3f}, {float(z_corrected.max()):.3f}], "
        f"n_diffusion_positive={n_diffusion_positive} (delta={prof['n_diffusion_added']:+d}, "
        f"threshold={corrected_threshold:.3f})"
    )

    # --- Stage C: Homogeneity Propagation (propagate corrected predictions) ---
    # Eq. (4) of Huang et al. 2021: initialize G[L] = Y, G[U] = z_corrected[U].
    # This is required because z_corrected now keeps Z[L] (the base prediction)
    # on labeled rows under the FDiff-scale convention above.
    t0 = time.time()
    G_init = z_corrected.copy()
    if len(pinned_idx) > 0:
        G_init[pinned_idx] = pinned_labels
    z_smooth, homogeneity_propagation_iters = _diffuse_with_early_exit(
        S, G_init, pinned_idx, pinned_labels,
        alpha=beta, num_iter=num_iter, tol=tol, patience=patience,
        clip01=True,  # paper Alg. 3 line 53: clip((1-beta)G + beta*S*z, 0, 1) each iter
    )
    prof["homogeneity_propagation_time"] = time.time() - t0
    prof["homogeneity_propagation_iters"] = homogeneity_propagation_iters
    prof["z_corrected"] = z_corrected
    logger.info(
        f"Homogeneity Propagation done in {homogeneity_propagation_iters} iters, "
        f"z_smooth range [{float(z_smooth.min()):.3f}, {float(z_smooth.max()):.3f}]"
    )

    # --- Threshold ---
    # When base_threshold is set as percentile P, use the same percentile on
    # the final z_smooth so the output keeps the top (100-P)% of predictions.
    # This is what makes the "keep only >p95 data" promise actually hold end-
    # to-end (the earlier base_probas cutoff was only diagnostic).
    if base_threshold is not None:
        _pct = float(base_threshold)
        _pct = 0.0 if _pct < 0.0 else (100.0 if _pct > 100.0 else _pct)
        final_threshold = float(np.quantile(z_smooth, _pct / 100.0))
        final_threshold_source = f"p{_pct:.1f} of z_smooth"
    else:
        final_threshold = float(confidence_threshold)
        final_threshold_source = f"confidence_threshold={confidence_threshold}"
    positive_mask = z_smooth >= final_threshold
    n_pred_pos = int(np.sum(positive_mask))
    prof["n_propagation_added"] = n_pred_pos - n_diffusion_positive
    prof["final_threshold_used"] = final_threshold
    prof["final_threshold_source"] = final_threshold_source
    logger.info(
        f"Refinement stage times: LR={prof['lr_training_time']:.3f}s  "
        f"graph_norm={prof['graph_norm_time']:.3f}s  "
        f"correct={prof['bias_diffusion_time']:.3f}s ({bias_diffusion_iters} iters)  "
        f"smooth={prof['homogeneity_propagation_time']:.3f}s ({homogeneity_propagation_iters} iters)"
    )
    logger.info(
        f"Bias Diffusion + Homogeneity Propagation: predicted {n_pred_pos}/{n} positive "
        f"(threshold={final_threshold:.3f} from {final_threshold_source}), "
        f"base={n_base_positive} → correct={n_diffusion_positive} ({prof['n_diffusion_added']:+d}) "
        f"→ smooth={n_pred_pos} ({prof['n_propagation_added']:+d})"
    )
    prof["n_predicted_positive"] = n_pred_pos
    return positive_mask, z_smooth, prof


def _compute_oof_z_smooth_and_corrected(
    embeddings: np.ndarray,
    W: csr_matrix,
    labeled_indices: np.ndarray,
    labeled_values: np.ndarray,
    *,
    n_folds: int = 5,
    confidence_threshold: float = 0.5,
    diffusion_alpha: float = 0.8,
    propagation_beta: float = 0.8,
    refine_num_iter: int = 50,
    refine_tol: float = 1e-4,
    refine_patience: int = 5,
    imbalance_strategy: str = "balanced",
    lr_C: float = 1.0,
    base_threshold: Optional[float] = None,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return OOF (z_smooth, z_corrected) for each labelled row.

    Each labelled row is scored by a Bias Diffusion + Homogeneity Propagation model whose pinned set excludes
    that row, removing the in-sample pinning bias before learned-deferral
    training.
    """
    idx = np.asarray(labeled_indices, dtype=np.int64).reshape(-1)
    y = np.asarray(labeled_values, dtype=np.int32).reshape(-1)
    if idx.shape[0] != y.shape[0]:
        raise ValueError("labeled_indices and labeled_values length mismatch")
    m = int(idx.shape[0])
    if m == 0:
        raise ValueError("OOF Bias Diffusion + Homogeneity Propagation requires at least one labelled row")

    z_smooth_oof = np.zeros(m, dtype=np.float64)
    z_corrected_oof = np.zeros(m, dtype=np.float64)
    n_splits = max(2, min(int(n_folds), m)) if m > 1 else 1

    if n_splits == 1:
        train_splits = [(np.zeros(0, dtype=np.int64), np.arange(m, dtype=np.int64))]
    else:
        counts = np.bincount(y.astype(np.int32), minlength=2)
        if int(counts.min()) >= 2:
            k = min(n_splits, int(counts.min()))
            splitter = StratifiedKFold(
                n_splits=k, shuffle=True, random_state=int(random_seed)
            )
            train_splits = list(splitter.split(idx, y))
        else:
            k = min(n_splits, m)
            splitter = KFold(n_splits=k, shuffle=True, random_state=int(random_seed))
            train_splits = list(splitter.split(idx))

    fold_sizes: List[int] = []
    fold_times: List[float] = []
    fold_wrong_rates: List[float] = []
    for fold_id, (train_pos, hold_pos) in enumerate(train_splits):
        hold_pos = np.asarray(hold_pos, dtype=np.int64)
        train_pos = np.asarray(train_pos, dtype=np.int64)
        if train_pos.size == 0:
            train_idx = idx[hold_pos]
            train_y = y[hold_pos]
        else:
            train_idx = idx[train_pos]
            train_y = y[train_pos]
        t0 = time.time()
        _, z_smooth_fold, prof_fold = predict_with_score_refinement(
            embeddings,
            W,
            train_idx,
            train_y,
            confidence_threshold=confidence_threshold,
            alpha=diffusion_alpha,
            beta=propagation_beta,
            num_iter=refine_num_iter,
            tol=refine_tol,
            patience=refine_patience,
            imbalance_strategy=imbalance_strategy,
            lr_C=lr_C,
            base_threshold=base_threshold,
        )
        z_corrected_fold = np.asarray(
            prof_fold.get("z_corrected", z_smooth_fold), dtype=np.float64
        ).reshape(-1)
        hold_idx = idx[hold_pos]
        z_smooth_oof[hold_pos] = z_smooth_fold[hold_idx]
        z_corrected_oof[hold_pos] = z_corrected_fold[hold_idx]
        wrong = (z_smooth_oof[hold_pos] >= confidence_threshold).astype(np.int32) != y[
            hold_pos
        ]
        fold_sizes.append(int(hold_pos.size))
        fold_times.append(time.time() - t0)
        fold_wrong_rates.append(float(np.mean(wrong)) if wrong.size else 0.0)
        logger.info(
            f"[learned_deferral] OOF fold {fold_id + 1}/{len(train_splits)}: "
            f"holdout={hold_pos.size}, wrong_rate={fold_wrong_rates[-1]:.3f}, "
            f"time={fold_times[-1]:.2f}s"
        )

    info = {
        "n_folds": int(len(train_splits)),
        "fold_sizes": fold_sizes,
        "fold_times": fold_times,
        "fold_wrong_rates": fold_wrong_rates,
        "mean_wrong_rate": float(np.mean(fold_wrong_rates)) if fold_wrong_rates else 0.0,
        "total_time": float(sum(fold_times)),
    }
    return z_smooth_oof, z_corrected_oof, info


def auto_kmeans_k(n: int) -> int:
    """Heuristic KMeans K: round(sqrt(N/2)), capped [5, 200]."""
    return min(max(5, round(math.sqrt(n / 2))), 200)


def auto_knn_k(n: int) -> int:
    """Heuristic KNN K: max(5, min(round(log2(N)*3), 50))."""
    if n <= 1:
        return 1
    return max(5, min(round(math.log2(n) * 3), 50))


def label_propagation_cutssl(
    embeddings: np.ndarray,
    labeled_indices: np.ndarray,
    labeled_values: np.ndarray,
    class_prior: float = 0.0,
    knn_k: int = 0,
    s_values: Optional[List[float]] = None,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """CutSSL-inspired label propagation.

    Args:
        embeddings: (N, d) array.
        labeled_indices: Indices of labeled rows.
        labeled_values: 0/1 labels.
        class_prior: Estimated positive rate. 0 = estimate from labeled set.
        knn_k: KNN neighbors for graph. 0 = auto: max(5, min(log2(N)*3, 50)).
        s_values: Regularization strengths. None = [0.0, 0.1].
        max_iter: Max CG iterations per regularization.

    Returns:
        pseudo_labels: (N,) predicted 0/1 labels for all rows.
        confidence_scores: (N,) confidence in [0, 1].
    """
    if s_values is None:
        s_values = [0.0, 0.1]

    n_total = embeddings.shape[0]

    if knn_k <= 0:
        knn_k = auto_knn_k(n_total)

    if class_prior <= 0.0:
        n_pos = int(np.sum(labeled_values == 1))
        n_labeled = len(labeled_values)
        class_prior = n_pos / max(n_labeled, 1)
        class_prior = max(class_prior, 1e-4)
    logger.info(
        f"Label propagation: N={n_total}, labeled={len(labeled_indices)}, "
        f"class_prior={class_prior:.4f}, knn_k={knn_k}"
    )

    t0 = time.time()
    W = _build_knn_graph(embeddings, knn_k)
    graph_time = time.time() - t0
    logger.info(f"KNN graph built in {graph_time:.2f}s (nnz={W.nnz})")  # detail logged inside _build_knn_graph

    L = _compute_graph_laplacian(W)

    t0 = time.time()
    f = _solve_constrained_propagation(
        L, labeled_indices, labeled_values, class_prior, n_total, s_values, max_iter,
    )
    solve_time = time.time() - t0
    logger.info(f"Propagation solved in {solve_time:.2f}s")

    pseudo_labels = (f >= 0.5).astype(np.int32)
    confidence_scores = np.abs(f - 0.5) * 2.0

    n_prop_pos = int(np.sum(pseudo_labels == 1))
    logger.info(
        f"Propagation result: {n_prop_pos}/{n_total} positive "
        f"({100.0 * n_prop_pos / max(n_total, 1):.2f}%)"
    )

    return pseudo_labels, confidence_scores


# ---------------------------------------------------------------------------
# Phase E: Final prediction modes
# ---------------------------------------------------------------------------


def _predict_with_propagation(
    pseudo_labels: np.ndarray,
    confidence_scores: np.ndarray,
    labeled_indices: np.ndarray,
    labeled_values: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    """activeSampling propagation mode: use propagated labels with confidence filter.

    Returns boolean mask of positive predictions.
    """
    final_labels = pseudo_labels.copy()
    final_labels[labeled_indices] = labeled_values.astype(np.int32)

    final_confidence = confidence_scores.copy()
    final_confidence[labeled_indices] = 1.0

    mask = (final_labels == 1) & (final_confidence >= confidence_threshold)
    return mask


def _predict_with_weighted_lr(
    embeddings: np.ndarray,
    pseudo_labels: np.ndarray,
    confidence_scores: np.ndarray,
    labeled_indices: np.ndarray,
    labeled_values: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_stats: Dict[int, Dict[str, Any]],
    confidence_threshold: float,
) -> np.ndarray:
    """Weighted LR mode: train LR on all rows using confidence-based weights.

    Returns boolean mask of positive predictions.
    """
    n = len(pseudo_labels)
    all_labels = pseudo_labels.copy()
    all_labels[labeled_indices] = labeled_values.astype(np.int32)

    sample_weights = np.zeros(n, dtype=np.float64)
    labeled_set = set(labeled_indices.tolist())

    pure_clusters = set()
    for k, stats in cluster_stats.items():
        if stats["classification"] in (CLUSTER_PURE_POS, CLUSTER_PURE_NEG):
            pure_clusters.add(k)

    for i in range(n):
        if i in labeled_set:
            sample_weights[i] = 1.0
        elif cluster_labels[i] in pure_clusters:
            sample_weights[i] = float(confidence_scores[i]) * 0.8
        else:
            sample_weights[i] = float(confidence_scores[i]) * 0.5

    sample_weights = np.maximum(sample_weights, 1e-6)

    unique_classes = np.unique(all_labels)
    if len(unique_classes) < 2:
        return all_labels == 1

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
    lr.fit(embeddings, all_labels, sample_weight=sample_weights)

    probas = lr.predict_proba(embeddings)
    pos_idx = list(lr.classes_).index(1) if 1 in lr.classes_ else 0
    pos_probas = probas[:, pos_idx]
    return pos_probas >= confidence_threshold


# ---------------------------------------------------------------------------
# Phase E.3: Full pipeline orchestrator
# ---------------------------------------------------------------------------


def cluster_proxy_pipeline(
    embeddings: np.ndarray,
    texts: List[Union[str, list]],
    label_fn_factory: Callable[[], LabelFnType],
    sample_size: int,
    cluster_proxy_mode: str = "propagation",
    knn_k: int = 0,
    confidence_threshold: float = 0.5,
    lr_features: Optional[np.ndarray] = None,
    sampling_method: str = "random",
    similarities: Optional[np.ndarray] = None,
    n_strata: int = 10,
    bas_pilot_ratio: float = 0.15,
    allocation_method: str = "neyman",
    reg_method: str = "exponential",
    reg_strength: float = 0.5,
    n_clusters: int = 0,
    auto_k: bool = False,
    refine: bool = False,
    diffusion_alpha: float = 0.8,
    propagation_beta: float = 0.8,
    refine_num_iter: int = 50,
    refine_tol: float = 1e-4,
    refine_patience: int = 5,
    refine_enabled: bool = True,
    activeSampling_B1: int = 5,
    activeSampling_vs_ratio: float = 0.5,
    activeSampling_n_modes: int = 1,
    activeSampling_k_intervals: Optional[int] = None,
    activeSampling_k_auto: bool = False,
    activeSampling_k_max: int = 4,
    activeSampling_k_gain_tol: float = 0.02,
    activeSampling_class_balance: bool = False,
    activeSampling_n_phases: int = 1,
    activeSampling_lr_blend_schedule: str = "linear",
    activeSampling_cold_start_phases: int = 1,
    activeSampling_lr_explore_frac: float = 0.0,
    activeSampling_phase_budget_split: str = "equal",
    # uqe_active (UQE Algorithm 2, arXiv:2407.09522 §4.1.2 / §C.2)
    uqe_batch_size: int = 16,
    uqe_exploration_sigma_0: float = 0.0,
    uqe_exploration_decay: float = 0.9,
    # defer dual-threshold (replaces single-tau on the active-learning sampling + Bias Diffusion + Homogeneity Propagation path)
    defer_enabled: bool = False,
    verify_uncertain: bool = True,
    decision_mode: str = "force_label",
    deferral_features: Optional[Sequence[str]] = None,
    # Router π_i rank axis (paper §3.4): "blended" = rank on the sampler's
    # final blended ordering axis σ (default); "similarity" = force rank on raw
    # cosine similarity regardless of the sampler axis.
    deferral_rank_axis: str = "blended",
    deferral_hidden_dims_grid: Optional[Sequence[Sequence[int]]] = None,
    deferral_alpha_grid: Optional[Sequence[float]] = None,
    deferral_k_folds: int = 5,
    deferral_inner_cv_k: int = 3,
    deferral_dagger_beta: float = 0.00,
    deferral_min_inner_cv_auc: float = 0.60,
    deferral_min_probe_auc: float = 0.60,
    lr_C: float = 1.0,
    base_threshold: Optional[float] = None,
    lr_class_weight_override: Optional[str] = None,
    random_seed: Optional[int] = 42,
    viz_tag: str = "",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sample + predict pipeline.

    Supported sampling methods: "random", "activeSampling", "stratified",
    "kmeans_stratified".

    Args:
        embeddings: (N, d) row/pair embeddings — used for sampling and the
            KNN graph feeding Bias Diffusion + Homogeneity Propagation / label propagation.
        texts: LLM message for each row/pair.
        label_fn_factory: Callable() -> label_fn, where label_fn(texts) -> labels.
        sample_size: Total LLM call budget for sampling.
        cluster_proxy_mode: "propagation" or "weighted_lr"
            (only used by stratified / kmeans_stratified).
        knn_k: KNN neighbors for the propagation / Bias Diffusion + Homogeneity Propagation graph (0=auto).
        confidence_threshold: Decision threshold.
        lr_features: Optional (N, f) feature array for LR training.
            When provided, LR trains on these features instead of embeddings.
        refine_enabled: True = Bias Diffusion + Homogeneity Propagation post-process, False = LR-only.

    Returns:
        positive_mask: (N,) boolean mask of predicted positives.
        profiling_info: Dict with timing and statistics.
    """
    profiling: Dict[str, Any] = {}
    vllm_time_acc = [0.0]
    n_total = embeddings.shape[0]

    raw_label_fn = label_fn_factory()

    def label_fn(texts: List[Union[str, list]]) -> np.ndarray:
        t0 = time.time()
        result = raw_label_fn(texts)
        vllm_time_acc[0] += time.time() - t0
        return result

    profiling["sampling_method"] = sampling_method
    if random_seed is not None:
        np.random.seed(int(random_seed))
        import random as _py_random
        _py_random.seed(int(random_seed))
        profiling["random_seed"] = int(random_seed)
        logger.info(f"Pipeline random_seed={int(random_seed)} (global numpy + stdlib random seeded)")
    activeSampling_W: Optional[csr_matrix] = None
    random_W: Optional[csr_matrix] = None
    # Final blended ordering axis σ from the active-learning sampling sampler, used for the
    # router's π_i rank feature (paper §3.4). Populated below for the activeSampling
    # path; stays None (→ similarity-rank fallback) otherwise.
    activeSampling_final_axis: Optional[np.ndarray] = None
    learned_deferral_active = (
        bool(defer_enabled)
        and str(decision_mode) == "learned_deferral"
        and sampling_method == "activeSampling"
    )
    probe_indices = np.zeros(0, dtype=np.int64)
    probe_labels = np.zeros(0, dtype=np.int32)
    activeSampling_exclude_mask = np.zeros(n_total, dtype=bool)
    activeSampling_sample_size = int(sample_size)
    if learned_deferral_active and float(deferral_dagger_beta) > 0.0 and n_total > 0:
        requested_probe = max(
            8, int(math.ceil(float(deferral_dagger_beta) * max(int(sample_size), 1)))
        )
        max_probe = max(0, min(n_total, int(sample_size) - 8))
        if max_probe > 0:
            n_probe = min(requested_probe, max_probe)
            rng_probe = np.random.default_rng(
                int(random_seed) if random_seed is not None else 42
            )
            probe_indices = np.sort(
                rng_probe.choice(n_total, size=n_probe, replace=False).astype(np.int64)
            )
            activeSampling_exclude_mask[probe_indices] = True
            activeSampling_sample_size = max(1, int(sample_size) - int(n_probe))
            profiling["deferral_probe_n"] = int(n_probe)
            profiling["deferral_probe_beta"] = float(deferral_dagger_beta)
            profiling["deferral_activeSampling_sample_budget"] = int(activeSampling_sample_size)
        else:
            profiling["deferral_probe_n"] = 0
            profiling["deferral_probe_beta"] = float(deferral_dagger_beta)

    if sampling_method == "stratified":
        if similarities is None:
            raise ValueError("similarities array is required for sampling_strategy='stratified'")
        t0 = time.time()
        labeled_indices, labeled_values, strat_prof = sample_stratified_by_similarity(
            similarities, texts, label_fn,
            sample_size, n_strata, bas_pilot_ratio, allocation_method,
        )
        profiling["sampling_time"] = time.time() - t0
        profiling["stratum_time"] = float(strat_prof.get("stratum_time", 0.0))
        profiling["sampling_time"] = max(0.0, profiling["sampling_time"] - profiling["stratum_time"])
        profiling["n_labeled"] = int(len(labeled_indices))
        n_pos_labeled = int(np.sum(labeled_values == 1))
        profiling["n_pos_labeled"] = n_pos_labeled
        profiling["stratified_profiling"] = strat_prof
        cluster_stats = {}
        final_cluster_labels = np.zeros(n_total, dtype=np.int32)
    elif sampling_method == "kmeans_stratified":
        if similarities is None:
            raise ValueError("similarities array is required for sampling_method='kmeans_stratified'")
        t0 = time.time()
        labeled_indices, labeled_values, ks_prof = sample_kmeans_stratified(
            embeddings, similarities, texts, label_fn,
            sample_size, n_clusters=n_clusters,
            pilot_ratio=bas_pilot_ratio,
            reg_method=reg_method,
            reg_strength=reg_strength,
            auto_k=auto_k,
            refine=refine,
        )
        total_time = time.time() - t0
        profiling["stratum_time"] = float(ks_prof.get("stratum_time", 0.0))
        profiling["sampling_time"] = max(0.0, total_time - profiling["stratum_time"])
        profiling["n_labeled"] = int(len(labeled_indices))
        n_pos_labeled = int(np.sum(labeled_values == 1))
        profiling["n_pos_labeled"] = n_pos_labeled
        profiling["kmeans_stratified_profiling"] = ks_prof
        cluster_stats = {}
        final_cluster_labels = np.zeros(n_total, dtype=np.int32)
    elif sampling_method == "activeSampling":
        if similarities is None:
            raise ValueError("similarities array is required for sampling_method='activeSampling'")

        t0 = time.time()
        if learned_deferral_active and probe_indices.size > 0:
            eligible_idx = np.where(~activeSampling_exclude_mask)[0].astype(np.int64)
            sample_embeddings = embeddings[eligible_idx]
            sample_similarities = similarities[eligible_idx]
            sample_texts = [texts[int(i)] for i in eligible_idx.tolist()]
            local_labeled, labeled_values, activeSampling_W, dir_prof = sample_activeSampling(
                sample_embeddings, sample_similarities, sample_texts, label_fn,
                activeSampling_sample_size, knn_k=knn_k, B1=activeSampling_B1,
                vs_ratio=activeSampling_vs_ratio, n_modes=activeSampling_n_modes,
                k_intervals=activeSampling_k_intervals,
                k_auto=activeSampling_k_auto,
                k_max=activeSampling_k_max,
                k_gain_tol=activeSampling_k_gain_tol,
                class_balance=activeSampling_class_balance,
                n_phases=activeSampling_n_phases,
                lr_blend_schedule=activeSampling_lr_blend_schedule,
                cold_start_phases=activeSampling_cold_start_phases,
                lr_explore_frac=activeSampling_lr_explore_frac,
                phase_budget_split=activeSampling_phase_budget_split,
                lr_C=lr_C,
            )
            labeled_indices = eligible_idx[np.asarray(local_labeled, dtype=np.int64)]
            probe_texts = [texts[int(i)] for i in probe_indices.tolist()]
            probe_labels = np.asarray(label_fn(probe_texts), dtype=np.int32)
            dir_prof["deferral_probe_excluded"] = int(probe_indices.size)
            dir_prof["deferral_probe_labels_pos"] = int((probe_labels == 1).sum())
            if activeSampling_W is not None and activeSampling_W.shape[0] != n_total:
                activeSampling_W = None
        else:
            labeled_indices, labeled_values, activeSampling_W, dir_prof = sample_activeSampling(
                embeddings, similarities, texts, label_fn,
                sample_size, knn_k=knn_k, B1=activeSampling_B1,
                vs_ratio=activeSampling_vs_ratio, n_modes=activeSampling_n_modes,
                k_intervals=activeSampling_k_intervals,
                k_auto=activeSampling_k_auto,
                k_max=activeSampling_k_max,
                k_gain_tol=activeSampling_k_gain_tol,
                class_balance=activeSampling_class_balance,
                n_phases=activeSampling_n_phases,
                lr_blend_schedule=activeSampling_lr_blend_schedule,
                cold_start_phases=activeSampling_cold_start_phases,
                lr_explore_frac=activeSampling_lr_explore_frac,
                phase_budget_split=activeSampling_phase_budget_split,
                lr_C=lr_C,
            )
        total_time = time.time() - t0
        profiling["stratum_time"] = float(dir_prof.get("graph_time", 0.0))
        profiling["sampling_time"] = max(0.0, total_time - profiling["stratum_time"])
        profiling["n_labeled"] = int(len(labeled_indices))
        n_pos_labeled = int(np.sum(labeled_values == 1))
        profiling["n_pos_labeled"] = n_pos_labeled
        # Extract the sampler's final ordering axis σ for the router π_i rank
        # (paper §3.4) before profiling is serialized. Only usable when it
        # spans all n_total candidates (i.e. the non-probe path); the probe
        # path samples a subset, so its axis is in subset space and we fall
        # back to the similarity rank.
        _final_axis = dir_prof.pop("_final_axis", None)
        if _final_axis is not None and len(_final_axis) == n_total:
            activeSampling_final_axis = np.asarray(_final_axis, dtype=np.float64).reshape(-1)
        profiling["activeSampling_profiling"] = dir_prof
        cluster_stats = {}
        final_cluster_labels = np.zeros(n_total, dtype=np.int32)

        # ----------------------------------------------------------------
        # Router 6:4 stratified disjoint split (paper §3.4). When the
        # learned-deferral router is active, split the actively-sampled
        # labels once into L_proxy (pins the Bias Diffusion + Homogeneity Propagation call below) and a held-out
        # L_router (trains the router). Stratified to preserve the
        # positive:negative ratio with a floor of 2 router positives. Because
        # L_router is excluded from proxy training and propagation pinning,
        # its propagated scores z^prop are unbiased and need no OOF. Falls
        # back to the 5-fold OOF path (no override) when a disjoint split is
        # infeasible (too few labels / positives).
        # ----------------------------------------------------------------
        ld_router_indices = np.zeros(0, dtype=np.int64)
        ld_router_values = np.zeros(0, dtype=np.int32)
        ld_full_labeled_indices = labeled_indices.copy()
        ld_full_labeled_values = labeled_values.copy()
        ld_split_succeeded = False
        if learned_deferral_active:
            from learned_deferral import stratified_proxy_router_split

            proxy_mask_ld, router_mask_ld, ld_split_info = (
                stratified_proxy_router_split(
                    labeled_values,
                    router_ratio=0.4,
                    min_router_pos=2,
                    random_state=int(random_seed) if random_seed is not None else 42,
                )
            )
            profiling["router_split_info"] = dict(ld_split_info)
            if ld_split_info.get("feasible"):
                ld_router_indices = labeled_indices[router_mask_ld]
                ld_router_values = labeled_values[router_mask_ld]
                # Override for downstream Bias Diffusion + Homogeneity Propagation: only L_proxy is pinned.
                labeled_indices = labeled_indices[proxy_mask_ld]
                labeled_values = labeled_values[proxy_mask_ld]
                ld_split_succeeded = True
                logger.info(
                    f"[router] 6:4 split labeled={ld_split_info['n_total_labeled']} -> "
                    f"L_proxy(pos={ld_split_info['n_proxy_pos']}, "
                    f"neg={ld_split_info['n_proxy_neg']}, "
                    f"total={ld_split_info['n_proxy']}) | "
                    f"L_router(pos={ld_split_info['n_router_pos']}, "
                    f"neg={ld_split_info['n_router_neg']}, "
                    f"total={ld_split_info['n_router']})"
                )
            else:
                logger.warning(
                    "[router] 6:4 split infeasible (%s); falling back to "
                    "5-fold OOF over the full labeled set",
                    ld_split_info.get("fallback_reason"),
                )
        profiling["router_split_succeeded"] = ld_split_succeeded
    elif sampling_method == "random":
        t0 = time.time()
        _rnd_seed = int(np.random.randint(0, 2**31 - 1))
        labeled_indices, labeled_values, random_W, rnd_prof = sample_random_pure(
            embeddings, texts, label_fn, sample_size, seed=_rnd_seed,
        )
        total_time = time.time() - t0
        profiling["stratum_time"] = 0.0
        profiling["sampling_time"] = total_time
        profiling["n_labeled"] = int(len(labeled_indices))
        n_pos_labeled = int(np.sum(labeled_values == 1))
        profiling["n_pos_labeled"] = n_pos_labeled
        profiling["random_profiling"] = rnd_prof
        cluster_stats = {}
        final_cluster_labels = np.zeros(n_total, dtype=np.int32)
    elif sampling_method == "top_k":
        # Top-K similarity baseline (Proxy-100x §5.4). Deterministic.
        if similarities is None:
            raise ValueError(
                "similarities array is required for sampling_method='top_k'"
            )
        t0 = time.time()
        labeled_indices, labeled_values, _topk_W, topk_prof = sample_top_k_similarity(
            embeddings, similarities, texts, label_fn, sample_size,
        )
        total_time = time.time() - t0
        profiling["stratum_time"] = 0.0
        profiling["sampling_time"] = total_time
        profiling["n_labeled"] = int(len(labeled_indices))
        n_pos_labeled = int(np.sum(labeled_values == 1))
        profiling["n_pos_labeled"] = n_pos_labeled
        profiling["top_k_profiling"] = topk_prof
        cluster_stats = {}
        final_cluster_labels = np.zeros(n_total, dtype=np.int32)
    elif sampling_method == "uqe_active":
        # UQE Algorithm 2 (literal implementation per arXiv:2407.09522 §4.1.2).
        # Spec: docs/uqe_active_implementation.md
        t0 = time.time()
        labeled_indices, labeled_values, _uqe_W, uqe_prof = sample_uqe_active(
            embeddings,
            similarities,
            texts,
            label_fn,
            sample_size,
            batch_size=int(uqe_batch_size),
            exploration_sigma_0=float(uqe_exploration_sigma_0),
            exploration_decay=float(uqe_exploration_decay),
            lr_C=float(lr_C),
            random_seed=int(random_seed) if random_seed is not None else 42,
        )
        total_time = time.time() - t0
        profiling["stratum_time"] = 0.0
        profiling["sampling_time"] = total_time
        profiling["n_labeled"] = int(len(labeled_indices))
        n_pos_labeled = int(np.sum(labeled_values == 1))
        profiling["n_pos_labeled"] = n_pos_labeled
        profiling["uqe_active_profiling"] = uqe_prof
        # uqe_active does not build its own KNN graph; Bias Diffusion + Homogeneity Propagation will build it.
        cluster_stats = {}
        final_cluster_labels = np.zeros(n_total, dtype=np.int32)
    else:
        raise ValueError(
            f"Unknown sampling_method={sampling_method!r}. Supported: "
            "random, activeSampling, stratified, kmeans_stratified, uqe_active, top_k."
        )

    # activeSampling/random/uqe_active/top_k use Bias Diffusion + Homogeneity Propagation (LR base +
    # residual propagation + prediction smoothing) when refine_enabled is
    # True; stratified and kmeans_stratified use the classic LP + final
    # prediction pipeline.
    _refine_methods = ("activeSampling", "random", "uqe_active", "top_k")
    lr_input = lr_features if lr_features is not None else embeddings
    profiling["feature_mode"] = "similarity" if lr_features is not None else "embedding"

    if sampling_method == "activeSampling":
        _prebuilt_W = activeSampling_W
    elif sampling_method == "random":
        _prebuilt_W = random_W
    else:
        # uqe_active and any other method without a prebuilt W
        _prebuilt_W = None

    if sampling_method in _refine_methods and refine_enabled:
        if _prebuilt_W is None:
            _knn = knn_k if knn_k > 0 else auto_knn_k(n_total)
            logger.info(f"Bias Diffusion + Homogeneity Propagation needs KNN graph but none pre-built — building (k={_knn})...")
            t0_knn = time.time()
            _knn_sub: Dict[str, float] = {}
            _prebuilt_W = _build_knn_graph(
                embeddings, _knn, viz_tag=viz_tag, sub_timings=_knn_sub,
            )
            profiling["knn_graph_build_time"] = time.time() - t0_knn
            profiling["knn_graph_build_detail"] = _knn_sub
            profiling["knn_cache_hit"] = bool(_knn_sub.get("cache_hit", False))
        else:
            profiling["knn_graph_build_time"] = 0.0
            profiling["knn_graph_build_detail"] = {}
            profiling["knn_cache_hit"] = False
        t0 = time.time()
        if lr_class_weight_override in ("balanced", "not_balanced"):
            _refine_imbalance = lr_class_weight_override
        else:
            _refine_imbalance = "not_balanced" if sampling_method == "random" else "balanced"
        profiling["lr_class_weight_used"] = _refine_imbalance
        positive_mask, _final_scores, refine_prof = predict_with_score_refinement(
            lr_input, _prebuilt_W, labeled_indices, labeled_values,
            confidence_threshold=confidence_threshold,
            alpha=diffusion_alpha, beta=propagation_beta,
            num_iter=refine_num_iter, tol=refine_tol, patience=refine_patience,
            imbalance_strategy=_refine_imbalance,
            lr_C=lr_C,
            base_threshold=base_threshold,
        )
        refine_total = time.time() - t0
        profiling["propagation_time"] = (
            float(refine_prof.get("bias_diffusion_time", 0.0))
            + float(refine_prof.get("homogeneity_propagation_time", 0.0))
            + float(refine_prof.get("graph_norm_time", 0.0))
        )
        profiling["prediction_time"] = max(
            0.0, refine_total - profiling["propagation_time"]
            - float(refine_prof.get("lr_training_time", 0.0)),
        )
        profiling["lr_training_time"] = float(refine_prof.get("lr_training_time", 0.0))
        profiling["refinement_profiling"] = refine_prof

        if defer_enabled and sampling_method == "activeSampling":
            # defer dual-threshold for the active-learning sampling + Bias Diffusion + Homogeneity Propagation path.
            # Reference: docs/model_cascade_summary.md and Liskowski &
            # Schmaus 2026.
            #
            # Pipeline:
            #   1. Calibrate z_smooth at the labelled rows (in-bag).
            #   2. Fit a logistic GAM (B-spline basis + Laplace approx).
            #   3. Optimise (tau_low, tau_high) by minimising
            #         alpha * normalised_error + (1 - alpha) * delegation
            #      via differential evolution (paper Eq. 31 / 32).
            #   4. Auto-accept rows above tau_high, auto-reject below
            #      tau_low. Optionally verify the uncertain band with a
            #      second LLM round.
            t_defer = time.time()
            n_total_int = int(n_total)
            z_smooth_arr = np.asarray(_final_scores, dtype=np.float64).reshape(-1)
            if z_smooth_arr.shape[0] != n_total_int:
                raise ValueError(
                    f"[defer] z_smooth length ({z_smooth_arr.shape[0]}) != "
                    f"n_total ({n_total_int})"
                )

            y_cal = np.asarray(labeled_values, dtype=np.float64)
            n_pos_cal = int(np.sum(y_cal > 0.5))
            n_neg_cal = int(np.sum(y_cal <= 0.5))
            if str(decision_mode) == "learned_deferral":
                from learned_deferral import fit_deferral_classifier

                z_corrected_full = np.asarray(
                    refine_prof.get("z_corrected", z_smooth_arr), dtype=np.float64
                ).reshape(-1)
                # Paper §3.4 π_i: rank on the sampler's final blended ordering
                # axis σ = (1-λ_T) z(s) + λ_T z(ℓ_T), not raw similarity. Fall
                # back to similarity order when the sampler stayed similarity-
                # only or the axis is unavailable (e.g. probe path).
                _want_blended = str(deferral_rank_axis).lower() != "similarity"
                if (
                    _want_blended
                    and activeSampling_final_axis is not None
                    and activeSampling_final_axis.shape[0] == n_total_int
                ):
                    sorted_idx = np.argsort(-activeSampling_final_axis)
                    profiling["deferral_rank_axis"] = "blended_sampler_axis"
                else:
                    sorted_idx = (
                        np.argsort(-similarities) if similarities is not None else None
                    )
                    profiling["deferral_rank_axis"] = (
                        "similarity_forced" if not _want_blended else "similarity"
                    )

                if ld_split_succeeded and ld_router_indices.shape[0] > 0:
                    # Paper §3.4: train the router on the held-out L_router
                    # partition. Because L_router was excluded from proxy
                    # training and Bias Diffusion + Homogeneity Propagation pinning, its propagated scores z^prop
                    # are already unbiased — pass them directly as the router's
                    # training scores (no OOF needed).
                    router_train_idx = np.asarray(ld_router_indices, dtype=np.int64)
                    router_train_y = np.asarray(ld_router_values, dtype=np.int32)
                    z_router_train = z_smooth_arr[router_train_idx]
                    z_corr_router_train = z_corrected_full[router_train_idx]
                    profiling["deferral_train_source"] = "held_out_router_6_4"
                    profiling["deferral_oof_time"] = 0.0
                    oof_info = dict(profiling.get("router_split_info", {}))
                    oof_info["train_source"] = "held_out_router_6_4"
                else:
                    # Fallback: 5-fold OOF over the full labeled set (the
                    # disjoint split was infeasible for this query/budget).
                    t_oof = time.time()
                    z_smooth_oof, z_corrected_oof, oof_info = (
                        _compute_oof_z_smooth_and_corrected(
                            embeddings=lr_input,
                            W=_prebuilt_W,
                            labeled_indices=labeled_indices,
                            labeled_values=labeled_values,
                            n_folds=int(deferral_k_folds),
                            confidence_threshold=float(confidence_threshold),
                            diffusion_alpha=float(diffusion_alpha),
                            propagation_beta=float(propagation_beta),
                            refine_num_iter=int(refine_num_iter),
                            refine_tol=float(refine_tol),
                            refine_patience=int(refine_patience),
                            imbalance_strategy=_refine_imbalance,
                            lr_C=float(lr_C),
                            base_threshold=base_threshold,
                            random_seed=int(random_seed) if random_seed is not None else 42,
                        )
                    )
                    profiling["deferral_oof_time"] = time.time() - t_oof
                    router_train_idx = np.asarray(labeled_indices, dtype=np.int64)
                    router_train_y = np.asarray(labeled_values, dtype=np.int32)
                    z_router_train = z_smooth_oof
                    z_corr_router_train = z_corrected_oof
                    profiling["deferral_train_source"] = "oof_5fold"

                decision = fit_deferral_classifier(
                    z_smooth_full=z_smooth_arr,
                    z_corrected_full=z_corrected_full,
                    similarity_full=similarities,
                    W_full=_prebuilt_W,
                    sorted_idx_full=sorted_idx,
                    z_smooth_oof=z_router_train,
                    z_corrected_oof=z_corr_router_train,
                    labeled_indices=router_train_idx,
                    labeled_values=router_train_y,
                    threshold_proxy=float(confidence_threshold),
                    feature_names=deferral_features,
                    hidden_dims_grid=deferral_hidden_dims_grid,
                    alpha_grid=deferral_alpha_grid,
                    inner_cv_k=int(deferral_inner_cv_k),
                    min_inner_cv_auc=float(deferral_min_inner_cv_auc),
                    probe_indices=probe_indices if probe_indices.size else None,
                    probe_labels=probe_labels if probe_labels.size else None,
                    min_probe_auc=float(deferral_min_probe_auc),
                    random_state=int(random_seed) if random_seed is not None else 42,
                    classifier_kind="lr",  # paper §3.4: logistic-regression router
                )

                defer_mask = decision.defer_mask.copy()
                proxy_pred = decision.proxy_pred.copy()
                # Pin the FULL actively-sampled set (L_proxy ∪ L_router) to its
                # known expert-model labels — both partitions carry ground truth.
                pin_idx = np.asarray(ld_full_labeled_indices, dtype=np.int64)
                pin_y = np.asarray(ld_full_labeled_values, dtype=np.float64)[: pin_idx.shape[0]]
                defer_mask[pin_idx] = False
                proxy_pred[pin_idx] = pin_y > 0.5
                if probe_indices.size > 0:
                    defer_mask[probe_indices] = False

                final_pos = proxy_pred.copy()
                verify_count = 0
                verify_pos = 0
                verify_time = 0.0
                if verify_uncertain and int(defer_mask.sum()) > 0:
                    cand_idx = np.where(defer_mask)[0]
                    verify_texts = [texts[int(i)] for i in cand_idx.tolist()]
                    t0_verify = time.time()
                    verify_labels = np.asarray(label_fn(verify_texts), dtype=np.int32)
                    verify_time = time.time() - t0_verify
                    for idx_row, lab in zip(cand_idx.tolist(), verify_labels.tolist()):
                        final_pos[int(idx_row)] = int(lab) == 1
                    verify_count = int(cand_idx.size)
                    verify_pos = int((verify_labels == 1).sum())

                defer_total_time = time.time() - t_defer
                deleg_rate = float(defer_mask.mean()) if defer_mask.size else 0.0
                logger.info(
                    f"[learned_deferral] train_wrong={decision.info.get('n_train_wrong')} "
                    f"train_right={decision.info.get('n_train_right')} "
                    f"inner_auc={decision.info.get('inner_cv_auc', float('nan'))} "
                    f"probe_auc={decision.info.get('probe_auc', float('nan'))} "
                    f"defer={int(defer_mask.sum())}/{n_total_int} "
                    f"verify={verify_count} mode={decision.info.get('method')}"
                )

                profiling["defer_enabled"] = True
                profiling["defer_cold_start"] = False
                profiling["defer_calibration_source"] = "oof_z_smooth"
                profiling["defer_n_cal"] = int(len(labeled_indices))
                profiling["defer_n_pos_cal"] = n_pos_cal
                profiling["defer_n_neg_cal"] = n_neg_cal
                profiling["defer_decision_mode"] = "learned_deferral"
                profiling["defer_tau_low"] = float(confidence_threshold)
                profiling["defer_tau_high"] = float(confidence_threshold)
                profiling["defer_delegation_rate"] = deleg_rate
                profiling["defer_uncertain_count"] = int(defer_mask.sum())
                profiling["defer_accept_count"] = int((final_pos & ~defer_mask).sum())
                profiling["defer_reject_count"] = int((~final_pos & ~defer_mask).sum())
                profiling["defer_verify_uncertain"] = bool(verify_uncertain)
                profiling["defer_verify_count"] = verify_count
                profiling["defer_verify_positive"] = verify_pos
                profiling["defer_verify_time"] = verify_time
                profiling["defer_total_time"] = float(defer_total_time)
                profiling["deferral_oof_info"] = oof_info
                profiling["deferral_probe_n"] = int(probe_indices.size)
                profiling["deferral_probe_positive"] = int((probe_labels == 1).sum())
                for key, value in decision.info.items():
                    profiling[f"deferral_{key}"] = value
                positive_mask = final_pos
        if isinstance(refine_prof.get("z_corrected"), np.ndarray):
            refine_prof["z_corrected_shape"] = tuple(
                int(x) for x in refine_prof["z_corrected"].shape
            )
            refine_prof.pop("z_corrected", None)
    elif sampling_method in _refine_methods and not refine_enabled:
        profiling.setdefault("knn_graph_build_time", 0.0)
        t0 = time.time()
        lr_fit_time = 0.0
        lr_predict_time = 0.0
        unique_classes = np.unique(labeled_values)
        if len(unique_classes) < 2:
            positive_mask = np.full(n_total, int(unique_classes[0]) == 1, dtype=bool)
            logger.warning(
                f"{sampling_method} LR-only: single class ({int(unique_classes[0])}) in labeled set"
            )
        else:
            if lr_class_weight_override == "balanced":
                _lr_cw = "balanced"
            elif lr_class_weight_override == "not_balanced":
                _lr_cw = None
            else:
                _lr_cw = None if sampling_method == "random" else "balanced"
            profiling["lr_class_weight_used"] = _lr_cw if _lr_cw is not None else "not_balanced"
            lr = LogisticRegression(
                class_weight=_lr_cw, C=lr_C, max_iter=1000, solver="lbfgs",
            )
            t_fit = time.time()
            lr.fit(lr_input[labeled_indices], labeled_values)
            lr_fit_time = time.time() - t_fit
            # Use sklearn's .predict() for the hard 0/1 decision (argmax /
            # threshold 0.5). No confidence_threshold / base_threshold
            # applied in the LR-only branch — the user gets exactly what
            # sklearn thinks is positive.
            t_pred = time.time()
            positive_mask = lr.predict(lr_input).astype(bool)
            lr_predict_time = time.time() - t_pred
            profiling["lr_only_threshold_source"] = "lr.predict()"
        lr_time = time.time() - t0
        profiling["lr_fit_time"] = lr_fit_time
        profiling["lr_predict_time"] = lr_predict_time
        # lr_training_time kept as old semantic (fit + predict) for back-compat.
        profiling["lr_training_time"] = lr_time
        profiling["propagation_time"] = 0.0
        profiling["prediction_time"] = lr_predict_time
        # Expose lr_fit / lr_predict in refinement_profiling so the grid
        # runner can read the same keys whether Bias Diffusion + Homogeneity Propagation is enabled or not.
        profiling["refinement_profiling"] = {
            "lr_fit_time": lr_fit_time,
            "lr_predict_time": lr_predict_time,
            "lr_training_time": lr_time,
            "graph_norm_time": 0.0,
            "bias_diffusion_time": 0.0,
            "homogeneity_propagation_time": 0.0,
        }
        logger.info(
            f"{sampling_method} LR-only: predicted {int(np.sum(positive_mask))}/{n_total} positive "
            f"(from lr.predict())"
        )
    else:
        profiling.setdefault("knn_graph_build_time", 0.0)
        t0 = time.time()
        class_prior = n_pos_labeled / max(len(labeled_values), 1)
        pseudo_labels, confidence_scores = label_propagation_cutssl(
            embeddings, labeled_indices, labeled_values,
            class_prior=class_prior, knn_k=knn_k,
        )
        profiling["propagation_time"] = time.time() - t0

        t0 = time.time()
        if cluster_proxy_mode == "weighted_lr":
            positive_mask = _predict_with_weighted_lr(
                lr_input, pseudo_labels, confidence_scores,
                labeled_indices, labeled_values,
                final_cluster_labels, cluster_stats, confidence_threshold,
            )
        else:
            positive_mask = _predict_with_propagation(
                pseudo_labels, confidence_scores,
                labeled_indices, labeled_values, confidence_threshold,
            )
        profiling["prediction_time"] = time.time() - t0
        profiling["lr_training_time"] = (
            profiling["prediction_time"] if cluster_proxy_mode == "weighted_lr" else 0.0
        )
    # Report the actual prediction mode that was used.
    if sampling_method in _refine_methods and not refine_enabled:
        actual_mode = "lr_only"
    elif sampling_method in _refine_methods and refine_enabled:
        actual_mode = "score_refinement"
    else:
        actual_mode = cluster_proxy_mode
    profiling["mode"] = actual_mode
    profiling["vllm_invocation_time"] = vllm_time_acc[0]
    profiling["n_total"] = int(n_total)

    n_predicted_pos = int(np.sum(positive_mask))
    profiling["n_predicted_positive"] = n_predicted_pos
    logger.info(
        f"Cluster proxy done: mode={actual_mode}, "
        f"labeled={len(labeled_indices)}, predicted_pos={n_predicted_pos}/{n_total}"
    )

    return positive_mask, profiling
