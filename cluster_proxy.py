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
from router_cap import cap_router_candidates

LabelFnType = Callable[[List[Union[str, list]]], np.ndarray]
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg as conjugate_gradient
from scipy.stats import beta as beta_dist
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from bargain_refined import bargain_mode as _bargain_mode, run_bargain_on_refined as _run_bargain_on_refined


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
_KNN_CACHE_VERSION = "unit-gaussian-union-v2|" + os.environ.get("LLMSQL_KNN_CACHE_VERSION", "v1")
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
    h.update(np.ascontiguousarray(embeddings).tobytes())
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

    # Section 4 constructs the graph on unit embeddings, including callers
    # that provide raw features instead of the public PCA representation.
    from sklearn.preprocessing import normalize
    embeddings = normalize(np.asarray(embeddings, dtype=np.float64), norm="l2")

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

    # Remove each row's own ID explicitly: with duplicate embeddings it need
    # not be the first zero-distance neighbor returned by either backend.
    neighbors = indices != np.arange(n)[:, None]
    positions = np.argsort(~neighbors, axis=1, kind="stable")[:, :k]
    nbr_idx = np.take_along_axis(indices, positions, axis=1)
    nbr_dist = np.take_along_axis(distances, positions, axis=1)
    if sigma <= 0.0:
        med = float(np.median(nbr_dist[:, -1]))
        sigma_used = med if med > 1e-10 else 1.0
    else:
        sigma_used = float(sigma)
    sub_timings["sigma_used"] = sigma_used
    sigma_sq_2 = 2.0 * sigma_used * sigma_used

    # --- CSR build (vectorized; replaces ~N*k Python iterations) --------
    t0 = time.time()
    row_ind = np.repeat(np.arange(n, dtype=np.int64), k)
    col_ind = np.ascontiguousarray(nbr_idx, dtype=np.int64).ravel()
    d2 = np.square(np.ascontiguousarray(nbr_dist, dtype=np.float64).ravel())
    weights = np.exp(-d2 / sigma_sq_2)
    W = csr_matrix((weights, (row_ind, col_ind)), shape=(n, n))
    sub_timings["csr_build_time"] = time.time() - t0

    # --- Symmetrize -----------------------------------------------------
    t0 = time.time()
    # Every edge in the union receives the same Gaussian weight, whether
    # selected from one endpoint or both (Section 4, Graph Construction).
    W = W.maximum(W.T)
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
    num_iter_bias: Optional[int] = None,
    num_iter_prop: Optional[int] = None,
    tol: float = 0.0,
    patience: int = 5,
    imbalance_strategy: str = "balanced",
    lr_C: float = 1.0,
    base_threshold: Optional[float] = None,
    return_logit: bool = False,
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
    # T1, T2 in the paper: the caps of the two phases. They share `num_iter` unless the caller
    # separates them (the appendix sensitivity study does). Both phases still exit early on
    # convergence, so the cap is an upper bound, not the number of iterations that will run.
    t1 = int(num_iter if num_iter_bias is None else num_iter_bias)
    t2 = int(num_iter if num_iter_prop is None else num_iter_prop)
    prof["bias_diffusion_cap"] = t1
    prof["homogeneity_propagation_cap"] = t2

    # --- Stage A: LR base prediction ---
    t0 = time.time()
    unique_classes = np.unique(labeled_values)
    if len(unique_classes) < 2:
        single_class = int(unique_classes[0]) if len(unique_classes) else None
        logger.warning(
            f"Bias Diffusion + Homogeneity Propagation: only one class ({single_class}) in labeled set — "
            f"returning constant prediction"
        )
        base_probas = np.full(n, .5 if single_class is None else float(single_class), dtype=np.float64)
        prof["no_training_labels"] = single_class is None
        prof["lr_fit_time"] = 0.0
        prof["lr_predict_time"] = time.time() - t0
        prof["lr_training_time"] = prof["lr_fit_time"] + prof["lr_predict_time"]
        prof["base_single_class"] = True
        if return_logit:
            prof["lr_logit"] = None
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
        if return_logit:
            # Raw LR logit for learned calibration (design3 §3); not clipped.
            prof["lr_logit"] = np.asarray(lr.decision_function(embeddings), dtype=np.float64)
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

    # --- Graph normalization: row-stochastic P = D^(-1) W (paper Alg. 1,
    # Phase 0: "row-normalized propagation matrix P"). Row-stochastic P makes
    # each diffusion step (1-a)x0 + a*P@x a convex combination of values in
    # [0,1], so both phases stay in [0,1] WITHOUT a per-iteration clip -- which
    # is exactly why Phase 2 carries no clip. The earlier symmetric normalization
    # D^(-1/2) W D^(-1/2) has max row-sum > 1 (not a convex combination) and,
    # combined with the unclipped Phase 2, let the refined score overshoot
    # outside [0,1], violating the paper's f^(T2) in [0,1]^N contract.
    t0 = time.time()
    deg = np.array(W.sum(axis=1)).flatten()
    d_inv = np.divide(1., deg, out=np.zeros_like(deg), where=deg > 0)
    D_inv = diags(d_inv, format="csr")
    S = D_inv @ W
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
        alpha=alpha, num_iter=t1, tol=tol, patience=patience,
    )
    # FDiff-scale (Huang et al. 2021, Sec. 2.2): write the corrected prediction
    # on unlabelled rows and pin L to its expert labels.
    z_corrected = base_probas.copy().astype(np.float64)
    if len(pinned_idx) > 0:
        unlabeled_mask = np.ones(n, dtype=bool)
        unlabeled_mask[pinned_idx] = False
        z_corrected[unlabeled_mask] = np.clip(
            base_probas[unlabeled_mask] + E_final[unlabeled_mask], 0.0, 1.0,
        )
    else:
        z_corrected = np.clip(base_probas + E_final, 0.0, 1.0)
    z_corrected[pinned_idx] = np.asarray(labeled_values,dtype=float)
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
        alpha=beta, num_iter=t2, tol=tol, patience=patience,
        clip01=False,  # paper Alg. 3 Phase 2: f^(t) = (1-beta) f^(0) + beta S f^(t-1), no per-iter clip
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


def auto_kmeans_k(n: int) -> int:
    """Heuristic KMeans K: round(sqrt(N/2)), capped [5, 200]."""
    return min(max(5, round(math.sqrt(n / 2))), 200)


def auto_knn_k(n: int) -> int:
    """Active manuscript: round(3 log2 N), bounded only by available neighbors.

    Historical capped appendix behavior requires PROBE_KNN_PROTOCOL=legacy.
    """
    if n <= 1:
        return 1
    k = round(math.log2(n) * 3)
    if os.environ.get("PROBE_KNN_PROTOCOL", "paper") == "legacy":
        return max(5, min(k, 50))
    return min(n - 1, max(1, k))


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


def _run_learned_calibration_stage(
    lr_input: np.ndarray,
    W: csr_matrix,
    texts: List[Union[str, list]],
    label_fn: Callable[[List[Union[str, list]]], np.ndarray],
    labeled_indices: np.ndarray,
    labeled_values: np.ndarray,
    positive_mask: np.ndarray,
    lr_logit: Optional[np.ndarray],
    *,
    config: Any,
    refined_score: Optional[np.ndarray] = None,
    confidence_threshold: float,
    diffusion_alpha: float,
    propagation_beta: float,
    refine_num_iter: int,
    refine_t1: Optional[int],
    refine_t2: Optional[int],
    refine_tol: float,
    refine_patience: int,
    imbalance_strategy: str,
    lr_C: float,
    base_threshold: Optional[float],
    profiling: Dict[str, Any],
) -> np.ndarray:
    """Learned calibration router (design3.md). Returns the final positive mask.

    Writes ``lc_*`` diagnostics plus the ``defer_*`` keys run.py already reads,
    so ``policy_rollback_to_llm_count`` counts the A and D calls.
    """
    from learned_calibration import OofFeatures, compute_oof_features, refined_logit, run_learned_calibration

    # calib_input="refined": the calibrator reads the logit of the graph-refined score z_smooth, and the
    # out-of-fold features of L use each fold's own refinement; the design of A stays on the LR logit.
    use_refined = getattr(config, "calib_input", "proxy") == "refined"
    if use_refined and refined_score is None:
        raise ValueError("calib_input='refined' needs the graph-refined score")
    x_cal = refined_logit(refined_score) if use_refined and lr_logit is not None else None

    t0 = time.time()

    def refine_fn(train_idx: np.ndarray, train_y: np.ndarray):
        mask, z_fold, prof = predict_with_score_refinement(
            lr_input, W, train_idx, train_y,
            confidence_threshold=confidence_threshold,
            alpha=diffusion_alpha, beta=propagation_beta,
            num_iter=refine_num_iter, num_iter_bias=refine_t1, num_iter_prop=refine_t2,
            tol=refine_tol, patience=refine_patience,
            imbalance_strategy=imbalance_strategy, lr_C=lr_C,
            base_threshold=base_threshold, return_logit=True,
        )
        if use_refined:
            return mask, (None if prof.get("lr_logit") is None else refined_logit(z_fold))
        return mask, prof.get("lr_logit")

    def label_rows(row_ids: np.ndarray) -> np.ndarray:
        if hasattr(label_fn, "label_ids"):
            return label_fn.label_ids(row_ids, stage="A/D")
        return np.asarray(label_fn([texts[int(i)] for i in row_ids]), dtype=np.int32)

    if hasattr(label_fn, "label_ids"):
        label_rows.for_stage = lambda ids, stage: label_fn.label_ids(ids, stage=stage)

    if lr_logit is None:
        oof = OofFeatures(None, None, "proxy_logit_unavailable")
    else:
        oof = compute_oof_features(
            refine_fn, labeled_indices, labeled_values,
            n_folds=config.n_folds, seed=config.seed,
        )
    from oracle_budget import OracleLabelsUnavailable
    try:
        result = run_learned_calibration(
            x=lr_logit,
            h=np.asarray(positive_mask, dtype=np.float64),
            labeled_indices=labeled_indices,
            labeled_values=labeled_values,
            oof=oof,
            label_rows=label_rows,
            config=config,
            x_cal=x_cal,
        )
    except OracleLabelsUnavailable:
        session = getattr(label_fn, "oracle_session", None)
        if session is None:
            raise
        final = np.zeros(len(texts), dtype=bool)
        for key, value in session.ledger.cache.items():
            final[int(key)] = bool(value)
        profiling.update(lc_status="abstain", lc_reason="oracle_labels_unavailable",
                         lc_calls_total=session.ledger.used)
        return final
    router_calls = int(result.calib_idx.size + result.verify_idx.size)
    elapsed = time.time() - t0
    profiling.update(result.diagnostics)
    from experiments.evaluation_export import remember_calibration
    remember_calibration(result, x_cal if use_refined else lr_logit,
                         np.asarray(positive_mask,dtype=float), labeled_indices)
    audit_dir = os.environ.get("LASO_LC_AUDIT_DIR", "").strip()
    session = getattr(label_fn, "oracle_session", None)
    if audit_dir and lr_logit is not None and session is not None:
        # Evaluation only (calibration_audit.py): labels an extra sample on a separate ledger after routing;
        # decisions and call counts are unchanged, but the run's wall clock includes the audit.
        from calibration_audit import run_audit
        profiling.update(run_audit(
            result=result, x=lr_logit, h=np.asarray(positive_mask, dtype=np.float64),
            labeled_indices=labeled_indices, labeled_values=labeled_values, session=session,
            config=config, out_dir=audit_dir, n=int(os.environ.get("LASO_LC_AUDIT_N", "3000")), x_cal=x_cal))
    profiling["lc_time_s"] = float(elapsed)
    profiling["defer_enabled"] = True
    profiling["defer_decision_mode"] = "learned_calibration"
    profiling["defer_verify_uncertain"] = True
    profiling["defer_verify_count"] = router_calls
    profiling["defer_verify_positive"] = int(result.final_pos[result.calib_idx].sum()
                                             + result.final_pos[result.verify_idx].sum())
    profiling["defer_candidates_before_cap"] = router_calls
    profiling["defer_router_cap"] = None if config.budget_scope == "preparation" else max(0, int(config.total_budget) - int(len(labeled_indices)))
    profiling["budget_sample_size"] = int(config.sample_budget)
    profiling["defer_total_time"] = float(elapsed)
    logger.info(
        f"[learned_calibration] status={result.status} reason={result.diagnostics.get('lc_reason')} "
        f"A={result.calib_idx.size} D={result.verify_idx.size} "
        f"calls={result.diagnostics.get('lc_calls_total')}/{config.total_budget}"
    )
    return result.final_pos


def _whole_query_llm_fallback(texts, label_fn, raw_label_fn, labeled_indices, labeled_values,
                              profiling, vllm_time_acc):
    """Label every candidate outside L with the LLM and return the LLM answer (paper §3.2)."""
    n_total = len(texts)
    labeled_indices = np.asarray(labeled_indices, dtype=np.int64)
    rest = np.setdiff1d(np.arange(n_total), labeled_indices)
    t0 = time.time()
    if rest.size and hasattr(label_fn, "label_ids"):
        values = np.asarray(label_fn.label_ids(rest.tolist(), stage="fallback"), dtype=np.int32)
    elif rest.size:
        values = np.asarray(label_fn([texts[int(i)] for i in rest]), dtype=np.int32)
    else:
        values = np.zeros(0, dtype=np.int32)
    positive_mask = np.zeros(n_total, dtype=bool)
    positive_mask[labeled_indices[np.asarray(labeled_values) == 1]] = True
    positive_mask[rest[values == 1]] = True
    unresolved = rest[(values != 0) & (values != 1)]
    profiling.update(
        whole_query_fallback=True, whole_query_fallback_calls=int(rest.size),
        whole_query_fallback_unresolved=int(unresolved.size),
        whole_query_fallback_time=time.time() - t0, vllm_time=float(vllm_time_acc[0]),
        lc_status="whole_query_llm", n_total=int(n_total),
        result_status="llm_executed" if not unresolved.size else "llm_executed_with_failures",
        discovery_status="positive_found" if positive_mask.any() else
                         "no_positive_found" if unresolved.size else "verified_empty",
        quality_certificate=None,
    )
    if hasattr(raw_label_fn, "ledger"):
        profiling["oracle_budget"] = raw_label_fn.ledger.snapshot()
        profiling["unresolved_ids"] = unresolved.tolist()
    profiling["n_predicted_positive"] = int(positive_mask.sum())
    logger.info(f"Whole-query LLM fallback: labelled {rest.size} rows, "
                f"predicted_pos={int(positive_mask.sum())}/{n_total}")
    return positive_mask, profiling


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
    refine_t1: Optional[int] = None,
    refine_t2: Optional[int] = None,
    refine_tol: float = 0.0,
    refine_patience: int = 5,
    refine_enabled: bool = True,
    # uqe_active (UQE Algorithm 2, arXiv:2407.09522 §4.1.2 / §C.2)
    uqe_batch_size: int = 16,
    uqe_exploration_sigma_0: float = 0.0,
    uqe_exploration_decay: float = 0.9,
    # defer dual-threshold (replaces single-tau on the active-learning sampling + Bias Diffusion + Homogeneity Propagation path)
    defer_enabled: bool = False,
    verify_uncertain: bool = True,
    router_max_calls: Optional[int] = None,
    decision_mode: str = "force_label",
    learned_calibration_config: Optional[Any] = None,
    # Router π_i rank axis (paper §3.4): "blended" = rank on the sampler's
    # final blended ordering axis σ (default); "similarity" = force rank on raw
    # cosine similarity regardless of the sampler axis.
    lr_C: float = 1.0,
    base_threshold: Optional[float] = None,
    lr_class_weight_override: Optional[str] = None,
    random_seed: Optional[int] = 0,
    viz_tag: str = "",
    activeSampling_backend: str = "fixed_ce",
    fixed_ce_config: Optional[Any] = None,
    fixed_ce_scores: Optional[np.ndarray] = None,
    paper_query_embedding: Optional[np.ndarray] = None,
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
    if os.environ.get("PROBE_PAPER_JOB"):
        from experiments.paper_pipeline import execute
        return execute(embeddings, texts, label_fn_factory, sample_size,
                       fixed_ce_scores if fixed_ce_scores is not None else similarities,
                       fixed_ce_config, learned_calibration_config,
                       query_embedding=paper_query_embedding)

    if decision_mode in ('learned_deferral','dasp_topk'):
        raise ValueError('Retired router; use learned_calibration')
    profiling: Dict[str, Any] = {}
    vllm_time_acc = [0.0]
    n_total = embeddings.shape[0]

    if str(decision_mode) == "learned_calibration":
        if learned_calibration_config is None:
            raise ValueError("decision_mode='learned_calibration' requires learned_calibration_config")
        if sampling_method != "activeSampling" or not refine_enabled or not defer_enabled:
            raise ValueError(
                "decision_mode='learned_calibration' requires sampling_method='activeSampling', "
                "refine_enabled=True and defer_enabled=True"
            )
        if int(sample_size) > int(learned_calibration_config.sample_budget):
            raise ValueError(
                f"sample_size={int(sample_size)} exceeds learned_calibration_config.sample_budget="
                f"{int(learned_calibration_config.sample_budget)}"
            )

    if activeSampling_backend == "fixed_ce" and learned_calibration_config is not None:
        from dataclasses import replace
        learned_calibration_config = replace(learned_calibration_config, fixed_calibration=True)

    raw_label_fn = label_fn_factory()

    def label_fn(texts: List[Union[str, list]]) -> np.ndarray:
        t0 = time.time()
        result = raw_label_fn(texts)
        vllm_time_acc[0] += time.time() - t0
        return result

    if hasattr(raw_label_fn, "label_ids"):
        def label_ids(ids, *, stage="L"):
            t0 = time.time()
            try:
                return raw_label_fn.label_ids(ids, stage=stage)
            finally:
                vllm_time_acc[0] += time.time() - t0
        label_fn.label_ids = label_ids
        label_fn.oracle_session = raw_label_fn

    profiling["sampling_method"] = sampling_method
    if random_seed is not None:
        np.random.seed(int(random_seed))
        import random as _py_random
        _py_random.seed(int(random_seed))
        profiling["random_seed"] = int(random_seed)
        logger.info(f"Pipeline random_seed={int(random_seed)} (global numpy + stdlib random seeded)")
    activeSampling_W: Optional[csr_matrix] = None
    random_W: Optional[csr_matrix] = None
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
        if activeSampling_backend != 'fixed_ce':
            raise ValueError('Only the fixed_ce sampling backend is supported')
        from fixed_ce_sampling import SamplingConfig, sample_fixed_ce_texts
        if fixed_ce_scores is None or np.asarray(fixed_ce_scores).shape != (n_total,):
            raise ValueError('fixed_ce backend requires aligned explicit fixed_ce_scores')
        similarities = np.asarray(fixed_ce_scores, dtype=float)
        cfg = fixed_ce_config or SamplingConfig(seed=int(random_seed or 0))
        t0 = time.time()
        labeled_indices, labeled_values, activeSampling_W, dir_prof = sample_fixed_ce_texts(
            similarities, texts, label_fn, sample_size, config=cfg, original_ids=list(range(n_total)))
        total_time = time.time() - t0
        profiling["stratum_time"] = float(dir_prof.get("graph_time", 0.0))
        profiling["sampling_time"] = max(0.0, total_time - profiling["stratum_time"])
        profiling["n_labeled"] = int(len(labeled_indices))
        n_pos_labeled = int(np.sum(labeled_values == 1))
        profiling["n_pos_labeled"] = n_pos_labeled
        profiling["activeSampling_profiling"] = dir_prof
        # Paper Alg. 1, lines 12-13: without both classes in L no proxy can be trained, so the whole
        # query goes to the LLM and every later stage (proxy, refinement, calibration, routing) is
        # skipped. Stage 'fallback' is outside the preparation allowance.
        if (activeSampling_backend == "fixed_ce" and getattr(cfg, "protocol", "legacy") == "paper"
                and (n_pos_labeled == 0 or n_pos_labeled == len(labeled_values))):
            return _whole_query_llm_fallback(
                texts, label_fn, raw_label_fn, labeled_indices, labeled_values, profiling, vllm_time_acc)
        cluster_stats = {}
        final_cluster_labels = np.zeros(n_total, dtype=np.int32)

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
    # With refinement off every sampler reaches the same LR-only predictor, so a sampler ablation
    # varies only the sampler; stratified+refine keeps the classic LP pipeline.
    _lr_only_methods = _refine_methods + ("stratified",)
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
            num_iter=refine_num_iter, num_iter_bias=refine_t1, num_iter_prop=refine_t2,
            tol=refine_tol, patience=refine_patience,
            imbalance_strategy=_refine_imbalance,
            lr_C=lr_C,
            base_threshold=base_threshold,
            return_logit=(str(decision_mode) == "learned_calibration"),
        )
        _dump_dir = os.environ.get("PROBE_DUMP_REFINED_DIR")
        if _dump_dir:
            # Offline theta_ref study: refined scores f, the anchors, and the threshold used.
            np.savez_compressed(os.path.join(_dump_dir, "refined_scores.npz"),
                                scores=np.asarray(_final_scores, dtype=np.float32),
                                labeled_indices=np.asarray(labeled_indices, dtype=np.int64),
                                labeled_values=np.asarray(labeled_values, dtype=np.int8),
                                threshold=float(confidence_threshold))
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
            n_total_int = int(n_total)
            z_smooth_arr = np.asarray(_final_scores, dtype=np.float64).reshape(-1)
            if z_smooth_arr.shape[0] != n_total_int:
                raise ValueError(
                    f"[defer] z_smooth length ({z_smooth_arr.shape[0]}) != "
                    f"n_total ({n_total_int})"
                )

            if str(decision_mode) == "learned_calibration" and _bargain_mode() is not None:
                # Reviewer control (Fig 10): BARGAIN on the refined score replaces calibration + routing.
                positive_mask = _run_bargain_on_refined(
                    _final_scores, label_fn, labeled_indices, labeled_values,
                    seed=int(random_seed or 0), profiling=profiling)
            elif str(decision_mode) == "learned_calibration":
                positive_mask = _run_learned_calibration_stage(
                    lr_input, _prebuilt_W, texts, label_fn,
                    labeled_indices, labeled_values, positive_mask,
                    refine_prof.get("lr_logit"),
                    config=learned_calibration_config,
                    refined_score=_final_scores,
                    confidence_threshold=confidence_threshold,
                    diffusion_alpha=diffusion_alpha,
                    propagation_beta=propagation_beta,
                    refine_num_iter=refine_num_iter,
                    refine_t1=refine_t1,
                    refine_t2=refine_t2,
                    refine_tol=refine_tol,
                    refine_patience=refine_patience,
                    imbalance_strategy=_refine_imbalance,
                    lr_C=lr_C,
                    base_threshold=base_threshold,
                    profiling=profiling,
                )
        refine_prof.pop("lr_logit", None)
        if isinstance(refine_prof.get("z_corrected"), np.ndarray):
            refine_prof["z_corrected_shape"] = tuple(
                int(x) for x in refine_prof["z_corrected"].shape
            )
            refine_prof.pop("z_corrected", None)
    elif sampling_method in _lr_only_methods and not refine_enabled:
        profiling.setdefault("knn_graph_build_time", 0.0)
        t0 = time.time()
        lr_fit_time = 0.0
        lr_predict_time = 0.0
        unique_classes = np.unique(labeled_values)
        if len(unique_classes) < 2:
            single_class = int(unique_classes[0]) if len(unique_classes) else None
            positive_mask = np.full(n_total, single_class == 1, dtype=bool)
            logger.warning(
                f"{sampling_method} LR-only: observed class ({single_class}) in labeled set"
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
    if sampling_method in _lr_only_methods and not refine_enabled:
        actual_mode = "lr_only"
    elif sampling_method in _refine_methods and refine_enabled:
        actual_mode = "score_refinement"
    else:
        actual_mode = cluster_proxy_mode
    profiling["mode"] = actual_mode
    profiling["vllm_invocation_time"] = vllm_time_acc[0]
    profiling["n_total"] = int(n_total)

    if hasattr(raw_label_fn, "ledger"):
        ledger = raw_label_fn.ledger
        known = ledger.cache
        unresolved = [i for i in range(n_total) if i not in known]
        positives = [i for i, y in known.items() if y == 1]
        profiling["oracle_budget"] = ledger.snapshot()
        profiling["verified_positive_ids"] = positives
        profiling["unresolved_ids"] = unresolved
        profiling["quality_certificate"] = None
        if unresolved and (not positives or ledger.failures or profiling.get("lc_status") in ("abstain", "insufficient_evidence")):
            positive_mask = np.zeros(n_total, dtype=bool)
            positive_mask[positives] = True
            profiling["result_status"] = "abstain"
        else:
            profiling["result_status"] = "fully_verified" if not unresolved else "approximate_uncertified"
        profiling["discovery_status"] = "positive_found" if positives else "no_positive_found" if unresolved else "verified_empty"
        profiling["stop_reason"] = "budget_exhausted" if ledger.used >= ledger.limit else "sampling_complete"

    n_predicted_pos = int(np.sum(positive_mask))
    profiling["n_predicted_positive"] = n_predicted_pos
    logger.info(
        f"Cluster proxy done: mode={actual_mode}, "
        f"labeled={len(labeled_indices)}, predicted_pos={n_predicted_pos}/{n_total}"
    )

    return positive_mask, profiling
