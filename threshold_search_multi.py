"""optimized_multiple: threshold search when the matches form several clusters on the ordering axis.

Code convention (as threshold_search): row 0 is the highest-scoring row (the match end); ``w`` holds
+1 / -1 for labelled rows and 0 otherwise. A hypothesis is a union of row intervals predicted to match.
Its loss is #labelled matches outside + #labelled non-matches inside, plus ``lam`` per interval.

Everything here is a pure function of the labels, recomputed from scratch after every batch, so the
sampling windows can move, appear, merge and vanish freely: the whole-axis search that gives
optimized_single its backtracking carries over unchanged. See
docs/experiments/2026-09-21-optimized-multiple-plan.md.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from stratified_calibration import allocate_cumulative
from sampling_strata import geometric_strata

__all__ = [
    "penalised_intervals",
    "seed_rows",
    "boundary_plateaus",
    "build_windows",
    "allocate_batch",
    "stage2_order",
]

Interval = Tuple[int, int]      # rows, inclusive


def penalised_intervals(w: np.ndarray, lam: float = 1.5, k_max: int = 6,
                        pos_weight: float = 1.0) -> List[Interval]:
    """argmin over unions of AT MOST ``k_max`` intervals of  loss + lam * #intervals  (exact).

    Equivalently maximise sum(w inside) - lam * #intervals; DP over the labelled rows with the number
    of intervals as a state, O(n_L * k_max). Ties are broken lexicographically: fewer intervals,
    then fewer labelled rows inside (the evidence is neutral, so predict less and let the excluded
    matches act as seeds). Every returned interval starts and ends on a labelled match.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    pos = np.flatnonzero(w != 0.0)
    k_max = int(k_max)
    if pos.size == 0 or k_max <= 0:
        return []
    eps_k, eps_len = 1e-3, 1e-9                 # lexicographic tie-breaks, far below the 0.5 score grid
    v = np.where(w[pos] > 0, float(pos_weight), -1.0) - eps_len
    lam_eff = float(lam) + eps_k
    m, K = pos.size, k_max
    neg_inf = -1e18
    out_v = np.full(K + 1, neg_inf)
    in_v = np.full(K + 1, neg_inf)
    out_v[0] = 0.0
    entered = np.zeros((m, K + 1), dtype=bool)  # in-state (i, j) was entered from out-state (i-1, j-1)
    left_in = np.zeros((m, K + 1), dtype=bool)  # out-state (i, j) came from in-state (i-1, j)
    for i in range(m):
        start = np.full(K + 1, neg_inf)
        start[1:] = out_v[:-1] - lam_eff
        enter = start > in_v
        new_in = np.where(enter, start, in_v) + v[i]
        leave = in_v > out_v
        new_out = np.where(leave, in_v, out_v)
        entered[i], left_in[i] = enter, leave
        out_v, in_v = new_out, new_in
    best_out, best_in = int(np.argmax(out_v)), int(np.argmax(in_v))
    state_in = bool(in_v[best_in] > out_v[best_out])
    j = best_in if state_in else best_out
    intervals: List[Interval] = []
    right: Optional[int] = None
    for i in range(m - 1, -1, -1):
        if state_in:
            if right is None:
                right = i
            if entered[i, j]:
                intervals.append((int(pos[i]), int(pos[right])))
                right, state_in, j = None, False, j - 1
        else:
            state_in = bool(left_in[i, j])
    intervals.reverse()
    return intervals


def seed_rows(w: np.ndarray, intervals: Sequence[Interval]) -> List[int]:
    """Labelled matches outside every interval: candidate clusters of one row."""
    inside = np.zeros(len(w), dtype=bool)
    for a, b in intervals:
        inside[a : b + 1] = True
    return [int(q) for q in np.flatnonzero((np.asarray(w) > 0) & ~inside)]


def boundary_plateaus(w: np.ndarray, intervals: Sequence[Interval], seeds: Sequence[int]) -> List[Interval]:
    """Runs of unlabelled rows next to a boundary: the labels cannot place the boundary inside them.

    One on each side of every interval and seed, plus the run from the match end to the first
    labelled row (the virtual boundary that reproduces optimized_single when no match is labelled).
    """
    return sorted({(lo, hi) for lo, hi, _ in _tagged_plateaus(w, intervals, seeds)})


def _tagged_plateaus(w: np.ndarray, intervals: Sequence[Interval], seeds: Sequence[int]):
    w = np.asarray(w)
    n = w.size
    labelled = np.flatnonzero(w != 0.0)
    if labelled.size == 0:
        return [(0, n - 1, "interval")] if n else []
    out = []
    if labelled[0] > 0:
        out.append((0, int(labelled[0]) - 1, "interval"))
    for kind, items in (("interval", list(intervals)), ("seed", [(q, q) for q in seeds])):
        for a, b in items:
            ia = int(np.searchsorted(labelled, a))
            ib = int(np.searchsorted(labelled, b))
            prev_row = int(labelled[ia - 1]) if ia > 0 else -1
            next_row = int(labelled[ib + 1]) if ib + 1 < labelled.size else n
            if a - prev_row > 1:
                out.append((prev_row + 1, a - 1, kind))
            if next_row - b > 1:
                out.append((b + 1, next_row - 1, kind))
    return out


def build_windows(w: np.ndarray, intervals: Sequence[Interval], seeds: Sequence[int],
                  prev_windows: Sequence[Interval], r: int, c: float) -> Tuple[List[Interval], List[float]]:
    """Sampling windows for the next batch and their budget weights.

    * one window per boundary plateau, centred on it, width max(|plateau|, ceil(w_prev / c)), with
      w_prev the width of the previous window that contained its centre, or the scheduled
      n / c^(r-1) if none did (r = index of the batch about to be drawn);
    * the interior of every interval: two hit clusters with few labelled non-matches between them
      are merged by the penalised minimiser, and only labels inside can split them again;
    * overlapping windows merge. A merged window that stems from an interval (or the match end)
      has weight 1; all windows that stem from seeds only share ONE unit of weight, so isolated
      matches are always followed up but never crowd out the intervals.
    """
    n = int(np.asarray(w).size)
    sched = max(1.0, n / (c ** max(0, r - 1)))
    raw = []
    for lo, hi, kind in _tagged_plateaus(w, intervals, seeds):
        centre2 = lo + hi
        w_prev = sched
        for a, b in prev_windows:
            if 2 * a <= centre2 <= 2 * b:
                w_prev = float(b - a + 1)
                break
        width = int(min(n, max(hi - lo + 1, math.ceil(w_prev / c - 1e-9), 1)))
        i = (centre2 + 1 - width) // 2
        i = max(0, min(i, n - width))
        raw.append((i, i + width - 1, kind == "interval"))
    raw.extend((a, b, True) for a, b in intervals)
    raw.sort()
    merged: List[List] = []
    for a, b, strong in raw:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
            merged[-1][2] = merged[-1][2] or strong
        else:
            merged.append([a, b, strong])
    n_weak = sum(1 for m_ in merged if not m_[2])
    windows = [(int(a), int(b)) for a, b, _ in merged]
    weights = [1.0 if strong else 1.0 / n_weak for _, _, strong in merged]
    return windows, weights


def allocate_batch(windows: Sequence[Interval], labelled_mask: np.ndarray, batch: int,
                   explore_frac: float, weights: Optional[Sequence[float]] = None,
                   max_strata: int = 10, rng=None, uniform_remainder=False) -> Tuple[np.ndarray, int]:
    """Rows to label next: weighted shares over the windows, plus exploration outside them.

    Returns (rows, n_explore). Exploration draws equally from geometric strata (from the match end)
    of the unlabelled rows outside all windows. Whatever one side cannot absorb goes to the other,
    so the batch is spent in full whenever enough unlabelled rows exist. Uses np.random.
    """
    rng = np.random if rng is None else rng
    n = labelled_mask.size
    in_window = np.zeros(n, dtype=bool)
    for a, b in windows:
        in_window[a : b + 1] = True
    free = ~labelled_mask
    outside = np.flatnonzero(free & ~in_window)
    cands = [np.flatnonzero(free[a : b + 1]) + a for a, b in windows]
    wts = np.asarray(weights if weights is not None else [1.0] * len(cands), dtype=np.float64)
    room = int(sum(c.size for c in cands))
    n_explore = int(min(outside.size, round(explore_frac * batch)))
    n_exploit = int(min(room, batch - n_explore))
    n_explore = int(min(outside.size, batch - n_exploit))        # what the windows cannot hold

    share = np.zeros(len(cands), dtype=np.int64)
    left = n_exploit
    while left > 0:                                   # weighted shares, capped by what a window holds
        active = [i for i in range(len(cands)) if share[i] < cands[i].size]
        if not active:
            break
        tot = float(wts[active].sum())
        gave = 0
        for i in sorted(active, key=lambda t: -wts[t]):
            give = int(min(max(1, math.floor(left * wts[i] / tot)), cands[i].size - share[i], left - gave))
            share[i] += give
            gave += give
            if gave >= left:
                break
        left -= gave
    picks: List[np.ndarray] = []
    for i, k in enumerate(share.tolist()):
        if k > 0:
            picks.append(rng.choice(cands[i], size=k, replace=False))
    if n_explore > 0 and uniform_remainder:
        picks.append(rng.choice(outside, size=n_explore, replace=False))
    elif n_explore > 0:
        strata = geometric_strata(outside, max_strata=max(1, min(max_strata, n_explore)))
        sizes = np.array([s_.size for s_ in strata], dtype=np.int64)
        counts = allocate_cumulative(sizes, np.zeros_like(sizes), n_explore)
        for s_, k in zip(strata, counts.tolist()):
            if k > 0:
                picks.append(rng.choice(s_, size=k, replace=False))
    rows = np.concatenate(picks).astype(np.int64) if picks else np.zeros(0, dtype=np.int64)
    return rows, n_explore


def stage2_order(w: np.ndarray, intervals: Sequence[Interval], seeds: Sequence[int], q: int) -> List[int]:
    """Up to q unlabelled rows for Stage 2: from every boundary into its interval first (round-robin),
    then outwards from every boundary and seed. With no labelled match: the scan from the match end."""
    w = np.asarray(w)
    n = w.size
    taken: List[int] = []
    seen = set()

    def walk(walkers: List[List[int]]) -> None:     # each walker = [position, step, stop (exclusive)]
        live = [wk for wk in walkers]
        while live and len(taken) < q:
            nxt = []
            for wk in live:
                p, step, stop = wk
                while p != stop and (w[p] != 0 or p in seen):
                    p += step
                if p != stop:
                    seen.add(p)
                    taken.append(int(p))
                    wk[0] = p + step
                    if wk[0] != stop:
                        nxt.append(wk)
                if len(taken) >= q:
                    break
            live = nxt

    if not intervals and not seeds:
        walk([[0, 1, n]])
        return taken
    first_non = np.flatnonzero(w < 0)
    first_non = int(first_non[0]) if first_non.size else n

    def real_top(a: int) -> bool:                    # a labelled non-match lies above the boundary
        return first_non < a

    inward = []
    for a, b in intervals:
        if real_top(a):
            mid = (a + b) // 2
            inward.append([b, -1, mid])              # from the low-score boundary upwards
            inward.append([a, 1, mid + 1])           # from the high-score boundary downwards
        else:                                        # anchored at the match end: one cut, as in
            inward.append([b, -1, -1])               # optimized_single's match-side-first scan
    walk(inward)
    outward = []
    for a, b in list(intervals) + [(s_, s_) for s_ in seeds]:
        if b + 1 < n:
            outward.append([b + 1, 1, n])
        if a - 1 >= 0:
            outward.append([a - 1, -1, -1])
    walk(outward)
    return taken
