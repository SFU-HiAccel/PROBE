"""Fixed-score training sampler. Oracle takes an original candidate ID, returns 0/1.

This module owns B_L only. The caller must also enforce the physical request
budget (including retries) at the oracle transport shared by L/A/D. Audit bounds
are fixed-time SRSWOR bounds for the supplied oracle, not human ground truth.
"""
from collections import deque
from dataclasses import dataclass, field
from math import ceil
from typing import Callable, Hashable, MutableMapping

import numpy as np
from scipy.stats import hypergeom

from sampling_strata import geometric_strata
from threshold_search_multi import penalised_intervals, seed_rows, build_windows, allocate_batch


@dataclass(frozen=True)
class SamplingConfig:
    batch_size: int = 32
    pilot_fraction: float = .3
    pilot_mode: str = 'uniform'
    pilot_mix: tuple = (.25, .25, .5)  # top, geometric, uniform audit
    p_star: float = .05
    delta_sel: float = .05
    mu: float = 1.5
    k_max: int = 6
    explore_fraction: float = .25
    n_min: int = 2
    train_strata: int = 10
    seed: int = 0
    policy: str = 'gated'  # explicit structural ablations: multi / stratified
    fair_seeds: bool = True
    # 'legacy': hybrid-pilot protocol of the saved experiments. 'paper': Algorithms 1-2 of the
    # paper (uniform pilot without batch minimum, one-sided gate, high-to-low scan before the first
    # positive, geometric strata with minimum quota + pilot Neyman weights). See _paper_sas.
    protocol: str = 'paper'

    def __post_init__(self):
        if self.batch_size < 1 or self.train_strata < 1 or self.k_max < 1 or self.n_min < 1:
            raise ValueError('batch_size, train_strata, k_max and n_min must be positive')
        if not (0 <= self.pilot_fraction <= 1 and 0 <= self.p_star <= 1
                and 0 < self.delta_sel < 1 and 0 <= self.explore_fraction <= 1):
            raise ValueError('invalid probability/fraction')
        if len(self.pilot_mix) != 3 or min(self.pilot_mix) < 0 or not np.isclose(sum(self.pilot_mix), 1):
            raise ValueError('pilot_mix must be three nonnegative shares summing to one')
        if self.pilot_mode not in ('hybrid', 'uniform', 'strata') or self.policy not in ('gated', 'multi', 'stratified'):
            raise ValueError('unknown pilot mode or policy')
        if not np.isfinite(self.mu) or self.mu < 0:
            raise ValueError('mu must be nonnegative and finite')
        if self.protocol not in ('legacy', 'paper'):
            raise ValueError('protocol must be legacy or paper')


def finite_population_interval(N, n, k, delta=.05):
    """Invert both hypergeometric tails, allocating delta/2 to each tail.

    Returns integer population-positive bounds. No sample => [0,N].
    Binary searches exploit monotonicity in the unknown population count.
    """
    if not (0 <= k <= n <= N and 0 < delta < 1):
        raise ValueError('invalid finite-population observation')
    lo, hi = k, N - n + k
    a, b = lo, hi
    while a < b:
        m = (a + b) // 2
        if hypergeom.sf(k - 1, N, m, n) >= delta / 2:
            b = m
        else:
            a = m + 1
    lower = a
    a, b = lo, hi
    while a < b:
        m = (a + b + 1) // 2
        if hypergeom.cdf(k, N, m, n) >= delta / 2:
            a = m
        else:
            b = m - 1
    return lower, a


@dataclass
class PilotState:
    budget: int
    audit_ids: list = field(default_factory=list)
    sampled_ids: list = field(default_factory=list)
    positives: list = field(default_factory=list)
    interval: tuple = (0., 1.)
    regime: str = 'unknown'
    mode: str = 'multi'


@dataclass
class SamplingResult:
    order: list
    labels: dict
    pilot: PilotState
    trace: list
    attempted_ids: list
    failed_ids: list
    unresolved_ids: list
    stop_reason: str

    @property
    def verified_positives(self):
        return [i for i, y in self.labels.items() if y == 1]


def sample_fixed_ce(ids, scores, oracle: Callable[[Hashable], int], budget: int,
                    config=None, label_cache: MutableMapping | None = None, *,
                    batch_oracle: Callable[[list], list] | None = None):
    """One stable sort; persistent labels/windows; no GT argument or learned axis.

    B_L counts new oracle attempts, including failed ones. Cache hits are free.
    Failures are left unresolved and not retried by this sampler. Oracle errors
    disable pilot inference unless its complete predeclared audit was observed.
    """
    cfg = config or SamplingConfig()
    ids = list(ids)
    scores = np.asarray(scores, dtype=float)
    if len(set(ids)) != len(ids) or scores.shape != (len(ids),) or not np.isfinite(scores).all():
        raise ValueError('candidate IDs must be unique and scores finite/aligned')
    if budget < 0 or int(budget) != budget:
        raise ValueError('budget must be a nonnegative integer')
    N = len(ids)
    order = [ids[i] for i in np.argsort(-scores, kind='stable')]
    rank_by_id = {key: rank for rank, key in enumerate(order)}
    cache = {} if label_cache is None else label_cache
    if any(cache[i] not in (0, 1) for i in ids if i in cache):
        raise ValueError('cached labels must be binary')
    rng = np.random.default_rng(cfg.seed)
    audit_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 1]))
    audit_order = audit_rng.permutation(N)  # independent full-population prefix
    attempted, failed, trace = [], [], []
    unavailable = np.array([i in cache for i in order], dtype=bool)
    w = np.array([2 * cache[i] - 1 if i in cache else 0 for i in order], dtype=float)
    cap = min(int(budget), int((~unavailable).sum()))
    m0 = min(cap, max(cfg.batch_size, ceil(cfg.pilot_fraction * cap)))
    pilot = PilotState(m0)

    def buy(rows, stage):
        # Reserve all unique IDs before submitting each batch. A transport may
        # return None per failed row; a malformed/failed batch consumes its full
        # reservation and produces no fabricated labels.
        pending = []
        for row in rows:
            row = int(row)
            if unavailable[row] or len(attempted) >= cap:
                continue
            unavailable[row] = True
            attempted.append(order[row])
            pending.append(row)
        acquired = []
        chunk_size = cfg.batch_size if batch_oracle is not None else 1
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start:start + chunk_size]
            keys = [order[row] for row in chunk]
            try:
                values = list(batch_oracle(keys)) if batch_oracle is not None else [oracle(keys[0])]
                if len(values) != len(keys):
                    raise ValueError('oracle response length differs from reserved batch')
            except Exception as exc:
                failed.extend(keys)
                trace.append({'stage': stage, 'ids': keys, 'error': type(exc).__name__})
                continue
            for row, key, y in zip(chunk, keys, values):
                if not np.isscalar(y) or y not in (0, 1):
                    failed.append(key)
                    trace.append({'stage': stage, 'id': key, 'error': 'invalid_label'})
                    continue
                cache[key] = int(y)
                w[row] = 2 * int(y) - 1
                acquired.append(key)
        trace.append({'stage': stage, 'ids': acquired, 'attempts': len(attempted),
                      'positive_count': int((w > 0).sum())})

    def stratified_order(strata):
        queues = [deque(rng.permutation(s).tolist()) for s in strata]
        out = []
        while any(queues):
            for q in queues:
                if q:
                    out.append(q.popleft())
        return out

    if cfg.protocol == 'paper':
        _paper_sas(cfg, ids, order, rank_by_id, w, unavailable, cap, int(budget),
                   attempted, buy, rng, audit_order, pilot, trace, cache)
        labels = {i: cache[i] for i in ids if i in cache}
        unresolved = [i for i in ids if i not in cache]
        reason = 'pool_labelled' if not unresolved else 'budget_exhausted' if len(attempted) >= cap else 'oracle_failures'
        return SamplingResult(order, labels, pilot, trace, attempted, failed, unresolved, reason)

    if cfg.pilot_mode == 'uniform':
        n_top, n_geo, n_audit = 0, 0, m0
    elif cfg.pilot_mode == 'strata':
        n_top, n_geo, n_audit = 0, m0, 0
    else:
        # Largest remainder gives an exact fixed quota even for tiny budgets.
        raw = np.array(cfg.pilot_mix) * m0
        shares = np.floor(raw).astype(int)
        for i in np.argsort(-(raw - shares), kind='stable')[:m0 - shares.sum()]:
            shares[i] += 1
        n_top, n_geo, n_audit = shares.tolist()
    # Audit first: the complete predeclared prefix always fits, even with overlap.
    audit_rows = audit_order[:n_audit]
    pilot.audit_ids = [order[int(r)] for r in audit_rows]
    buy(audit_rows, 'pilot_audit')
    buy(np.flatnonzero(~unavailable)[:n_top], 'pilot_top')
    geo_order = stratified_order(geometric_strata(np.arange(N), cfg.train_strata))
    buy([r for r in geo_order if not unavailable[r]][:n_geo], 'pilot_geo')
    # Cache overlap can release quota; spend it without modifying the audit prefix.
    buy([r for r in geo_order if not unavailable[r]][:max(0, m0-len(attempted))], 'pilot_fill')
    pilot.sampled_ids = [i for i in attempted if i in cache]
    pilot.positives = [i for i in pilot.sampled_ids if cache[i]]
    if N and all(i in cache for i in pilot.audit_ids):
        bounds = finite_population_interval(N, n_audit, sum(cache[i] for i in pilot.audit_ids), cfg.delta_sel)
        pilot.interval = tuple(v / N for v in bounds)
        if pilot.interval[1] <= cfg.p_star:
            pilot.regime = 'highly_selective'
        elif pilot.interval[0] > cfg.p_star:
            pilot.regime = 'not_highly_selective'
    pilot.mode = ('stratified' if pilot.regime == 'not_highly_selective' else 'multi') if cfg.policy == 'gated' else cfg.policy
    # Fixed equal-rank strata; a pre-generated within-stratum permutation gives
    # SRSWOR at every deterministic allocation size. No label-adaptive allocation.
    strata_order = stratified_order(np.array_split(np.arange(N), min(N, cfg.train_strata))) if N else []
    queue, served = deque(), set()
    windows, r, exploration_credit, seed_credit = [], 0, 0., 0.
    c = max(1., N) ** (1 / max(1, ceil((cap-m0) / cfg.batch_size)))
    while len(attempted) < cap and (~unavailable).any():
        q = min(cfg.batch_size, cap-len(attempted))
        if pilot.mode == 'stratified':
            rows = [p for p in strata_order if not unavailable[p]][:q]
        else:
            intervals = penalised_intervals(w, cfg.mu, cfg.k_max)
            seeds = seed_rows(w, intervals)
            active = {order[s] for s in seeds}
            queue = deque(key for key in queue if key in active)
            queued = set(queue)
            for s in seeds:
                key = order[s]
                if key not in queued:
                    queue.append(key)
            r += 1
            windows, weights = build_windows(w, intervals, seeds, windows, r, c)
            mask = unavailable.copy()
            rows = []
            # Global uniform exploration can reach unseen clusters even when windows
            # cover the whole axis. Fractional credit avoids round-to-zero starvation.
            exploration_credit += cfg.explore_fraction*q
            ne = min(q, int(exploration_credit), int((~mask).sum()))
            exploration_credit -= ne
            if ne:
                chosen = rng.choice(np.flatnonzero(~mask), size=ne, replace=False)
                rows.extend(chosen.tolist()); mask[chosen] = True
            # Half of non-exploration capacity, with persistent remainder and FIFO
            # service across original IDs. Finite budget need not serve every seed.
            seed_credit += (q-ne) / 2 if queue and cfg.fair_seeds else 0
            ns = min(q-len(rows), int(seed_credit)) if cfg.fair_seeds else 0
            seed_credit -= ns
            for _ in range(ns):
                if not queue or not (~mask).any():
                    break
                key = queue.popleft(); queue.append(key)
                s = rank_by_id[key]
                free = np.flatnonzero(~mask)
                p = int(free[np.argmin(np.abs(free-s))])
                rows.append(p); mask[p] = True
                served.add(key)
            selected, _ = allocate_batch(windows, mask, q-len(rows), 0., weights, rng=rng)
            rows.extend(selected.tolist())
            trace.append({'stage': 'windows', 'intervals': intervals, 'seeds': seeds,
                          'windows': windows, 'served_seed_ids': list(served)})
        buy(rows, pilot.mode)
    labels = {i: cache[i] for i in ids if i in cache}
    unresolved = [i for i in ids if i not in cache]
    reason = 'pool_labelled' if not unresolved else 'budget_exhausted' if len(attempted) >= cap else 'oracle_failures'
    return SamplingResult(order, labels, pilot, trace, attempted, failed, unresolved, reason)


def one_sided_lower(N, n, k, delta=.05):
    """Paper Eq. (1): K_lower = min{K : P_{Hyp(N,K,n)}[X >= k] > delta}. No sample or no hit => 0."""
    if not (0 <= k <= n <= N and 0 < delta < 1):
        raise ValueError('invalid finite-population observation')
    if n == 0 or k == 0:
        return 0
    a, b = k, N - n + k          # sf(k-1) is non-decreasing in K, and equals 1 at K = N - n + k
    while a < b:
        m = (a + b) // 2
        if hypergeom.sf(k - 1, N, m, n) > delta:
            b = m
        else:
            a = m + 1
    return a


def paper_config(**overrides):
    """SamplingConfig with the paper defaults (Alg. 1-2): uniform pilot of 30% of B_L, p_star = 0.05, n_min = 2."""
    settings = dict(protocol='paper', pilot_mode='uniform', pilot_fraction=.3, p_star=.05, delta_sel=.05, n_min=2)
    settings.update(overrides)
    return SamplingConfig(**settings)


def one_sided_upper(N, n, k, delta=.05):
    """Paper Eq. (1): K_upper = max{K : P_{Hyp(N,K,n)}[X <= k] > delta}. No sample => N."""
    if not (0 <= k <= n <= N and 0 < delta < 1):
        raise ValueError('invalid finite-population observation')
    if n == 0:
        return N
    a, b = k, N - n + k          # cdf(k) is non-increasing in K, and equals 1 at K = k
    while a < b:
        m = (a + b + 1) // 2
        if hypergeom.cdf(k, N, m, n) > delta:
            a = m
        else:
            b = m - 1
    return a


def _stratum_quota(free_counts, weights, b, start):
    """Minimum quota of one per non-exhausted stratum (rotating start when b is smaller),
    then the rest by largest remainder on ``weights``, capped by what each stratum has left."""
    free_counts = np.asarray(free_counts, dtype=int)
    take = np.zeros(len(free_counts), dtype=int)
    live = [h for h in np.roll(np.arange(len(free_counts)), -start) if free_counts[h] > 0]
    for h in live[:b]:
        take[h] = 1
    left = b - int(take.sum())
    while left > 0:
        room = free_counts - take
        active = np.flatnonzero(room > 0)
        if active.size == 0:
            break
        wts = np.asarray(weights, dtype=float)[active]
        if wts.sum() <= 0:
            wts = free_counts[active].astype(float)
        raw = wts / wts.sum() * left
        give = np.minimum(np.floor(raw).astype(int), room[active])
        for i in np.argsort(-(raw - np.floor(raw)), kind='stable')[:left - int(give.sum())]:
            if give[i] < room[active][i]:
                give[i] += 1
        if give.sum() == 0:
            give[int(np.argmax(room[active]))] = 1
        take[active] += give
        left -= int(give.sum())
    return take


def _paper_sas(cfg, ids, order, rank_by_id, w, unavailable, cap, budget,
               attempted, buy, rng, audit_order, pilot, trace, cache):
    """Paper Algorithms 1-2 on the shared state of sample_fixed_ce (rows are ranks of ``order``)."""
    N = len(order)
    # Uniform pilot: m0 = min(N, floor(f_pilot * B_L)), no batch-size minimum (Alg. 1, lines 4-6).
    m0 = min(cap, int(np.floor(cfg.pilot_fraction * budget)))
    pilot.budget = m0
    audit_rows = [int(r) for r in audit_order[:m0]]
    pilot.audit_ids = [order[r] for r in audit_rows]
    buy(audit_rows, 'pilot_audit')
    seen = [order[r] for r in audit_rows if order[r] in cache]
    n0, k0 = len(seen), sum(cache[i] for i in seen)
    pilot.sampled_ids = seen
    pilot.positives = [i for i in seen if cache[i]]
    # Lower-bound gate (Alg. 1, lines 8-10): leave the multi-interval branch only when the pilot
    # shows, at confidence 1 - delta_sel, that the positive rate exceeds p_star.
    lower = one_sided_lower(N, n0, k0, cfg.delta_sel) / N if N else 0.
    pilot.interval = (lower, 1.)
    pilot.regime = 'not_highly_selective' if lower > cfg.p_star else 'highly_selective'
    if cfg.policy == 'gated':
        pilot.mode = 'multi' if pilot.regime == 'highly_selective' else 'stratified'
    else:
        pilot.mode = cfg.policy
    trace.append({'stage': 'gate', 'n0': n0, 'k0': k0, 'L0': lower, 'mode': pilot.mode})

    if pilot.mode == 'stratified':
        # Geometric strata fixed before labels; minimum quota + pilot Neyman weights N_h * S_h.
        strata = geometric_strata(np.arange(N), cfg.train_strata)
        queues = [deque(rng.permutation(s).tolist()) for s in strata]
        stratum_of = np.empty(N, dtype=int)
        for h, s in enumerate(strata):
            stratum_of[s] = h
        pos, cnt = np.zeros(len(strata)), np.zeros(len(strata))
        for row in audit_rows:
            if order[row] in cache:
                cnt[stratum_of[row]] += 1
                pos[stratum_of[row]] += cache[order[row]]
        p_hat = np.divide(pos, cnt, out=np.zeros_like(pos), where=cnt > 0)
        weights = [len(s) * np.sqrt(p * (1 - p)) for s, p in zip(strata, p_hat)]
        batch_no = 0
        while len(attempted) < cap and (~unavailable).any():
            for q in queues:
                while q and unavailable[q[0]]:
                    q.popleft()
            b = min(cfg.batch_size, cap - len(attempted), int((~unavailable).sum()))
            take = _stratum_quota([len(q) for q in queues], weights, b, batch_no % max(1, len(queues)))
            rows = [queues[h].popleft() for h in range(len(queues)) for _ in range(int(take[h]))]
            trace.append({'stage': 'strata_alloc', 'take': take.tolist()})
            buy(rows, 'stratified')
            batch_no += 1
        return

    # Multi-interval (Alg. 2): exploration share every batch; Explore (scan) phase until L holds n_min
    # positives, then the Exploit phase (intervals + seeds) with c fixed at its first batch.
    queue, served = deque(), set()
    windows, r, q_e, seed_credit, c = [], 0, 0., 0., None
    while len(attempted) < cap and (~unavailable).any():
        b = min(cfg.batch_size, cap - len(attempted))
        mask = unavailable.copy()
        rows = []
        q_e += cfg.explore_fraction * b
        ne = min(b, int(q_e), int((~mask).sum()))
        q_e -= ne
        if ne:
            chosen = rng.choice(np.flatnonzero(~mask), size=ne, replace=False)
            rows.extend(chosen.tolist()); mask[chosen] = True
        rest = b - len(rows)
        if int((w > 0).sum()) < cfg.n_min:
            top = np.flatnonzero(~mask)[:rest]          # Explore phase: highest-ranked unlabelled rows
            rows.extend(top.tolist())
            trace.append({'stage': 'scan', 'n': int(top.size)})
        else:
            if c is None:
                m = max(1, ceil((cap - len(attempted)) / cfg.batch_size))
                c = max(1., N) ** (1 / m)
            intervals = penalised_intervals(w, cfg.mu, cfg.k_max)
            seeds = seed_rows(w, intervals)
            active = {order[s] for s in seeds}
            queue = deque(key for key in queue if key in active)
            queued = set(queue)
            for s in seeds:
                if order[s] not in queued:
                    queue.append(order[s])
            r += 1
            windows, weights = build_windows(w, intervals, seeds, windows, r, c)
            seed_credit += rest / 2 if queue and cfg.fair_seeds else 0
            ns = min(rest, int(seed_credit)) if cfg.fair_seeds else 0
            seed_credit -= ns
            for _ in range(ns):
                if not queue or not (~mask).any():
                    break
                key = queue.popleft(); queue.append(key)
                free = np.flatnonzero(~mask)
                p = int(free[np.argmin(np.abs(free - rank_by_id[key]))])
                rows.append(p); mask[p] = True
                served.add(key)
            selected, _ = allocate_batch(windows, mask, b - len(rows), 0., weights, rng=rng)
            rows.extend(selected.tolist())
            trace.append({'stage': 'windows', 'intervals': intervals, 'seeds': seeds,
                          'windows': windows, 'served_seed_ids': list(served), 'c': c})
        buy(rows, 'multi')


def sample_fixed_ce_texts(scores, texts, label_fn, sample_size, *, config=None,
                          original_ids=None, label_cache=None):
    """Adapter for cluster_proxy's batched text-labeling interface.

    The returned indices are local text positions, while cache/trace IDs remain
    original IDs. The caller must supply verified CE scores; this adapter cannot
    distinguish cosine values from CE values by inspecting the numeric array.
    """
    from dataclasses import asdict
    import time
    started = time.perf_counter()
    ids = list(range(len(texts))) if original_ids is None else list(original_ids)
    if len(ids) != len(texts):
        raise ValueError('original_ids must align with texts')
    positions = {key: i for i, key in enumerate(ids)}
    result = sample_fixed_ce(
        ids, scores, None, sample_size, config, label_cache,
        batch_oracle=(lambda keys: label_fn.label_ids(keys, stage="L"))
        if hasattr(label_fn, "label_ids") else
        (lambda keys: label_fn([texts[positions[key]] for key in keys])),
    )
    indices = np.array([positions[key] for key in result.labels], dtype=np.int64)
    values = np.array(list(result.labels.values()), dtype=np.int32)
    profile = {
        'sampler': 'fixed_ce_pilot', 'graph_time': 0.,
        'config': asdict(config or SamplingConfig()),
        'sampling_seconds': time.perf_counter() - started,
        'pilot_state': asdict(result.pilot), 'trace': result.trace,
        'attempted_ids': result.attempted_ids, 'failed_ids': result.failed_ids,
        'unresolved_ids': result.unresolved_ids, 'stop_reason': result.stop_reason,
        'verified_positive_ids': result.verified_positives,
        'sort_count': 1, '_final_axis': np.asarray(scores, dtype=float).copy(),
        '_fixed_order': np.array([positions[key] for key in result.order], dtype=np.int64),
    }
    return indices, values, None, profile
