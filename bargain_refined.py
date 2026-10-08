"""BARGAIN on PROBE's graph-refined proxy scores (reviewer control for Fig 10).

Replaces PROBE's learned calibration + cost-aware routing with BARGAIN-A (accuracy target), BARGAIN-P
(precision target) or BARGAIN-R (recall target), keeping everything before it (SAS sample L, LR proxy, graph refinement) unchanged.
Enabled only when PROBE_BARGAIN_MODE is set, so default PROBE runs are unaffected.

  PROBE_BARGAIN_MODE    A | P | R
  PROBE_BARGAIN_TARGET  accuracy (A), precision (P) or recall (R) target, default 0.9
  PROBE_BARGAIN_DELTA   failure probability, default 0.1
  PROBE_BARGAIN_BUDGET_FRAC  BARGAIN-P oracle budget as a fraction of N, default 1 (no limit)
  PROBE_BARGAIN_R_BUDGET_FRAC  BARGAIN-R oracle sample budget as a fraction of N, default 0.1. BARGAIN-R labels a
                        sample of this size to place its threshold and returns every row above it, so a budget of N
                        would label nearly all rows. 0.1 matches PROBE's 10% sampling budget.
  PROBE_BARGAIN_ROOT    path holding the BARGAIN package, default third_party/bargain

Proxy: label = refined score f >= 0.5, confidence = f for a positive label and 1 - f otherwise, so BARGAIN
ranks rows by f exactly as PROBE does. Oracle: the run's shared labeling session (stage "BARGAIN_route"),
so labels already in L are reused for free and every new call is counted and timed by run.py.
BARGAIN-P runs with budget = N (no call limit) by default, as in the paper's main BARGAIN baseline. With spare
budget BARGAIN-P labels the rows below its threshold too, so an unlimited run approaches full LLM cost.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)
STAGE = "BARGAIN_route"


def bargain_mode() -> str | None:
    mode = os.environ.get("PROBE_BARGAIN_MODE", "").strip().upper()
    if not mode:
        return None
    if mode not in {"A", "P", "R"}:
        raise ValueError(f"PROBE_BARGAIN_MODE must be A, P or R, got {mode!r}")
    return mode


def _import_bargain():
    from pathlib import Path
    root = os.environ.get("PROBE_BARGAIN_ROOT", str(Path(__file__).resolve().parent / "third_party/bargain"))
    if root not in sys.path:
        sys.path.insert(0, root)
    from BARGAIN.models.AbstractModels import Oracle, Proxy
    from BARGAIN.process.BARGAIN_A import BARGAIN_A
    from BARGAIN.process.BARGAIN_P import BARGAIN_P
    from BARGAIN.process.BARGAIN_R import BARGAIN_R
    return Proxy, Oracle, BARGAIN_A, BARGAIN_P, BARGAIN_R


def run_bargain_on_refined(refined_score: np.ndarray, label_fn: Any, labeled_indices: np.ndarray,
                           labeled_values: np.ndarray, *, seed: int, profiling: Dict[str, Any]) -> np.ndarray:
    """Run BARGAIN over all rows with the refined score as proxy. Returns the final positive mask."""
    mode = bargain_mode()
    target = float(os.environ.get("PROBE_BARGAIN_TARGET", "0.9"))
    delta = float(os.environ.get("PROBE_BARGAIN_DELTA", "0.1"))
    Proxy, Oracle, BargainA, BargainP, BargainR = _import_bargain()

    f = np.clip(np.asarray(refined_score, dtype=np.float64), 0.0, 1.0)
    n = int(f.size)
    pred = (f >= 0.5).astype(int)
    conf = np.where(pred == 1, f, 1.0 - f)
    known = {int(i): int(v) for i, v in zip(np.asarray(labeled_indices), np.asarray(labeled_values))}
    requested: set[int] = set()
    unresolved: set[int] = set()

    def oracle_labels(ids: Sequence[int]) -> np.ndarray:
        ids = [int(i) for i in ids]
        if not ids:
            return np.zeros(0, dtype=int)
        requested.update(ids)
        raw = label_fn.label_ids(ids, stage=STAGE)
        out = np.zeros(len(ids), dtype=int)
        for k, (i, v) in enumerate(zip(ids, raw)):
            if v in (0, 1, True, False):
                out[k] = int(v)
            else:
                unresolved.add(i)
                from oracle_budget import OracleLabelsUnavailable
                raise OracleLabelsUnavailable("BARGAIN received an unresolved oracle label")
        return out

    class RefinedProxy(Proxy):
        def get_preds_and_scores(self, indxs: List, data_records: List):
            idx = np.asarray(indxs, dtype=int)
            return pred[idx], conf[idx]

    class SessionOracle(Oracle):
        def get_pred(self, data_records: List, indxs: List = None) -> np.ndarray:
            return oracle_labels(list(indxs))

        def is_answer_correct(self, data_indxs: List, data_records: List, proxy_output_at_indxs: List) -> np.ndarray:
            truth = oracle_labels(list(data_indxs))
            return truth == np.asarray(proxy_output_at_indxs, dtype=int)

    records = np.arange(n)
    t0 = time.time()
    if mode == "A":
        algo = BargainA(RefinedProxy(verbose=False), SessionOracle(verbose=False), target=target, delta=delta,
                        seed=seed, verbose=False)
        out, used_oracle = algo.process(records, return_oracle_usage=True)
        mask = np.asarray(out, dtype=int) == 1
        n_proxy_rows = int(n - np.sum(used_oracle))
    elif mode == "R":
        budget = max(1, int(round(float(os.environ.get("PROBE_BARGAIN_R_BUDGET_FRAC", "0.1")) * n)))
        profiling["bargain_budget"] = budget
        algo = BargainR(RefinedProxy(verbose=False), SessionOracle(verbose=False), delta=delta, target=target,
                        budget=budget, seed=seed)
        mask = np.zeros(n, dtype=bool)
        mask[np.asarray(algo.process(records), dtype=int)] = True
        n_proxy_rows = -1
    else:
        budget = max(1, int(round(float(os.environ.get("PROBE_BARGAIN_BUDGET_FRAC", "1")) * n)))
        profiling["bargain_budget"] = budget
        algo = BargainP(RefinedProxy(verbose=False), SessionOracle(verbose=False), delta=delta, target=target,
                        budget=budget, seed=seed)
        mask = np.zeros(n, dtype=bool)
        mask[np.asarray(algo.process(records), dtype=int)] = True
        n_proxy_rows = -1
    elapsed = time.time() - t0

    for i, v in known.items():  # rows of L keep their LLM label
        mask[i] = v == 1
    new_calls = len(requested - set(known))
    profiling.update(
        bargain_mode=mode, bargain_target=target, bargain_delta=delta, bargain_time_s=float(elapsed),
        bargain_oracle_rows=len(requested), bargain_new_calls=int(new_calls),
        bargain_unresolved=len(unresolved), bargain_proxy_rows=n_proxy_rows,
        # keys run.py already reads for the delegation count / rollback cost
        defer_enabled=True, defer_decision_mode=f"bargain_{mode}", defer_verify_uncertain=True,
        defer_verify_count=int(new_calls), defer_candidates_before_cap=int(new_calls),
        defer_router_cap=None, defer_total_time=float(elapsed),
        defer_verify_positive=int(mask[list(requested)].sum()) if requested else 0,
    )
    logger.info(f"[bargain_refined] mode={mode} target={target} delta={delta} n={n} "
                f"new_calls={new_calls} positives={int(mask.sum())} time={elapsed:.1f}s")
    return mask
