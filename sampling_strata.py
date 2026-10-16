"""Rank strata, extracted from PROBE_zp threshold_search.py."""
from typing import List
import numpy as np

def geometric_strata(order: np.ndarray, max_strata: int) -> List[np.ndarray]:
    """Split ``order`` (row ids, highest score first) into strata that double in width.

    With S = min(max_strata, floor(log2 n) + 1) strata, stratum k spans ~ w0 * 2^k rows with
    w0 = n / (2^S - 1). Equal draws per stratum then give a sampling density that falls as 1/rank:
    close to top-K when the axis is right, but still reaching rows far below the budget when it is not.
    """
    order = np.asarray(order).reshape(-1)
    n = order.size
    if n == 0:
        return []
    s = int(max(1, min(int(max_strata), int(np.floor(np.log2(n))) + 1)))
    w0 = n / (2.0 ** s - 1.0)
    bounds = [int(np.ceil(w0 * (2.0 ** k - 1.0))) for k in range(s + 1)]
    bounds[-1] = n
    strata = [order[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]
    return strata
