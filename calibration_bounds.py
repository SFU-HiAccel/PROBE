"""Fixed-design finite-population certificates, never for adaptive stopping.

A is SRSWOR within a fixed partition with counts fixed before observing A.
The resulting simultaneous population-count event supports any subsequent
output/routing rule. Oracle labels must be correct and stationary for this
finite population; the bound is not a statement about human ground truth.
"""
import numpy as np
from scipy.stats import hypergeom


def hypergeom_upper_total(k, n, size, alpha):
    if not 0 <= k <= n <= size or not 0 < alpha < 1:
        raise ValueError('invalid hypergeometric observation or error probability')
    if n == 0:
        return int(size)
    low, high = int(k), int(size-n+k)
    while low < high:
        mid = (low+high+1)//2
        if hypergeom.cdf(k,size,mid,n) >= alpha:
            low = mid
        else:
            high = mid-1
    return low


def missed_positive_bound(pool, design, y_known, d_idx, d_y, final_pos, eta=.05):
    """Return missed-positive and pre-routing unobserved-positive upper counts.

    pool is the complete population outside L. In particular, A=0 must retain
    this population even when there is no StratifiedDesign object.
    """
    if not 0 < eta < 1:
        raise ValueError('eta must be in (0,1)')
    pool = np.asarray(pool, dtype=np.int64)
    if len(set(pool.tolist())) != len(pool):
        raise ValueError('pool IDs must be unique')
    strata = [pool] if design is None else list(design.strata)
    sampled = [np.empty(0,dtype=np.int64)] if design is None else list(design.sampled)
    flattened = np.concatenate(strata) if strata else np.empty(0,dtype=np.int64)
    if sorted(flattened.tolist()) != sorted(pool.tolist()):
        raise ValueError('strata must partition the entire pool, including unsampled rows')
    d_idx = np.asarray(d_idx,dtype=np.int64)
    d_y = np.asarray(d_y)
    if len(d_idx) != len(d_y) or len(set(d_idx.tolist())) != len(d_idx) or not np.isin(d_y,[0,1]).all():
        raise ValueError('routing labels must be unique/aligned/binary')
    final_pos = np.asarray(final_pos,dtype=bool)
    d_pos = d_idx[d_y==1]
    missed, upper_pool = 0,0
    for stratum, taken in zip(strata,sampled):
        taken = np.asarray(taken,dtype=np.int64)
        if not np.isin(taken,stratum).all() or len(set(taken.tolist())) != len(taken):
            raise ValueError('invalid calibration IDs')
        values = np.asarray(y_known)[taken]
        if not np.isin(values,[0,1]).all():
            raise ValueError('calibration labels must be observed and binary')
        k = int(values.sum())
        rest = np.setdiff1d(stratum,taken)
        upper = hypergeom_upper_total(k,len(taken),len(stratum),eta/max(1,len(strata)))-k
        upper_pool += upper
        # Any adaptive verification can remove its known positives from this
        # already-valid population bound. It cannot add evidence to A's design.
        unresolved_neg = int((~np.isin(rest,d_idx) & ~final_pos[rest]).sum())
        missed += min(unresolved_neg,max(0,upper-int(np.isin(rest,d_pos).sum())))
    return int(missed),int(upper_pool)
