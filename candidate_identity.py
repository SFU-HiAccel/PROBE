"""Candidate eligibility is independent of labels, scores and row ordering."""


def exclude_reflexive_pairs(pairs, left, right, id_columns=None):
    if id_columns is None:
        # Compatibility for older explicit positional self-join callers.
        return [(i,j) for i,j in pairs if len(left)!=len(right) or i!=j]
    if len(id_columns)!=2:
        raise ValueError('self_pair_id_columns requires one column per input')
    lc,rc=id_columns
    if any(lc not in row or row[lc] is None for row in left) or any(rc not in row or row[rc] is None for row in right):
        raise ValueError('self-pair keys must exist and be non-null')
    return [(i,j) for i,j in pairs if str(left[i][lc]) != str(right[j][rc])]
