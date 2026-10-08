from candidate_identity import exclude_reflexive_pairs


def test_duplicate_and_reordered_self_join_keys():
    left=[{'l':'a'},{'l':'a'},{'l':'b'}]
    right=[{'r':'b'},{'r':'a'},{'r':'a'}]
    pairs=[(i,j) for i in range(3) for j in range(3)]
    assert exclude_reflexive_pairs(pairs,left,right,('l','r')) == [(0,0),(1,0),(2,1),(2,2)]


def test_keys_work_for_unequal_inputs():
    assert exclude_reflexive_pairs([(0,0),(0,1)],[{'l':1}],[{'r':'1'},{'r':'2'}],('l','r')) == [(0,1)]
