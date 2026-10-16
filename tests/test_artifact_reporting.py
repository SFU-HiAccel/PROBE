import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("paper_report", Path(__file__).resolve().parents[1] / "experiments/report.py")
report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)


def rows(seeds):
    return [dict(experiment="main", query="fixture", system="PROBE", model="test", metric="binary_f1",
                 point="{}", seed=s, f1=.5 + s*.01, delegation=.1, latency=10.) for s in seeds]


def test_incomplete_seeds_have_no_borrowed_ci():
    group = report.aggregate(rows([0, 1]), {"main": [0, 1, 2, 3, 4]})[0]
    assert not group["complete"] and group["f1_ci95_halfwidth"] is None
    full = report.aggregate(rows([0, 1, 2, 3, 4]), {"main": [0, 1, 2, 3, 4]})[0]
    assert full["complete"] and full["f1_ci95_halfwidth"] is None
    assert full["f1_mean"] == pytest.approx(.52)


def test_duplicate_seed_is_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        report.aggregate(rows([0, 0]), {"main": [0, 1, 2, 3, 4]})


def test_geometric_mean_zero_and_missing_policy():
    assert report.geometric_mean([.5, 0., .8]) == 0.
    assert report.geometric_mean([.5, .8]) == pytest.approx((.4)**.5)
    with pytest.raises(ValueError):
        report.geometric_mean([])


def test_different_source_or_weight_revisions_cannot_form_five_seeds():
    data=rows(range(5))
    for row in data:row['identity']={'source':'same','revision':'one'}
    data[-1]['identity']['revision']='two'
    with pytest.raises(ValueError,match='Incompatible seed'):
        report.aggregate(data,{'main':list(range(5))})


def test_refinement_comparison_requires_shared_labels():
    data=rows([0,0])
    for row in data:
        row.update(experiment='refinement',point='{"budget_pct": 10}',
                   shared_identity={'source':'same'},acquisition='one')
    data[-1]['acquisition']='two'
    with pytest.raises(ValueError,match='share acquired labels'):
        report.validate_pairing(data)
