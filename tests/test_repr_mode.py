"""PROBE_REPR: one model supplies the fixed sampling axis and the proxy/graph features."""
import numpy as np
import pytest
from pyspark.sql import SparkSession

import ce_ranker
import cluster_proxy as cp
import oracle_budget
import proxy_model as pm
from fixed_ce_sampling import SamplingConfig as _SamplingConfig
from learned_calibration import LearnedCalibrationConfig

D = 5          # bi-encoder embedding dim in these doubles
H = 7          # cross-encoder hidden dim in these doubles


def test_repr_mode_validation(monkeypatch):
    monkeypatch.setenv("PROBE_REPR", "colbert")
    with pytest.raises(ValueError):
        pm._repr_mode()
    monkeypatch.delenv("PROBE_REPR")
    assert pm._repr_mode() == "ce"


def test_reduce_rows_is_unit_norm_and_capped():
    x = np.random.default_rng(0).normal(size=(40, 20))
    y = pm._reduce_rows(x, rank=8)
    assert y.shape == (40, 8)
    np.testing.assert_allclose(np.linalg.norm(y, axis=1), 1, atol=1e-5)
    assert pm._reduce_rows(x[:4], rank=8).shape == (4, 3)   # at most n - 1 components


def test_pair_features_are_product_and_absdiff():
    rng = np.random.default_rng(1)
    left, right = rng.normal(size=(3, 4)), rng.normal(size=(2, 4))
    pairs = [(0, 0), (2, 1)]
    f = pm._build_pair_hadamard_absdiff(pairs, left, right, rank=4)
    ln = left / np.linalg.norm(left, axis=1, keepdims=True)
    rn = right / np.linalg.norm(right, axis=1, keepdims=True)
    assert f.shape == (2, 8)
    # With rank == d the PCA is a rotation, so sum(Pu*Pv) is still the cosine.
    np.testing.assert_allclose(f[:, :4].sum(1), [ln[0] @ rn[0], ln[2] @ rn[1]], atol=1e-5)
    assert (f[:, 4:] >= 0).all()


@pytest.fixture(scope="module")
def spark():
    session = (SparkSession.builder.master("local[1]").appName("repr-mode")
               .config("spark.ui.enabled", "false").config("spark.driver.host", "127.0.0.1")
               .config("spark.sql.shuffle.partitions", "1")
               .config("spark.sql.execution.arrow.pyspark.enabled", "true").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.mark.parametrize("kind", ["filter", "join"])
@pytest.mark.parametrize("mode", ["mixed", "biencoder", "ce"])
def test_repr_mode_routes_axis_and_features(spark, monkeypatch, kind, mode):
    monkeypatch.setenv("PROBE_REPR", mode)
    monkeypatch.setattr(cp, "_KNN_CACHE_ENABLED", False)

    def embeddings(rows, *args, **kwargs):
        return np.random.default_rng(len(rows)).normal(size=(len(rows), D))
    monkeypatch.setattr(pm, "embed_texts", embeddings)
    monkeypatch.setattr(pm, "embed_query", lambda *a, **k: np.ones((1, D)))

    def sides(left, right, *args, **kwargs):
        l, r = embeddings(left), embeddings(right)
        return dict(left_embs=l, right_embs=r, left_text_embs=l, right_text_embs=r,
                    left_img_embs=None, right_img_embs=None)
    monkeypatch.setattr(pm, "embed_rows_independent", sides)
    ce_calls = []

    def ce(*args, return_hidden=False, **kwargs):
        ids = args[4] if len(args) == 5 else args[7]
        ce_calls.append(return_hidden)
        s = np.arange(len(ids), dtype=np.float32)
        h = np.random.default_rng(9).normal(size=(len(ids), H)).astype(np.float32)
        return (s, {"fingerprint": "t"}, h) if return_hidden else (s, {"fingerprint": "t"})
    monkeypatch.setattr(ce_ranker, "score_filter", ce)
    monkeypatch.setattr(ce_ranker, "score_join", ce)
    seen = {}
    real = cp.cluster_proxy_pipeline

    def spy(features, *args, **kwargs):
        seen["features"] = np.asarray(features)
        seen["lr_features"] = kwargs.get("lr_features")
        seen["axis"] = np.asarray(kwargs.get("fixed_ce_scores"))
        seen["similarities"] = np.asarray(kwargs.get("similarities"))
        return real(features, *args, **kwargs)
    monkeypatch.setattr(cp, "cluster_proxy_pipeline", spy)

    class Request:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def __call__(self, key): return int(key % 3 == 0), {}
    monkeypatch.setattr(oracle_budget, "OpenAIOracleRequest", Request)
    cfg = LearnedCalibrationConfig(total_budget=12, sample_frac=.5, calib_frac=.4, route_frac=.1,
                                   fixed_calibration=True, seed=0)
    kwargs = dict(model="deterministic-test", url="unused", embedding_model="test",
                  sample_size=cfg.sample_budget, sampling_method="activeSampling", budget_size=10,
                  fixed_ce_config=SamplingConfig(batch_size=2, seed=0),
                  decision_mode="learned_calibration", learned_calibration_config=cfg,
                  defer_enabled=True, refine_enabled=True, knn_k=3)
    if kind == "filter":
        df = spark.range(24).selectExpr("id", "concat('row ', CAST(id AS STRING)) AS text")
        pm.proxy_filter(df, "Relevant? {text}", **kwargs).count()
    else:
        left = spark.range(4).selectExpr("id AS left_id", "concat('l', CAST(id AS STRING)) AS left_text")
        right = spark.range(6).selectExpr("id AS right_id", "concat('r', CAST(id AS STRING)) AS right_text")
        pm.proxy_join(left, right, "Related? {left_text} {right_text}", **kwargs).count()

    n = 24
    assert seen["features"].shape[0] == n and seen["lr_features"] is None
    if mode == "biencoder":
        assert ce_calls == []                                      # no cross-encoder pass at all
        np.testing.assert_allclose(seen["axis"], seen["similarities"])   # axis = the model's cosine
        # filter: the row embeddings; join: [Pu*Pv, |Pu-Pv|] with r = min(128, D, n_rows - 1) = D
        assert seen["features"].shape[1] == (D if kind == "filter" else 2 * D)
    elif mode == "ce":
        assert ce_calls == [True]
        np.testing.assert_array_equal(seen["axis"], np.arange(n))
        assert seen["features"].shape[1] == H                      # PCA keeps min(256, H, n - 1)
        np.testing.assert_allclose(np.linalg.norm(seen["features"], axis=1), 1, atol=1e-5)
    else:
        assert ce_calls == [False]
        np.testing.assert_array_equal(seen["axis"], np.arange(n))
        assert seen["features"].shape[1] == D                      # unchanged default behaviour


def SamplingConfig(**kwargs):
    # These tests exercise saved legacy hard-budget contracts explicitly.
    kwargs.setdefault("protocol", "legacy")
    kwargs.setdefault("pilot_fraction", .1)
    kwargs.setdefault("pilot_mode", "hybrid")
    kwargs.setdefault("n_min", 1)
    return _SamplingConfig(**kwargs)
