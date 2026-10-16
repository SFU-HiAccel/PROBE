"""Graph equations and image-only case study from the current manuscript."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

def test_gaussian_graph_matches_normalized_union_reference(monkeypatch):
    import cluster_proxy as cp
    monkeypatch.setattr(cp, '_select_knn_backend', lambda: 'sklearn')
    e = np.array([[10., 0.], [2., 1.], [1., 2.], [-1., 1.]])
    norm = e / np.linalg.norm(e, axis=1, keepdims=True)
    dist = np.linalg.norm(norm[:, None, :] - norm[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    neighbors = np.argmin(dist, axis=1)
    sigma = np.median(dist[np.arange(4), neighbors])
    w = np.zeros((4, 4))
    w[np.arange(4), neighbors] = np.exp(-dist[np.arange(4), neighbors]**2/(2*sigma*sigma))
    expected = np.maximum(w, w.T)
    graph = cp._build_knn_graph(e, 1, cache=False)
    np.testing.assert_allclose(graph.toarray(), expected, atol=1e-12)


def test_duplicate_embeddings_never_create_self_edges(monkeypatch):
    import cluster_proxy as cp
    monkeypatch.setattr(cp, '_select_knn_backend', lambda: 'sklearn')
    graph = cp._build_knn_graph(np.ones((6, 2)), 3, cache=False)
    assert not np.any(graph.diagonal())
    assert np.all(graph.getnnz(axis=1) >= 3)


def test_both_refinement_phases_match_algorithm_three():
    import cluster_proxy as cp
    from scipy.sparse import csr_matrix
    emb = np.random.default_rng(7).normal(size=(8, 3))
    w = np.ones((8, 8)) - np.eye(8)
    idx, y = np.array([0, 2, 5, 7]), np.array([0, 1, 0, 1])
    lr = LogisticRegression(C=1., class_weight='balanced', max_iter=1000)
    lr.fit(emb[idx], y)
    z = lr.predict_proba(emb)[:, 1]
    p = w / w.sum(axis=1, keepdims=True)
    bias = np.zeros(8); bias[idx] = y-z[idx]
    for _ in range(4):
        bias = .8 * p @ bias; bias[idx] = y-z[idx]
    corrected = np.clip(z+bias, 0, 1); corrected[idx] = y
    f = corrected.copy()
    for _ in range(6):
        f = .2*corrected + .8*p@f; f[idx] = y
    mask, score, prof = cp.predict_with_score_refinement(emb, csr_matrix(w), idx, y, .5,
        num_iter_bias=4, num_iter_prop=6, tol=0)
    np.testing.assert_allclose(prof['z_corrected'], corrected, atol=1e-12)
    np.testing.assert_allclose(score, f, atol=1e-12)
    np.testing.assert_array_equal(mask, f >= .5)


def test_cars_q7_gt_is_restricted_to_dented_images(tmp_path):
    import pandas as pd
    from experiments.workload_ground_truth import cars_q7_image_ground_truth
    root=tmp_path/'SemBench/files/cars/data'
    (root/'sf_157376').mkdir(parents=True);(root/'full_data').mkdir()
    images=root/'sf_157376/image_car_data_157376.csv'
    images.write_text('image_id,car_id\ni1,c1\ni2,c2\ni3,c3\n')
    annotations=root/'full_data/image_data_full.csv'
    annotations.write_text('image_id,car_id,damage_status\ni1,c1,dented\ni2,c2,normal\ni3,c3,dented;paint_scratches\ni4,c4,dented\n')
    path=cars_q7_image_ground_truth(tmp_path,'157376',tmp_path/'result')
    assert pd.read_csv(path).car_id.tolist()==['c1','c3']
    annotations.write_text('image_id,car_id,damage_status\ni1,c1,dented\n')
    with pytest.raises(ValueError,match='Missing Cars Q7'):
        cars_q7_image_ground_truth(tmp_path,'157376',tmp_path/'other')


def test_cars_q7_loader_keeps_evaluation_labels_out_of_candidates(tmp_path, monkeypatch):
    import run
    from pyspark.sql import SparkSession
    root=tmp_path/'SemBench/files/cars/data'
    (root/'sf_157376').mkdir(parents=True);(root/'full_data').mkdir()
    (root/'sf_157376/image_car_data_157376.csv').write_text(
        'image_id,car_id,image_path\ni1,c1,one.png\ni2,c2,two.png\n')
    (root/'full_data/image_data_full.csv').write_text(
        'image_id,car_id,damage_status\ni1,c1,dented\ni2,c2,normal\n')
    spark=SparkSession.builder.master('local[1]').appName('cars-q7-paper-loader').getOrCreate()
    seen={}
    monkeypatch.setattr(run,'get_spark',lambda:spark)
    monkeypatch.setattr(run,'DATA_ROOT',str(tmp_path))
    monkeypatch.setattr(run,'results_dir',str(tmp_path/'outputs'))
    monkeypatch.setattr(run,'_iter_emb_models',lambda query:['unit'])
    monkeypatch.setattr(run,'ABLATION_CONFIGS',[('unit','activeSampling',True)])
    monkeypatch.setattr(run,'KNN_K_VALUES',['auto'])
    monkeypatch.setattr(run,'all_results',[])
    monkeypatch.setattr(run,'get_budget_pcts_for_query',lambda query:[10])
    def execute(tag, df, prompt, gt_path, n, *args, **kwargs):
        seen.update(query=tag, columns=df.columns, n=n, prompt=prompt, rows=df.collect(),gt=gt_path)
        return df, 'unit', 0
    monkeypatch.setattr(run,'run_experiment_filter',execute)
    monkeypatch.setattr(run,'collect_result',lambda *args,**kwargs:{'unit':True})
    monkeypatch.setattr(run,'maybe_append_no_opt_filter',lambda *args,**kwargs:None)
    try:
        run.QUERY_RUNNERS['cars_q7']()
    finally:
        spark.stop()
    assert seen['n']==2 and seen['query']=='cars_q7' and 'dented' in seen['prompt']
    assert 'damage_status' not in seen['columns']
    assert all(r.image_path.startswith(str(tmp_path)) for r in seen['rows'])
    assert run.QUERY_MODALITY['cars_q7']=='multimodal'


def test_default_refinement_executes_exact_paper_rounds():
    from scipy.sparse import csr_matrix
    from cluster_proxy import predict_with_score_refinement
    embeddings=np.array([[1.,0.],[0.,1.],[.5,.5],[.2,.8]])
    graph=csr_matrix(np.ones((4,4))-np.eye(4))
    _,_,profile=predict_with_score_refinement(embeddings,graph,np.array([0,1]),np.array([1,0]),.5)
    assert profile['bias_diffusion_iters']==profile['homogeneity_propagation_iters']==50
