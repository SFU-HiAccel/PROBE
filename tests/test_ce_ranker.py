import numpy as np
import pytest
from ce_ranker import CEConfig, row_contents, score_filter, score_join


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, left, right, predicate, batch):
        self.calls.append((left, right, predicate, batch))
        return np.arange(len(left), dtype=np.float32), None


@pytest.fixture
def runtime(tmp_path):
    model = tmp_path / ('a' * 40)
    model.mkdir()
    (model / 'tokenizer_config.json').write_text('{"version":1}')
    return dict(model_path=model, cache_dir=tmp_path / 'cache', encoder=FakeEncoder(),
                config=CEConfig(device='cpu'))


def test_filter_consumes_predicate_selected_fields_and_cache(runtime):
    rows = [{'text': 'red', 'gt': 1}, {'text': 'blue', 'gt': 0}]
    args = (rows, ['text'], [], 'Is red?', ['a','b'])
    scores, first = score_filter(*args, **runtime)
    _, second = score_filter(*args, **runtime)
    assert not first['cache_hit'] and second['cache_hit']
    assert len(runtime['encoder'].calls) == 1
    left, right, predicate, _ = runtime['encoder'].calls[0]
    assert predicate == 'Is red?'
    assert left[0]['text'] == predicate
    assert right[0]['text'] == 'text: red'
    rows[0]['gt'] = 0  # unselected offline labels cannot affect the cache or model input
    _, unchanged = score_filter(*args, **runtime)
    assert unchanged['fingerprint'] == first['fingerprint']
    rows[0]['text'] = 'green'
    _, changed = score_filter(*args, **runtime)
    assert changed['fingerprint'] != first['fingerprint']
    assert not changed['cache_hit']


def test_predicate_id_processor_and_truncation_invalidate(runtime):
    def run(predicate='p', ids=('a',), **extra):
        kw = dict(runtime); kw.update(extra)
        return score_filter([{'t':'abc'}], ['t'], [], predicate, ids, **kw)[1]['fingerprint']
    initial = run()
    assert run('other') != initial
    assert run(ids=('b',)) != initial
    assert run(config=CEConfig(device='cpu', max_text_chars=1)) != initial
    (runtime['model_path'] / 'tokenizer_config.json').write_text('{"version":2}')
    assert run() != initial


def test_join_alignment_predicate_and_image_bytes(runtime, tmp_path):
    image = tmp_path / 'sample.bin'
    image.write_bytes(b'first')  # fake backend; image decoding is separately required
    left = [{'name':'a', 'pic':str(image)}]
    right = [{'name':'b'}, {'name':'c'}]
    args = (left, right, [(0,1),(0,0)], ['name','pic'], ['name'], ['pic'], 'Same brand?', ['ac','ab'])
    _, first = score_join(*args, **runtime)
    actual_left, actual_right, pred, _ = runtime['encoder'].calls[-1]
    assert pred == 'Same brand?'
    assert actual_right[0]['text'] == 'name: c'
    assert actual_left[0]['image'] == str(image)
    image.write_bytes(b'second')
    _, second = score_join(*args, **runtime)
    assert second['fingerprint'] != first['fingerprint']
    with pytest.raises(ValueError):
        score_join(left,right,[(-1,0)],['name'],['name'],[], 'p',['bad'],**runtime)


def test_reject_silent_multi_image_loss():
    with pytest.raises(ValueError, match='one image'):
        row_contents([{'a':'a.png','b':'b.png'}], ['a','b'], ['a','b'])


def test_corrupt_cache_rejected(runtime):
    args = ([{'t':'x'}], ['t'], [], 'p', ['a'])
    _, info = score_filter(*args, **runtime)
    path = runtime['cache_dir'] / (info['fingerprint'] + '.npz')
    np.savez(path, scores=np.array([np.nan]), fingerprint=info['fingerprint'])
    with pytest.raises(ValueError, match='invalid CE cache'):
        score_filter(*args, **runtime)


def test_image_cache_cannot_bypass_strict_global_policy(runtime, tmp_path, monkeypatch):
    from PIL import ImageFile
    image = tmp_path / 'image.bin'
    image.write_bytes(b'fake backend input')
    args = ([{'pic': str(image)}], ['pic'], ['pic'], 'Has zebra?', ['id'])
    monkeypatch.setattr(ImageFile, 'LOAD_TRUNCATED_IMAGES', False)
    _, info = score_filter(*args, **runtime)
    assert not info['cache_hit']
    monkeypatch.setattr(ImageFile, 'LOAD_TRUNCATED_IMAGES', True)
    with pytest.raises(ValueError, match='strict image decoding'):
        score_filter(*args, **runtime)


def test_backend_rejects_truncated_jpeg_and_permissive_global(tmp_path, monkeypatch):
    from PIL import Image, ImageFile
    from ce_ranker_backend import _QwenVL
    path = tmp_path / 'truncated.jpg'
    Image.new('RGB', (64,64), (12,34,56)).save(path)
    path.write_bytes(path.read_bytes()[:-2])
    backend = _QwenVL.__new__(_QwenVL)
    backend.max_pixels = 262144
    backend._images = {}
    monkeypatch.setattr(ImageFile, 'LOAD_TRUNCATED_IMAGES', False)
    with pytest.raises(OSError, match='truncated'):
        backend._image(str(path))
    monkeypatch.setattr(ImageFile, 'LOAD_TRUNCATED_IMAGES', True)
    with pytest.raises(ValueError, match='strict image decoding'):
        backend._image(str(path))
    assert not backend._images


class HiddenEncoder:
    """Returns a (n, 3) hidden state alongside the scores."""
    def __init__(self):
        self.calls = 0

    def encode(self, left, right, predicate, batch):
        self.calls += 1
        n = len(left)
        return np.arange(n, dtype=np.float32), np.arange(3 * n, dtype=np.float32).reshape(n, 3)


def test_hidden_states_are_returned_and_cached(runtime):
    runtime['encoder'] = HiddenEncoder()
    rows = [{'text': 'red'}, {'text': 'blue'}]
    args = (rows, ['text'], [], 'Is red?', ['a', 'b'])
    scores, info, hidden = score_filter(*args, return_hidden=True, **runtime)
    assert hidden.shape == (2, 3) and not info['cache_hit']
    s2, info2, h2 = score_filter(*args, return_hidden=True, **runtime)
    assert info2['cache_hit'] and runtime['encoder'].calls == 1
    np.testing.assert_array_equal(h2, hidden)
    np.testing.assert_array_equal(s2, scores)


def test_score_only_cache_is_reencoded_once_for_hidden(runtime):
    rows = [{'text': 'red'}, {'text': 'blue'}]
    args = (rows, ['text'], [], 'Is red?', ['a', 'b'])
    score_filter(*args, **runtime)                     # FakeEncoder: scores only
    runtime['encoder'] = HiddenEncoder()
    _, info, hidden = score_filter(*args, return_hidden=True, **runtime)
    assert not info['cache_hit'] and hidden.shape == (2, 3)
    _, info, _ = score_filter(*args, return_hidden=True, **runtime)
    assert info['cache_hit'] and runtime['encoder'].calls == 1


def test_hidden_request_without_hidden_encoder_fails(runtime):
    with pytest.raises(ValueError, match='no hidden'):
        score_filter([{'text': 'red'}], ['text'], [], 'Is red?', ['a'], return_hidden=True, **runtime)


def test_jina_model_is_accepted_and_unknown_rejected():
    assert CEConfig(model='jinaai/jina-reranker-m0').model == 'jinaai/jina-reranker-m0'
    with pytest.raises(ValueError):
        CEConfig(model='cross-encoder/ms-marco-MiniLM-L-6-v2')
