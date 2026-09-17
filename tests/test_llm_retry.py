"""Retry behavior for expert LLM calls (budget-F1 experiment robustness fix).

proxy_model.py used to turn any failed expert request into label 0 immediately.
Over ~1M concurrent requests a transient client-side error (e.g. an httpx timeout
with an empty message) is expected; a single one aborted a multi-hour run.
Transient failures must be retried; only persistent failures may fall back to
label 0, and those must still log "defaulting to 0" so the run watcher aborts.
"""
import asyncio
import logging

import pytest

import proxy_model as pm


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _ok(answer="True"):
    return _FakeResponse(200, {"choices": [{"message": {"content": answer}}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


class _FakeClient:
    """Replays a scripted sequence of responses or exceptions, one per POST."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def post(self, url, json=None):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    monkeypatch.setattr(pm, "LLM_RETRY_BASE_DELAY_S", 0.0, raising=False)


def _run_raw(client):
    return asyncio.run(pm._call_llm_single_raw_async(
        client, "http://x/v1/chat/completions", "m", "sys", "Is it a zebra?",
        max_tokens=16, temperature=0.0, semaphore=asyncio.Semaphore(4)))


def test_transient_exception_is_retried_and_succeeds(caplog):
    client = _FakeClient([TimeoutError(), _ok("True")])
    with caplog.at_level(logging.WARNING):
        label = _run_raw(client)
    assert label == 1
    assert client.calls == 2
    assert "defaulting to 0" not in caplog.text


def test_retryable_5xx_is_retried(caplog):
    client = _FakeClient([_FakeResponse(503, text="busy"), _ok("True")])
    with caplog.at_level(logging.WARNING):
        label = _run_raw(client)
    assert label == 1
    assert client.calls == 2
    assert "defaulting to 0" not in caplog.text


def test_persistent_failure_falls_back_to_zero_and_is_logged(caplog):
    attempts = pm.LLM_MAX_ATTEMPTS
    client = _FakeClient([TimeoutError()] * attempts)
    with caplog.at_level(logging.WARNING):
        label = _run_raw(client)
    assert label == 0
    assert client.calls == attempts
    assert "defaulting to 0" in caplog.text


def test_client_error_is_not_retried(caplog):
    client = _FakeClient([_FakeResponse(400, text="bad media path")])
    with caplog.at_level(logging.WARNING):
        label = _run_raw(client)
    assert label == 0
    assert client.calls == 1
    assert "defaulting to 0" in caplog.text


def test_retry_log_lines_do_not_trigger_the_run_watcher(caplog):
    # The watcher aborts on "LLM call failed|Raw LLM call HTTP|Raw async LLM call failed|defaulting to 0".
    import re
    watcher = re.compile(r"LLM call failed|Raw LLM call HTTP|Raw async LLM call failed|defaulting to 0")
    client = _FakeClient([TimeoutError(), _ok("False")])
    with caplog.at_level(logging.WARNING):
        label = _run_raw(client)
    assert label == 0  # the model answered False; this is a real label, not a fallback
    assert client.calls == 2
    assert not watcher.search(caplog.text)


def _ok_batch(answers):
    return _FakeResponse(200, {"choices": [{"index": i, "text": a} for i, a in enumerate(answers)],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


def _run_batch(client, texts):
    return asyncio.run(pm._label_chunk_via_completions_async(
        client, "http://x/v1/completions", "m", "sys", texts,
        max_tokens=16, temperature=0.0, semaphore=asyncio.Semaphore(4)))


def test_batched_completions_retries_transient_exception(caplog):
    client = _FakeClient([ConnectionError(), _ok_batch(["True", "False", "True"])])
    with caplog.at_level(logging.WARNING):
        labels = _run_batch(client, ["a", "b", "c"])
    assert labels == [1, 0, 1]
    assert client.calls == 2
    assert "defaulting to 0" not in caplog.text


def test_batched_completions_persistent_failure_is_logged(caplog):
    client = _FakeClient([_FakeResponse(502, text="bad gateway")] * pm.LLM_MAX_ATTEMPTS)
    with caplog.at_level(logging.WARNING):
        labels = _run_batch(client, ["a", "b"])
    assert labels == [0, 0]
    assert client.calls == pm.LLM_MAX_ATTEMPTS
    assert "defaulting to 0" in caplog.text
