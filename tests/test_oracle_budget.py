import asyncio
from concurrent.futures import ThreadPoolExecutor
import pytest


def test_oracle_journal_records_evidence_without_inventing_failed_labels(tmp_path, monkeypatch):
    import json
    from oracle_budget import OracleLedger
    path = tmp_path / "oracle.jsonl"
    monkeypatch.setenv("PROBE_ORACLE_LOG", str(path))
    ledger = OracleLedger(2)
    assert ledger.reserve(7, "L")
    ledger.record(7, 1, {"prompt_tokens": 3, "_response": {"content": "True", "model": "test"}})
    assert ledger.reserve(8, "A")
    ledger.record(8, None)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["id"], r["stage"], r["label"]) for r in rows] == [(7, "L", 1), (8, "A", None)]
    assert rows[0]["response"]["content"] == "True"
    assert ledger.snapshot()["unique_labels"] == 1 and ledger.snapshot()["failures"] == 1
from oracle_budget import OracleLedger, BudgetedOracle, binary_labels_or_raise, OracleLabelsUnavailable


def test_parallel_reservations_cannot_exceed_cap():
    ledger = OracleLedger(13)
    with ThreadPoolExecutor(20) as executor:
        results = list(executor.map(lambda i: ledger.reserve(i, 'L'), range(200)))
    assert sum(results) == ledger.used == ledger.http_requests == 13


def test_shared_stages_cache_and_duplicate_ids():
    async def run():
        ledger = OracleLedger(3)
        calls = []
        async def request(i):
            calls.append(i)
            await asyncio.sleep(0)
            return i % 2, {'prompt_tokens': 5, 'completion_tokens': 1}
        oracle = BudgetedOracle(ledger, request)
        assert await oracle.label([1,1,2], stage='L') == [1,1,0]
        assert await oracle.label([2,3,4], stage='A') == [0,1,None]
        assert await oracle.label([1,4], stage='D') == [1,None]
        assert len(calls) == len(set(calls)) == 3
        assert ledger.snapshot()['prompt_tokens'] == 15
        assert ledger.snapshot()['remaining'] == 0
    asyncio.run(run())


def test_retry_spends_budget_and_failure_never_becomes_zero():
    async def run():
        ledger = OracleLedger(2)
        async def fail(i):
            raise TimeoutError()
        oracle = BudgetedOracle(ledger, fail, max_attempts=5)
        assert await oracle.label([7], stage='L') == [None]
        assert ledger.used == ledger.failures == 2
        assert ledger.events[1]['retry']
        assert not ledger.cache
    asyncio.run(run())


def test_concurrent_batches_reuse_successful_labels():
    async def run():
        ledger = OracleLedger(9)
        async def request(i):
            await asyncio.sleep(0)
            return 1, {}
        oracle = BudgetedOracle(ledger, request)
        a,b = await asyncio.gather(oracle.label([1,2],stage='L'),oracle.label([2,3],stage='A'))
        assert a == b == [1,1]
        assert ledger.used == 3
    asyncio.run(run())


def test_zero_budget_and_invalid_response():
    async def run():
        async def request(i):
            return 'false', {}
        empty = BudgetedOracle(OracleLedger(0), request)
        assert await empty.label([1], stage='L') == [None]
        invalid = BudgetedOracle(OracleLedger(1), request)
        assert await invalid.label([1], stage='A') == [None]
        assert invalid.ledger.failures == 1
    asyncio.run(run())
    for labels in ([None], [-1], [.4], ['true']):
        with pytest.raises(OracleLabelsUnavailable):
            binary_labels_or_raise(labels)
    assert binary_labels_or_raise([0,1]).tolist() == [0,1]


def test_calibration_rejects_missing_labels_before_integer_coercion():
    import numpy as np
    from learned_calibration import _label
    for response in ([None], [-1], [.9]):
        with pytest.raises(OracleLabelsUnavailable):
            _label(lambda _: response, np.array([0]))


def test_inflight_reservation_is_unique_and_failure_releases_for_retry():
    ledger = OracleLedger(2)
    assert ledger.reserve('a','L')
    assert not ledger.reserve('a','A')
    ledger.record('a',None)
    assert ledger.reserve('a','L',retry=True)
    ledger.record('a',1)
    assert not ledger.reserve('a','D')


def test_transport_does_not_interpret_explanations_as_labels():
    from types import SimpleNamespace as Obj
    from oracle_budget import OpenAIOracleRequest
    async def run():
        adapter = OpenAIOracleRequest({0:'a'},model='model',base_url='http://local/v1',system_prompt='binary')
        for content, expected in [('True',1),(' False ',0),('',None),('not true',None),('maybe false',None)]:
            async def create(**kwargs):
                assert kwargs['messages'][1]['content'] == 'a'
                return Obj(choices=[Obj(message=Obj(content=content))],usage=Obj(prompt_tokens=3,completion_tokens=1))
            adapter.client = Obj(chat=Obj(completions=Obj(create=create)))
            value, usage = await adapter(0)
            assert value == expected
            assert usage['prompt_tokens'] == 3
    asyncio.run(run())
