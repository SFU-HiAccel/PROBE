"""Shared L/A/D oracle ledger: reserve before dispatch; errors are not negatives.

The hard budget unit is candidate evaluation attempts. A retry spends again.
HTTP request count and tokens are independent diagnostics. Cached successful
labels never spend twice. Callers must share this session across all stages.
"""
import os
import asyncio
from dataclasses import dataclass, field
from threading import Lock, RLock
from numbers import Real

def _binary(value):
    return isinstance(value, Real) and value in (0, 1)


class OracleLabelsUnavailable(RuntimeError):
    """An inference stage received unresolved labels rather than binary evidence."""


@dataclass
class OracleLedger:
    limit: int
    used: int = 0
    http_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failures: int = 0
    cache: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)
    _inflight: set = field(default_factory=set, repr=False)

    def __post_init__(self):
        if self.limit < 0 or int(self.limit) != self.limit:
            raise ValueError('oracle limit must be a nonnegative integer')

    def reserve(self, candidate_id, stage, *, retry=False):
        with self._lock:
            if candidate_id in self.cache or candidate_id in self._inflight or self.used >= self.limit:
                return False
            self._inflight.add(candidate_id)
            self.used += 1
            self.http_requests += 1  # one candidate per request in this transport
            self.events.append({'id': candidate_id, 'stage': stage, 'retry': retry,
                                'attempt': self.used})
            return True

    def record(self, candidate_id, label, usage=None):
        with self._lock:
            self._inflight.discard(candidate_id)
            usage = usage or {}
            self.prompt_tokens += int(usage.get('prompt_tokens', 0) or 0)
            self.completion_tokens += int(usage.get('completion_tokens', 0) or 0)
            if not _binary(label):
                self.failures += 1
            else:
                self.cache[candidate_id] = int(label)
            import os
            path = os.environ.get("PROBE_ORACLE_LOG")
            if path:
                import json
                from pathlib import Path
                journal = Path(path)
                journal.parent.mkdir(parents=True, exist_ok=True)
                event = next((e for e in reversed(self.events) if e['id'] == candidate_id), {})
                record = dict(event, label=int(label) if _binary(label) else None,
                              response=usage.get("_response"),
                              usage={k: v for k, v in usage.items() if not k.startswith("_")})
                with journal.open("a") as handle:
                    handle.write(json.dumps(record, default=str) + "\n")

    def snapshot(self):
        with self._lock:
            return {'limit': self.limit, 'candidate_attempts': self.used,
                    'http_requests': self.http_requests, 'unique_labels': len(self.cache),
                    'failures': self.failures, 'prompt_tokens': self.prompt_tokens,
                    'completion_tokens': self.completion_tokens,
                    'remaining': self.limit-self.used}


class BudgetedOracle:
    """Async request(id) -> (binary label or None, usage dict).

    request must disable SDK/HTTP automatic retries. All retries are performed
    here, each reserving one unit. Batches deduplicate IDs before creating tasks;
    an async lock prevents concurrent batches from repurchasing the same ID.
    A failed or unaffordable ID returns None, never a negative label.
    """
    def __init__(self, ledger, request, *, concurrency=32, max_attempts=1):
        if concurrency < 1 or max_attempts < 1:
            raise ValueError('concurrency and max_attempts must be positive')
        self.ledger = ledger
        self.request = request
        self.concurrency = concurrency
        self.max_attempts = max_attempts
        self._batch_lock = asyncio.Lock()

    async def label(self, ids, *, stage):
        ids = list(ids)
        async with self._batch_lock:
            sem = asyncio.Semaphore(self.concurrency)
            async def one(key):
                if key in self.ledger.cache:
                    return
                async with sem:
                    for attempt in range(self.max_attempts):
                        if not self.ledger.reserve(key, stage, retry=attempt > 0):
                            return
                        try:
                            value, usage = await self.request(key)
                        except Exception:
                            value, usage = None, {}
                        self.ledger.record(key, value, usage)
                        if _binary(value):
                            return
            await asyncio.gather(*(one(key) for key in dict.fromkeys(ids)))
            return [self.ledger.cache.get(key) for key in ids]


def binary_labels_or_raise(values):
    """Keep unresolved labels out of calibration statistics and routing fits."""
    import numpy as np
    values = list(values)
    if any(not np.isscalar(y) or y not in (0, 1) for y in values):
        raise OracleLabelsUnavailable('oracle labels unresolved: budget exhausted or request failed')
    return np.asarray(values, dtype=np.int32)


class OpenAIOracleRequest:
    """One physical chat request with SDK retries disabled, for text or images.

    Use as an async context manager. Candidate contents are supplied by ID;
    labels/ground truth are not part of this interface. A non-binary response
    remains unresolved, including empty and truncated responses.
    """
    def __init__(self, contents, *, model, base_url, system_prompt,
                 max_tokens=16, temperature=0., extra_body=None, timeout=None):
        self.contents = contents
        self.model = model
        self.base_url = base_url.removesuffix('/chat/completions')
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_body = extra_body or {}
        # SEMBENCH_ORACLE_TIMEOUT (seconds, default 300): thinking-mode requests can queue and reason for
        # longer than 300 s; a timed-out request leaves its label unresolved and calibration abstains.
        self.timeout = float(timeout if timeout is not None else os.environ.get('SEMBENCH_ORACLE_TIMEOUT', 300))
        self.client = None

    async def __aenter__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(base_url=self.base_url, api_key='dummy',
                                  timeout=self.timeout, max_retries=0)
        return self

    async def __aexit__(self, *args):
        await self.client.close()
        self.client = None

    async def __call__(self, candidate_id):
        if self.client is None:
            raise RuntimeError('OpenAIOracleRequest must be entered before use')
        import label_cache  # opt-in (PROBE_LABEL_CACHE); budget is still charged by the caller
        cache_key = None
        if label_cache.enabled():
            cache_key = label_cache.key(self.model, self.system_prompt, self.contents[candidate_id],
                                        self.extra_body, self.max_tokens)
            hit = label_cache.get(cache_key)
            if hit is not None:
                try:  # a cached answer is still an oracle answer for the delegation counters
                    from proxy_model import _count_answer
                    _count_answer(hit, 'cache')
                except ImportError:
                    pass
                return hit, {'prompt_tokens':0, 'completion_tokens':0}
        value, usage = await self._request(candidate_id)
        if cache_key is not None:
            label_cache.put(cache_key, value)
        return value, usage

    async def _request(self, candidate_id):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{'role':'system', 'content':self.system_prompt},
                      {'role':'user', 'content':self.contents[candidate_id]}],
            max_tokens=self.max_tokens, temperature=self.temperature,
            top_p=1., presence_penalty=0., frequency_penalty=0.,
            extra_body=self.extra_body,
        )
        content = (response.choices[0].message.content or '')
        if '</think>' in content:   # thinking mode without a server reasoning parser: keep the final answer
            content = content.rsplit('</think>', 1)[1]
        # GLM-V models wrap the answer as <|begin_of_box|>True<|end_of_box|>.
        content = content.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')
        content = content.strip().lower()
        # First line only: a thinking model may add an explanation after "False".
        value = {'true':1, 'false':0}.get(content.split('\n', 1)[0].strip().rstrip('.'))
        if value is None and os.environ.get('SEMBENCH_UNPARSED_AS_FALSE') == '1':
            # Thinking-mode runs: an unreadable answer (e.g. reasoning cut off) counts as False, as on the
            # lenient path, instead of aborting calibration; it is still counted as unparsed below.
            try:
                from proxy_model import _count_answer
                _count_answer(None, content)
            except ImportError:
                pass
            return 0, {'prompt_tokens':getattr(response.usage, 'prompt_tokens', 0),
                       'completion_tokens':getattr(response.usage, 'completion_tokens', 0)}
        try:  # count it with the proxy_model answer counters (recorded per run by run.py)
            from proxy_model import _count_answer
            _count_answer(value, content)
        except ImportError:
            pass
        usage = response.usage
        return value, {'prompt_tokens':getattr(usage, 'prompt_tokens', 0),
                       'completion_tokens':getattr(usage, 'completion_tokens', 0),
                       '_response': {'id': getattr(response, 'id', None),
                                     'model': getattr(response, 'model', self.model),
                                     'content': response.choices[0].message.content}}


class PipelineOracle:
    """Synchronous pipeline adapter preserving original integer candidate IDs.

    label_ids is the preferred interface. The legacy text callback resolves
    object identity, or unique content as a fallback; ambiguous duplicate text
    must be addressed by ID instead of silently attaching a label to a row.
    """
    def __init__(self, texts, *, limit, request_factory, concurrency=32, max_attempts=1):
        import json
        self.texts = list(texts)
        self.ledger = OracleLedger(limit)
        self.request_factory = request_factory
        self.concurrency = concurrency
        self.max_attempts = max_attempts
        self.stage = 'L'
        self._identity = {}
        self._content = {}
        for i, text in enumerate(self.texts):
            self._identity.setdefault(id(text), []).append(i)
            self._content.setdefault(json.dumps(text, sort_keys=True), []).append(i)

    def label_ids(self, ids, *, stage=None):
        import numpy as np
        ids = [int(i) for i in ids]
        if any(i < 0 or i >= len(self.texts) for i in ids):
            raise ValueError('oracle candidate ID outside pool')
        async def execute():
            async with self.request_factory() as request:
                oracle = BudgetedOracle(self.ledger, request, concurrency=self.concurrency,
                                        max_attempts=self.max_attempts)
                return await oracle.label(ids, stage=stage or self.stage)
        values = asyncio.run(execute())
        return np.array([-1 if y is None else y for y in values], dtype=np.int32)

    def __call__(self, batch):
        import json
        ids = []
        for text in batch:
            matches = self._identity.get(id(text))
            if matches is None:
                matches = self._content.get(json.dumps(text, sort_keys=True), [])
            if len(matches) != 1:
                raise ValueError('ambiguous text callback: use label_ids for duplicate contents')
            ids.append(matches[0])
        return binary_labels_or_raise(self.label_ids(ids))


# Stage-aware accounting for budget_scope='preparation': only sampling (L) and calibration (A) count
# against the allowance; routing (D) and the fallback may use any remaining candidate.

ROUTING_STAGES = frozenset(('D', 'fallback', 'BARGAIN_route'))

class PreparationLedger(OracleLedger):
    def __init__(self, n, preparation_limit):
        # With max_attempts=1, N is the natural finite-pool limit, not a routing budget.
        super().__init__(limit=n)
        self._lock = RLock()
        self.preparation_limit = int(preparation_limit)
        self.preparation_used = 0
        self._attempted = set()
        if not 0 <= self.preparation_limit <= n:
            raise ValueError('Invalid preparation allowance')

    def reserve(self, candidate_id, stage, *, retry=False):
        with self._lock:
            if stage not in ROUTING_STAGES and stage not in ('L','A','training','calibration','proxy_calibration','BARGAIN'):
                raise ValueError('Unclassified oracle stage: '+str(stage))
            if candidate_id in self._attempted:
                return False
            preparation = stage not in ROUTING_STAGES
            if preparation and self.preparation_used >= self.preparation_limit:
                return False
            ok = super().reserve(candidate_id, stage, retry=retry)
            if ok:
                self._attempted.add(candidate_id)
                self.preparation_used += int(preparation)
            return ok

    def snapshot(self):
        with self._lock:
            return dict(super().snapshot(), budget_scope='preparation',
                        preparation_limit=self.preparation_limit,
                        preparation_attempts=self.preparation_used,
                        routing_attempts=self.used-self.preparation_used,
                        routing_budget=None)
