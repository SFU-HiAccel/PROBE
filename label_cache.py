"""Opt-in persistent oracle label cache shared across processes (sampling/encoder ablations).

Enabled only when PROBE_LABEL_CACHE=<sqlite path> is set; otherwise every function is a no-op and
behaviour is unchanged. The key is (model, system prompt, row content, labeller extra body, max_tokens),
so a cached label is the answer the same oracle gave to the same request. Budget accounting is NOT
affected: callers still charge every row they ask for; a hit only avoids the HTTP request. Use it
when an experiment measures quality only (latency of cached runs is meaningless).
"""
import hashlib
import json
import os
import sqlite3
import threading

_LOCK = threading.RLock()
_CONN = {}
STATS = {"hits": 0, "misses": 0}


def _path():
    return os.environ.get("PROBE_LABEL_CACHE") or None


def enabled() -> bool:
    return _path() is not None


def _conn():
    path = _path()
    with _LOCK:
        conn = _CONN.get(path)
        if conn is None:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            conn = sqlite3.connect(path, timeout=120, check_same_thread=False, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS labels (k TEXT PRIMARY KEY, y INTEGER NOT NULL)")
            _CONN[path] = conn
        return conn


def key(model, system_prompt, content, extra=None, max_tokens=None) -> str:
    blob = json.dumps([model, system_prompt, content, extra or {}, max_tokens],
                      sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get(k):
    if not enabled():
        return None
    with _LOCK:
        row = _conn().execute("SELECT y FROM labels WHERE k=?", (k,)).fetchone()
    STATS["hits" if row else "misses"] += 1
    return None if row is None else int(row[0])


def put(k, y) -> None:
    if not enabled() or y not in (0, 1):
        return
    with _LOCK:
        _conn().execute("INSERT OR IGNORE INTO labels (k, y) VALUES (?, ?)", (k, int(y)))
