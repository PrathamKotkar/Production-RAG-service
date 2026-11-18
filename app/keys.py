"""Cache-key construction (pure, importable without Redis)."""
from __future__ import annotations

import hashlib
import json
import re

_WS = re.compile(r"\s+")


def normalize_query(q: str) -> str:
    return _WS.sub(" ", q.strip().lower())


def make_key(kind: str, query: str, params: dict, version: str) -> str:
    payload = json.dumps(
        {"q": normalize_query(query), "p": params, "v": version}, sort_keys=True
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:32]
    return f"rag:{kind}:{digest}"
