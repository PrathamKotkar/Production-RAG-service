"""Closed-loop load test with a realistic (skewed) query mix.

Real traffic repeats popular questions, which is what makes a response cache pay off. Queries
are drawn from the eval set with a Zipf-like distribution (--skew controls how heavy the head
is). Cache hit rate and p95 are MEASURED from X-Cache headers and client-side timings.

  python scripts/loadtest.py --users 50 --duration 60 --skew 1.1 --endpoint /v1/search
  python scripts/loadtest.py --users 100 --duration 60 --endpoint /v1/ask
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from collections import Counter

import httpx


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


async def user(cli, endpoint, queries, weights, deadline, lat, states, errors, rng, mode, k):
    while time.perf_counter() < deadline:
        q = rng.choices(queries, weights)[0]
        t0 = time.perf_counter()
        try:
            r = await cli.post(endpoint, json={"query": q, "k": k, "mode": mode})
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                lat.append(ms)
                states[r.headers.get("X-Cache", "NONE")] += 1
            else:
                errors[r.status_code] += 1
        except httpx.HTTPError as e:
            errors[type(e).__name__] += 1


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--endpoint", default="/v1/search")
    ap.add_argument("--questions", default="data/eval_questions.jsonl")
    ap.add_argument("--users", type=int, default=50)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--skew", type=float, default=1.1, help="Zipf exponent; 0 = uniform")
    ap.add_argument("--mode", default="hybrid_rerank")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=5, help="seconds excluded from stats")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    queries = [json.loads(l)["question"] for l in open(args.questions) if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(queries)
    weights = [1.0 / (i + 1) ** args.skew for i in range(len(queries))]

    limits = httpx.Limits(max_connections=args.users * 2, max_keepalive_connections=args.users * 2)
    async with httpx.AsyncClient(base_url=args.api, timeout=30, limits=limits) as cli:
        await cli.post("/admin/cache/reset-stats")
        # Warmup (not measured)
        w_lat, w_states, w_err = [], Counter(), Counter()
        wd = time.perf_counter() + args.warmup
        await asyncio.gather(*[
            user(cli, args.endpoint, queries, weights, wd, w_lat, w_states, w_err, random.Random(i), args.mode, args.k)
            for i in range(min(args.users, 10))
        ])
        await cli.post("/admin/cache/reset-stats")

        lat, states, errors = [], Counter(), Counter()
        deadline = time.perf_counter() + args.duration
        t0 = time.perf_counter()
        await asyncio.gather(*[
            user(cli, args.endpoint, queries, weights, deadline, lat, states, errors, random.Random(1000 + i), args.mode, args.k)
            for i in range(args.users)
        ])
        elapsed = time.perf_counter() - t0
        server = (await cli.get("/metrics")).json()

    total = sum(states.values())
    hits = states["HIT"]
    report = {
        "endpoint": args.endpoint, "users": args.users, "duration_s": round(elapsed, 1),
        "requests": total, "rps": round(total / elapsed, 1), "errors": dict(errors),
        "latency_ms": {"p50": round(pct(lat, .5), 1), "p95": round(pct(lat, .95), 1), "p99": round(pct(lat, .99), 1),
                       "mean": round(statistics.fmean(lat), 1) if lat else None},
        "cache_hit_rate_client": round(hits / total, 4) if total else 0,
        "cache_states": dict(states),
        "cache_hit_rate_server": server["cache"]["hit_rate"],
        "skew": args.skew,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
