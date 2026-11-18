"""Bulk-load docs.jsonl through the public API and wait for the workers to finish.

  python scripts/ingest_bulk.py --file data/docs.jsonl --api http://localhost:8000 --batch 200
"""
from __future__ import annotations

import argparse
import json
import time

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/docs.jsonl")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()

    docs = [json.loads(l) for l in open(args.file) if l.strip()]
    print(f"submitting {len(docs)} docs in batches of {args.batch}")
    t0 = time.time()
    jobs = []
    with httpx.Client(base_url=args.api, timeout=60) as cli:
        for i in range(0, len(docs), args.batch):
            r = cli.post("/v1/documents", json={"documents": docs[i : i + args.batch]})
            r.raise_for_status()
            jobs.append(r.json()["job_id"])

        pending = set(jobs)
        while pending:
            for j in list(pending):
                st = cli.get(f"/v1/jobs/{j}").json()
                if st["status"] == "failed":
                    raise SystemExit(f"job {j} failed: {st['error']}")
                if st["status"] == "done":
                    pending.discard(j)
            print(f"  {len(jobs) - len(pending)}/{len(jobs)} jobs done ({time.time() - t0:.0f}s)")
            if pending:
                time.sleep(3)
        print(json.dumps(cli.get("/metrics").json(), indent=2))


if __name__ == "__main__":
    main()
