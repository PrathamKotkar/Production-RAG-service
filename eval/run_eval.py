"""Evaluate retrieval modes (and optionally answers) on a labeled question set.

  python -m eval.run_eval --questions data/eval_questions.jsonl \
      --modes vector,fts,hybrid,hybrid_rerank --ks 1,5,10 --answers 100 --out reports

Question format (JSONL): {"id","question","relevant_doc_ids":[...],"gold_answers":[...]}
Retrieval is chunk-level; metrics are computed on the distinct, rank-ordered doc ids.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from app.bootstrap import build_components

from .faithfulness import FaithfulnessScorer
from .metrics import answer_contains_gold, hit_at_k, mrr, ndcg_at_k, recall_at_k


def load_questions(path: str, limit: int | None) -> list[dict]:
    qs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return qs[:limit] if limit else qs


async def eval_retrieval(c, questions, mode, ks, concurrency=8):
    kmax = max(ks)
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = []
    rows = []

    async def one(q):
        async with sem:
            t0 = time.perf_counter()
            hits = await c.retriever.search(q["question"], k=kmax * 3, mode=mode)
            lat.append((time.perf_counter() - t0) * 1000)
            ranked = [h.doc_id for h in hits]
            rel = set(q["relevant_doc_ids"])
            rows.append({
                **{f"recall@{k}": recall_at_k(ranked, rel, k) for k in ks},
                **{f"hit@{k}": hit_at_k(ranked, rel, k) for k in ks},
                **{f"ndcg@{k}": ndcg_at_k(ranked, rel, k) for k in ks},
                "mrr@10": mrr(ranked, rel, 10),
            })

    await asyncio.gather(*[one(q) for q in questions])
    summary = {m: round(statistics.fmean(r[m] for r in rows), 4) for m in rows[0]}
    summary["latency_p50_ms"] = round(statistics.median(lat), 1)
    summary["latency_p95_ms"] = round(sorted(lat)[int(0.95 * len(lat)) - 1], 1)
    return summary


async def eval_answers(c, questions, mode, k, scorer):
    faith, correct = [], []
    for q in questions:
        hits = await c.retriever.search(q["question"], k=k, mode=mode)
        ctxs = [h.text for h in hits]
        ans = await c.generator.answer(q["question"], ctxs)
        faith.append(await scorer.score(ans, ctxs))
        if q.get("gold_answers"):
            correct.append(answer_contains_gold(ans, q["gold_answers"]))
    return {
        "n": len(questions),
        "faithfulness": round(statistics.fmean(faith), 4),
        "faithfulness_method": scorer.method,
        "answer_contains_gold": round(statistics.fmean(correct), 4) if correct else None,
        "generator": c.generator.backend,
    }


def to_markdown(retrieval: dict, answers: dict, ks: list[int]) -> str:
    cols = [f"recall@{k}" for k in ks] + ["hit@5", "ndcg@10", "mrr@10", "latency_p50_ms", "latency_p95_ms"]
    cols = [c for c in cols if all(c in v for v in retrieval.values())]
    out = ["## Retrieval", "", "| mode | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for mode, v in retrieval.items():
        out.append(f"| {mode} | " + " | ".join(str(v[c]) for c in cols) + " |")
    if answers:
        out += ["", "## Answers", "", "| mode | faithfulness | answer_contains_gold | method | generator |", "|---|---|---|---|---|"]
        for mode, v in answers.items():
            out.append(f"| {mode} | {v['faithfulness']} | {v['answer_contains_gold']} | {v['faithfulness_method']} | {v['generator']} |")
    return "\n".join(out) + "\n"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/eval_questions.jsonl")
    ap.add_argument("--modes", default="vector,fts,hybrid,hybrid_rerank")
    ap.add_argument("--ks", default="1,5,10")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--answers", type=int, default=0, help="score answers on first N questions (0=skip)")
    ap.add_argument("--answer-k", type=int, default=5)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    modes = args.modes.split(",")
    questions = load_questions(args.questions, args.limit)
    c = await build_components()
    await c.retriever.search("warmup", k=1, mode="hybrid_rerank")  # exclude model load from latency

    retrieval, answers = {}, {}
    scorer = FaithfulnessScorer(c.settings)
    for mode in modes:
        print(f"[eval] retrieval mode={mode} n={len(questions)}")
        retrieval[mode] = await eval_retrieval(c, questions, mode, ks)
        if args.answers:
            print(f"[eval] answers mode={mode} n={args.answers}")
            answers[mode] = await eval_answers(c, questions[: args.answers], mode, args.answer_k, scorer)

    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / "eval.json").write_text(json.dumps({"retrieval": retrieval, "answers": answers}, indent=2))
    md = to_markdown(retrieval, answers, ks)
    (Path(args.out) / "eval.md").write_text(md)
    print("\n" + md)
    await c.close()


if __name__ == "__main__":
    asyncio.run(main())
