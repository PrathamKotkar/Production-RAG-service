"""Build the 7,600-document corpus + labeled eval set from SQuAD v1.1 (CC BY-SA 4.0).

Each unique SQuAD paragraph is one document. Every eval question is labeled with the one
paragraph it was written from, so recall@k is measured against ground truth.

  python scripts/prepare_data.py --docs 7600 --eval 500 --out data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset


def doc_id(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=7600)
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    ds = load_dataset("rajpurkar/squad")
    val, train = ds["validation"], ds["train"]

    docs: dict[str, dict] = {}
    for row in val:  # validation contexts first: guarantees eval questions have their doc in the corpus
        docs.setdefault(doc_id(row["context"]), {"id": doc_id(row["context"]), "title": row["title"].replace("_", " "), "body": row["context"]})
    val_ids = set(docs)

    train_rows = list(train)
    rng.shuffle(train_rows)
    for row in train_rows:
        if len(docs) >= args.docs:
            break
        docs.setdefault(doc_id(row["context"]), {"id": doc_id(row["context"]), "title": row["title"].replace("_", " "), "body": row["context"]})

    # If validation alone exceeds --docs, keep all validation contexts (eval integrity > exact count)
    doc_list = list(docs.values())
    rng.shuffle(doc_list)

    val_rows = [r for r in val]
    rng.shuffle(val_rows)
    questions = [
        {
            "id": r["id"],
            "question": r["question"],
            "relevant_doc_ids": [doc_id(r["context"])],
            "gold_answers": list(dict.fromkeys(r["answers"]["text"])),
        }
        for r in val_rows[: args.eval]
    ]

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "docs.jsonl").write_text("\n".join(json.dumps(d) for d in doc_list) + "\n")
    (out / "eval_questions.jsonl").write_text("\n".join(json.dumps(q) for q in questions) + "\n")
    print(f"docs={len(doc_list)} (validation contexts={len(val_ids)})  eval_questions={len(questions)}")


if __name__ == "__main__":
    main()
