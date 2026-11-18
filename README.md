# Production RAG Service with Evaluation Harness

FastAPI · PostgreSQL/pgvector · Redis · Docker

A retrieval-augmented generation API over ~7,600 documents that **measures its own quality**: hybrid
retrieval with reranking, async ingestion, response caching, and an eval harness for recall@k and
answer faithfulness.

## Architecture

```
                 POST /v1/documents                     POST /v1/search | /v1/ask
 client ───────────────┐                                         │
                       ▼                                         ▼
                 ┌──────────┐  enqueue   ┌────────────┐   ┌───────────────┐  hit   ┌───────┐
                 │ FastAPI  │───────────▶│ Redis list │   │ QueryService  │◀──────▶│ Redis │
                 │  (api)   │            │ BLMOVE ack │   │ cache + single│  miss  │ cache │
                 └──────────┘            └─────┬──────┘   │ flight        │        └───────┘
                       │                       ▼          └──────┬────────┘
                       │               ┌──────────────┐          ▼
                       │               │ worker(s)    │   ┌─────────────────────────────┐
                       │               │ chunk→embed  │   │ Retriever                   │
                       │               │ →index       │   │  vector (HNSW cosine) ─┐    │
                       │               │ bump version │   │  full-text (GIN tsvector)─┤RRF
                       │               └──────┬───────┘   │  cross-encoder rerank top-N │
                       ▼                      ▼           └──────────────┬──────────────┘
                 ┌─────────────────────────────────────────────┐        ▼
                 │ PostgreSQL + pgvector: documents, chunks,   │◀── LLM generation (Claude)
                 │ ingest_jobs                                 │    cited answer, or extractive fallback
                 └─────────────────────────────────────────────┘
```

**Design choices**

| Concern | Choice | Why |
|---|---|---|
| Vector search | pgvector HNSW, cosine, `ef_search=80` | one datastore, ANN latency without a separate vector DB |
| Lexical search | `tsvector` + GIN, natural-language question → OR-query with stopwords removed | `websearch_to_tsquery` ANDs terms and returns nothing for most questions |
| Fusion | Reciprocal Rank Fusion (k=60) | score-scale-free; no tuning of weights between cosine and `ts_rank` |
| Rerank | cross-encoder `ms-marco-MiniLM-L-6-v2` on top 20 fused candidates | biggest precision gain per ms; runs in a thread so the loop stays free |
| Ingestion | Redis list with `BLMOVE` → processing list, `LREM` on ack, DLQ on failure, requeue on worker start | at-least-once, crash-safe; re-ingest is idempotent (delete+insert per doc) |
| Cache | Redis, key = hash(normalized query, k, mode, **index version**), TTL 1h | ingest bumps the version → instant invalidation without key scans |
| Stampede | per-process single-flight | concurrent identical misses share one computation |
| Faithfulness | LLM judge over decomposed claims; lexical-support fallback offline | both reported with the method name so numbers aren't mixed up |

## Quickstart

```bash
make up        # db, redis, api, worker  (set ANTHROPIC_API_KEY in .env for real LLM answers/judging)
make data      # 7,600 SQuAD paragraphs + 500 labeled questions -> ./data
make ingest    # through the API; workers embed and index
make eval      # -> reports/eval.md, reports/eval.json
make load      # 50 users x 60s, skewed query mix
make test      # unit tests, no services needed
```

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"query":"What year was the Eiffel Tower completed?","k":5,"mode":"hybrid_rerank"}' | jq
```

API: `POST /v1/documents` (202 + job id) · `GET /v1/jobs/{id}` · `POST /v1/search` · `POST /v1/ask` ·
`GET /metrics` · `GET /healthz`. Responses carry `X-Cache: HIT|MISS|COALESCED` and `X-Response-Time-ms`.
`mode` ∈ `vector | fts | hybrid | hybrid_rerank`, which is how the ablation is run.

## Evaluation methodology

- **Corpus:** 7,600 unique SQuAD v1.1 paragraphs (one paragraph = one document). CC BY-SA 4.0.
- **Labels:** each of 500 held-out questions is labeled with the single paragraph it was written from.
- **Metrics:** recall@{1,5,10}, hit@k, nDCG@10, MRR@10 on distinct doc ids (chunk hits are collapsed to
  documents preserving rank). Answer metrics: faithfulness (fraction of answer claims supported by the
  retrieved context) and answer-contains-gold.
- **Baseline:** `vector` mode is the vector-only baseline. `hybrid` and `hybrid_rerank` are compared against it.

### Reproducing the headline numbers honestly

The résumé bullets (recall@5 64% → 83%, 167 ms p95, 59% cache hit rate) are **targets this repo lets you
measure; they are not outputs of this repo until you run it.** Use *your* measured numbers:

1. `make eval` → copy the `recall@5` row for `vector` vs `hybrid_rerank`. SQuAD questions are
   lexically close to their paragraph, so vector-only may already score high; if the lift is smaller than
   claimed, either report the real lift or evaluate on a harder set (e.g. BEIR SciFact/FiQA, or questions
   paraphrased by an LLM) by producing a JSONL in the same format.
2. `make load` → p95 and hit rate are measured, not assumed. **Hit rate depends on the query mix**: `--skew`
   (Zipf exponent) controls repetition. State the skew (and that it is a synthetic mix) when quoting the
   number. Check `cache_hit_rate_client` and `_server` agree. For p95 under 50–100 users also record
   hardware (CPU cores), `--workers`, and whether the endpoint was `/v1/search` or `/v1/ask`; `/v1/ask`
   with a remote LLM will be dominated by LLM latency on cache misses.
3. Re-run with `--users 100` for the upper end of the concurrency range.

## Tuning knobs (`.env`)

`CANDIDATE_K` (per-retriever depth), `RERANK_TOP_N` (latency vs. quality), `HNSW_EF_SEARCH` (recall vs.
latency), `CHUNK_WORDS/CHUNK_OVERLAP`, `CACHE_TTL_SECONDS`, uvicorn `--workers` in the Dockerfile.
Changing the embedding model requires updating `vector(384)` in `sql/init.sql` and re-ingesting.

## Known limitations / next steps

- `/metrics` latency percentiles are per-process; use the load tester's client-side numbers for truth.
- Single-flight is per process (not cross-worker). A Redis lock would make it global.
- `requeue_stuck` assumes one worker deployment restarts at a time; with many workers use a lease/visibility timeout.
- Faithfulness via LLM judge costs tokens and is itself imperfect; spot-check a sample by hand.
- Offline `EMBED_BACKEND=hash` is for CI only (lexical hashing, not semantic) — never report numbers from it.
