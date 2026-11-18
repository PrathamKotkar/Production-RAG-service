CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT '',
    body       TEXT NOT NULL,
    metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
    status     TEXT NOT NULL DEFAULT 'pending',   -- pending | indexed
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Embedding dimension (384) must match EMBED_DIM / the embedding model.
CREATE TABLE IF NOT EXISTS chunks (
    id        BIGSERIAL PRIMARY KEY,
    doc_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx       INT  NOT NULL,
    text      TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (doc_id, idx)
);

-- ANN index for cosine distance
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- Full-text index
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id         TEXT PRIMARY KEY,
    total      INT  NOT NULL,
    done       INT  NOT NULL DEFAULT 0,
    status     TEXT NOT NULL DEFAULT 'queued',    -- queued | running | done | failed
    error      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
