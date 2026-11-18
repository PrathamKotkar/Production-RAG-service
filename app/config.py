from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    redis_url: str = "redis://localhost:6379/0"

    embed_backend: str = "st"  # st | hash
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384
    embed_batch_size: int = 64

    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 20
    candidate_k: int = 40
    rrf_k: int = 60
    hnsw_ef_search: int = 80

    chunk_words: int = 180
    chunk_overlap: int = 30

    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600

    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-5-5"
    llm_max_tokens: int = 400

    db_pool_min: int = 4
    db_pool_max: int = 20
    ingest_consumers: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
