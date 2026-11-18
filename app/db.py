from __future__ import annotations

from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from .config import Settings


async def _configure(conn):
    await register_vector_async(conn)
    await conn.commit()


async def create_pool(s: Settings) -> AsyncConnectionPool:
    pool = AsyncConnectionPool(
        s.database_url,
        min_size=s.db_pool_min,
        max_size=s.db_pool_max,
        configure=_configure,
        open=False,
    )
    await pool.open(wait=True, timeout=30)
    return pool
