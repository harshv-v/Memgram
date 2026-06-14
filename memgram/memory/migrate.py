"""Tiny migration runner — applies memory/migrations/*.sql in order.

Usage:
    DATABASE_URL=postgresql://... python -m memgram.memory.migrate
"""
import asyncio
import os
import pathlib

import asyncpg

MIGRATIONS_DIR = pathlib.Path(__file__).parent / "migrations"


async def migrate() -> None:
    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT NOW())"
        )
        for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
            done = await conn.fetchval("SELECT 1 FROM _migrations WHERE name = $1", sql_file.name)
            if done:
                print(f"skip  {sql_file.name}")
                continue
            sql = sql_file.read_text().replace(
                "{EMBED_DIMS}", os.environ.get("MEMGRAM_EMBED_DIMS", "1536")
            )
            await conn.execute(sql)
            await conn.execute("INSERT INTO _migrations (name) VALUES ($1)", sql_file.name)
            print(f"apply {sql_file.name}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
