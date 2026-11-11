"""SQL operations for catalog tables."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

UPSERT_DATASET = """
INSERT INTO datasets (
    repo_id, title, description, author, url, license, source,
    downloads, downloads_all_time, likes, trending_score,
    private, gated, disabled, sha, paperswithcode_id,
    created_at, last_modified, tags, card_json, siblings_json, raw_hub_json,
    hub_synced_at
) VALUES (
    %(repo_id)s, %(title)s, %(description)s, %(author)s, %(url)s, %(license)s, %(source)s,
    %(downloads)s, %(downloads_all_time)s, %(likes)s, %(trending_score)s,
    %(private)s, %(gated)s, %(disabled)s, %(sha)s, %(paperswithcode_id)s,
    %(created_at)s, %(last_modified)s, %(tags)s::jsonb, %(card_json)s::jsonb,
    %(siblings_json)s::jsonb, %(raw_hub_json)s::jsonb, NOW()
)
ON CONFLICT (repo_id) DO UPDATE SET
    title = EXCLUDED.title,
    description = EXCLUDED.description,
    author = EXCLUDED.author,
    url = EXCLUDED.url,
    license = EXCLUDED.license,
    source = EXCLUDED.source,
    downloads = EXCLUDED.downloads,
    downloads_all_time = EXCLUDED.downloads_all_time,
    likes = EXCLUDED.likes,
    trending_score = EXCLUDED.trending_score,
    private = EXCLUDED.private,
    gated = EXCLUDED.gated,
    disabled = EXCLUDED.disabled,
    sha = EXCLUDED.sha,
    paperswithcode_id = EXCLUDED.paperswithcode_id,
    created_at = EXCLUDED.created_at,
    last_modified = EXCLUDED.last_modified,
    tags = EXCLUDED.tags,
    card_json = EXCLUDED.card_json,
    siblings_json = EXCLUDED.siblings_json,
    raw_hub_json = EXCLUDED.raw_hub_json,
    hub_synced_at = NOW();
"""


def apply_schema(conn: psycopg.Connection[Any]) -> None:
    """Create tables and indexes if they do not exist."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.execute(sql)
    conn.commit()


def upsert_dataset(conn: psycopg.Connection[Any], row: dict[str, Any]) -> None:
    """Insert or update one dataset row."""
    conn.execute(UPSERT_DATASET, row)


def start_sync_run(conn: psycopg.Connection[Any], mode: str) -> int:
    """Record the start of a sync job and return its id."""
    cur = conn.execute(
        """
        INSERT INTO sync_runs (mode, status)
        VALUES (%s, 'running')
        RETURNING id
        """,
        (mode,),
    )
    run_id = cur.fetchone()[0]
    conn.commit()
    return int(run_id)


def finish_sync_run(
    conn: psycopg.Connection[Any],
    run_id: int,
    *,
    rows_upserted: int,
    status: str,
    error_message: str | None = None,
) -> None:
    """Mark a sync job finished."""
    conn.execute(
        """
        UPDATE sync_runs
        SET finished_at = NOW(),
            rows_upserted = %s,
            status = %s,
            error_message = %s
        WHERE id = %s
        """,
        (rows_upserted, status, error_message, run_id),
    )
    conn.commit()


def count_datasets(conn: psycopg.Connection[Any]) -> int:
    """Return total rows in datasets table."""
    cur = conn.execute("SELECT COUNT(*) FROM datasets")
    return int(cur.fetchone()[0])


def catalog_stats(conn: psycopg.Connection[Any]) -> dict[str, int]:
    """Return basic counts for verification and monitoring."""
    cur = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE LENGTH(TRIM(description)) > 0) AS with_description,
            COUNT(*) FILTER (WHERE private IS TRUE) AS private_count
        FROM datasets
        """
    )
    row = cur.fetchone()
    return {
        "total": int(row[0]),
        "with_description": int(row[1]),
        "private_count": int(row[2]),
    }


def last_successful_sync_finished_at(
    conn: psycopg.Connection[Any],
) -> datetime | None:
    """Return finished_at of the latest successful sync (watermark for daily incremental)."""
    cur = conn.execute(
        """
        SELECT finished_at
        FROM sync_runs
        WHERE status = 'success' AND finished_at IS NOT NULL
        ORDER BY finished_at DESC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return row[0]


def last_sync_run(conn: psycopg.Connection[Any]) -> dict[str, object] | None:
    """Return the most recent sync_runs row, if any."""
    cur = conn.execute(
        """
        SELECT id, mode, rows_upserted, status, started_at, finished_at, error_message
        FROM sync_runs
        ORDER BY id DESC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "mode": row[1],
        "rows_upserted": row[2],
        "status": row[3],
        "started_at": row[4],
        "finished_at": row[5],
        "error_message": row[6],
    }
