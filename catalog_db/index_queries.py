"""PostgreSQL reads for Phase 2 vector indexing and LLM hydration."""

from __future__ import annotations

from typing import Any, Iterator

import psycopg

INDEX_ROW_QUERY = """
SELECT
    repo_id AS id,
    title,
    description,
    source,
    tags::text AS tags,
    url,
    license,
    downloads,
    likes,
    gated
FROM datasets
WHERE private IS NOT TRUE
  AND (disabled IS NOT TRUE OR disabled IS NULL)
  AND (%(require_description)s IS FALSE OR LENGTH(TRIM(description)) > 0)
  AND (%(min_downloads)s::bigint IS NULL OR COALESCE(downloads, 0) >= %(min_downloads)s::bigint)
ORDER BY downloads DESC NULLS LAST, repo_id
"""

HYDRATE_QUERY = """
SELECT
    repo_id,
    title,
    description,
    url,
    license,
    source,
    downloads,
    likes,
    gated,
    tags::text AS tags
FROM datasets
WHERE repo_id = ANY(%(repo_ids)s)
"""


def iter_index_rows(
    conn: psycopg.Connection[Any],
    *,
    require_description: bool = True,
    min_downloads: int | None = None,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield catalog dicts for embedding (repo_id exposed as id)."""
    sql = INDEX_ROW_QUERY
    if limit is not None and limit > 0:
        sql += "\nLIMIT %(limit)s"
    params: dict[str, Any] = {
        "require_description": require_description,
        "min_downloads": min_downloads,
        "limit": limit,
    }
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql, params)
        for row in cur:
            yield dict(row)


def count_index_rows(
    conn: psycopg.Connection[Any],
    *,
    require_description: bool = True,
    min_downloads: int | None = None,
) -> int:
    """Count rows that would be included in a vector index build."""
    cur = conn.execute(
        """
        SELECT COUNT(*)
        FROM datasets
        WHERE private IS NOT TRUE
          AND (disabled IS NOT TRUE OR disabled IS NULL)
          AND (%s IS FALSE OR LENGTH(TRIM(description)) > 0)
          AND (%s::bigint IS NULL OR COALESCE(downloads, 0) >= %s::bigint)
        """,
        (require_description, min_downloads, min_downloads),
    )
    return int(cur.fetchone()[0])


def fetch_datasets_by_repo_ids(
    conn: psycopg.Connection[Any],
    repo_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Load full metadata for retrieval hydration keyed by repo_id."""
    if not repo_ids:
        return {}
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(HYDRATE_QUERY, {"repo_ids": repo_ids})
        rows = cur.fetchall()
    return {str(row["repo_id"]): dict(row) for row in rows}
