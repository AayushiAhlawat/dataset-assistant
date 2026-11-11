"""Load catalog rows from PostgreSQL for Phase 2 indexing."""

from __future__ import annotations

from typing import Any, Iterator

from catalog_db.connection import connect
from catalog_db.index_queries import count_index_rows, iter_index_rows


def load_catalog_from_postgres(
    *,
    require_description: bool = True,
    min_downloads: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load index candidates from Postgres (materialized list)."""
    with connect() as conn:
        return list(
            iter_index_rows(
                conn,
                require_description=require_description,
                min_downloads=min_downloads,
                limit=limit,
            )
        )


def iter_catalog_from_postgres(
    *,
    require_description: bool = True,
    min_downloads: int | None = None,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream index candidates from Postgres."""
    with connect() as conn:
        yield from iter_index_rows(
            conn,
            require_description=require_description,
            min_downloads=min_downloads,
            limit=limit,
        )


def postgres_index_count(
    *,
    require_description: bool = True,
    min_downloads: int | None = None,
) -> int:
    """Return how many datasets match index filters."""
    with connect() as conn:
        return count_index_rows(
            conn,
            require_description=require_description,
            min_downloads=min_downloads,
        )
