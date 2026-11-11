"""Enrich vector search hits with PostgreSQL metadata for the LLM."""

from __future__ import annotations

from typing import Any

from catalog_db.connection import connect
from catalog_db.index_queries import fetch_datasets_by_repo_ids
from catalog_search.catalog import LLM_DESCRIPTION_MAX_LEN, parse_tags
from catalog_search.search import SearchHit


def hydrate_hits(hits: list[SearchHit]) -> dict[str, dict[str, Any]]:
    """Load Postgres rows for retrieved repo_ids."""
    repo_ids = [str(h.get("repo_id", "")) for h in hits if h.get("repo_id")]
    if not repo_ids:
        return {}
    with connect() as conn:
        return fetch_datasets_by_repo_ids(conn, repo_ids)


def format_hits_for_llm(
    hits: list[SearchHit],
    details: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Serialize search hits as plain text for LLM context in the answer node."""
    if not hits:
        return "(No datasets retrieved.)"
    details = details or {}
    blocks: list[str] = []
    for hit in hits:
        repo_id = str(hit.get("repo_id", ""))
        row = details.get(repo_id, {})
        sim = hit.get("similarity")
        sim_note = f"similarity≈{sim:.3f}" if sim is not None else "similarity=n/a"

        title = str(row.get("title") or hit.get("title") or repo_id)
        url = str(row.get("url") or hit.get("url") or "")
        license_str = str(row.get("license") or hit.get("license") or "unknown")
        downloads = row.get("downloads", hit.get("downloads"))
        likes = row.get("likes")
        gated = row.get("gated")

        tags_raw = row.get("tags") if row else hit.get("tags")
        tags = parse_tags(tags_raw)
        tags_line = ", ".join(tags[:25]) if tags else "(none)"

        description = str(row.get("description") or hit.get("description_snippet") or "").strip()
        if len(description) > LLM_DESCRIPTION_MAX_LEN:
            description = description[:LLM_DESCRIPTION_MAX_LEN] + "…"
        description = description.replace("\n", " ") if description else "(no description)"

        extras: list[str] = []
        if downloads is not None:
            extras.append(f"downloads={downloads}")
        if likes is not None:
            extras.append(f"likes={likes}")
        if gated:
            extras.append(f"gated={gated}")
        stats = ", ".join(extras) if extras else "stats=n/a"

        blocks.append(
            "\n".join(
                [
                    f"- repo_id: {repo_id} ({sim_note})",
                    f"  title: {title}",
                    f"  url: {url}",
                    f"  license: {license_str or 'unknown'}",
                    f"  {stats}",
                    f"  tags: {tags_line}",
                    f"  description: {description}",
                ]
            )
        )
    return "\n\n".join(blocks)
