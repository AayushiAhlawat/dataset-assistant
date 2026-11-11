"""Build embeddable text + Chroma metadata from PostgreSQL catalog rows."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROMA_PATH = ROOT / "data" / "chroma"

COLLECTION_NAME = "hf_datasets"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DESCRIPTION_SNIPPET_LEN = 400
EMBED_DESCRIPTION_MAX_LEN = 3500
LLM_DESCRIPTION_MAX_LEN = 2800


def chroma_id_from_repo(repo_id: str) -> str:
    """Stable Chroma id (repo ids contain slashes)."""
    return repo_id.replace("/", "__")


def parse_tags(tags_cell: object) -> list[str]:
    """Parse tags from a JSON array string or list."""
    if tags_cell is None:
        return []
    if isinstance(tags_cell, float) and math.isnan(tags_cell):
        return []
    if isinstance(tags_cell, list):
        return [str(t) for t in tags_cell if str(t).strip()]
    raw = str(tags_cell).strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(t) for t in parsed if str(t).strip()]


def _repo_id(row: Mapping[str, Any]) -> str:
    if "id" in row and row["id"]:
        return str(row["id"])
    return str(row.get("repo_id") or "")


def catalog_row_to_document(row: Mapping[str, Any]) -> str:
    """Build the text blob embedded for one catalog row."""
    tags = parse_tags(row.get("tags"))
    parts = [
        f"Dataset: {_repo_id(row)}",
        f"Title: {row.get('title', '')}",
    ]
    description = str(row.get("description") or "").strip()
    if len(description) > EMBED_DESCRIPTION_MAX_LEN:
        description = description[:EMBED_DESCRIPTION_MAX_LEN] + "…"
    if description:
        parts.append(f"Description: {description}")
    if tags:
        parts.append(f"Tags: {', '.join(tags[:40])}")
    license_str = str(row.get("license") or "").strip()
    if license_str:
        parts.append(f"License: {license_str}")
    return "\n".join(parts)


def catalog_row_to_metadata(row: Mapping[str, Any]) -> dict[str, str | int]:
    """Build Chroma metadata for one catalog row (scalar fields only)."""
    description = str(row.get("description") or "").strip()
    snippet = description[:DESCRIPTION_SNIPPET_LEN]
    if len(description) > DESCRIPTION_SNIPPET_LEN:
        snippet += "…"
    tags = parse_tags(row.get("tags"))
    downloads = row.get("downloads")
    return {
        "repo_id": _repo_id(row),
        "title": str(row.get("title", "")),
        "url": str(row.get("url", "")),
        "source": str(row.get("source", "")),
        "license": str(row.get("license") or ""),
        "downloads": int(downloads or 0),
        "tags": json.dumps(tags),
        "description_snippet": snippet,
    }
