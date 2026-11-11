"""Checks for Phase 2 index readiness (used by Phase 3)."""

from __future__ import annotations

from pathlib import Path

import chromadb.errors

from catalog_search.catalog import COLLECTION_NAME, DEFAULT_CHROMA_PATH
from catalog_search.store import get_chroma_client, get_collection


def chroma_index_count(chroma_path: Path = DEFAULT_CHROMA_PATH) -> int:
    """Return the number of vectors in the hf_datasets collection."""
    client = get_chroma_client(chroma_path)
    collection = get_collection(client)
    return int(collection.count())


def chroma_index_ready(chroma_path: Path = DEFAULT_CHROMA_PATH) -> tuple[bool, str]:
    """Return (ok, message) for whether semantic search can run."""
    if not chroma_path.is_dir():
        return False, (
            f"Chroma directory not found: {chroma_path}. "
            "Run: python scripts/build_vector_index.py"
        )
    try:
        count = chroma_index_count(chroma_path)
    except (chromadb.errors.NotFoundError, ValueError):
        return False, (
            f"Chroma collection {COLLECTION_NAME!r} not found @ {chroma_path}. "
            "Run: python scripts/build_vector_index.py"
        )
    except Exception as exc:
        return False, f"Chroma check failed: {exc}"
    if count < 1:
        return False, (
            f"Chroma collection {COLLECTION_NAME!r} is empty. "
            "Run: python scripts/build_vector_index.py"
        )
    return True, f"Chroma index ready ({count:,} datasets)."
