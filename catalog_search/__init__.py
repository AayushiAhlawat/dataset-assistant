"""Phase 2: catalog loading and Chroma semantic search helpers."""

from catalog_search.catalog import (
    COLLECTION_NAME,
    DEFAULT_CHROMA_PATH,
    EMBEDDING_MODEL,
    catalog_row_to_document,
    catalog_row_to_metadata,
)
from catalog_search.hydrate import format_hits_for_llm
from catalog_search.search import semantic_search
from catalog_search.store import get_chroma_client, get_collection

__all__ = [
    "COLLECTION_NAME",
    "DEFAULT_CHROMA_PATH",
    "EMBEDDING_MODEL",
    "catalog_row_to_document",
    "catalog_row_to_metadata",
    "format_hits_for_llm",
    "get_chroma_client",
    "get_collection",
    "semantic_search",
]
