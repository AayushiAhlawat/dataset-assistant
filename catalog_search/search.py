"""Semantic search over the Chroma index."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import chromadb.errors

from catalog_search.catalog import DEFAULT_CHROMA_PATH, EMBEDDING_MODEL
from catalog_search.store import get_chroma_client, get_collection


class SearchHit(TypedDict, total=False):
    rank: int
    chroma_id: str
    repo_id: str
    url: str
    title: str
    license: str
    downloads: int
    tags: str
    description_snippet: str
    distance: float | None
    similarity: float | None


def distance_to_similarity(distance: float | None) -> float | None:
    """Map Chroma cosine distance to a rough similarity score in [0, 1]."""
    if distance is None:
        return None
    return max(0.0, min(1.0, 1.0 - float(distance)))


def semantic_search(
    query: str,
    *,
    k: int = 5,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
) -> list[SearchHit]:
    """Return the top-k catalog datasets most similar to ``query`` (Chroma + MiniLM)."""
    if not query.strip():
        return []
    if k < 1:
        raise ValueError("k must be at least 1")

    client = get_chroma_client(chroma_path)
    try:
        collection = get_collection(client)
    except (chromadb.errors.NotFoundError, ValueError) as exc:
        raise FileNotFoundError(
            "Vector index not found. Run: python scripts/build_vector_index.py"
        ) from exc

    result = collection.query(query_texts=[query.strip()], n_results=k)
    ids = (result.get("ids") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[SearchHit] = []
    for idx, doc_id in enumerate(ids):
        meta: dict[str, Any] = metadatas[idx] if idx < len(metadatas) else {}
        dist = distances[idx] if idx < len(distances) else None
        hits.append(
            SearchHit(
                rank=idx + 1,
                chroma_id=doc_id,
                repo_id=str(meta.get("repo_id", "")),
                url=str(meta.get("url", "")),
                title=str(meta.get("title", "")),
                license=str(meta.get("license", "")),
                downloads=int(meta.get("downloads") or 0),
                tags=str(meta.get("tags", "[]")),
                description_snippet=str(meta.get("description_snippet", "")),
                distance=dist,
                similarity=distance_to_similarity(dist),
            )
        )
    return hits


def search_model_name() -> str:
    """Return the sentence-transformers model id used for embeddings."""
    return EMBEDDING_MODEL
