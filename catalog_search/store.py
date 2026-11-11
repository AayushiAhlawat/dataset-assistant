"""Chroma persistent client and collection with sentence-transformers embeddings."""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Must be set before chromadb loads (otherwise telemetry stays enabled).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from catalog_search.catalog import COLLECTION_NAME, DEFAULT_CHROMA_PATH, EMBEDDING_MODEL

# Suppress broken posthog telemetry errors on some chromadb/posthog version pairs.
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """Create the Chroma embedding function backed by ``all-MiniLM-L6-v2``."""
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def get_chroma_client(persist_path: Path = DEFAULT_CHROMA_PATH) -> chromadb.ClientAPI:
    """Open a persistent Chroma client with anonymized telemetry disabled."""
    persist_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(anonymized_telemetry=False)
    settings.persist_directory = str(persist_path)
    settings.is_persistent = True
    return chromadb.PersistentClient(path=str(persist_path), settings=settings)


def get_collection(
    client: chromadb.ClientAPI,
    *,
    create: bool = False,
) -> Collection:
    """Get or create the ``hf_datasets`` collection with cosine space and MiniLM embeddings."""
    embedding_function = get_embedding_function()
    if create:
        return client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )
