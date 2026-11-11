#!/usr/bin/env python3
"""Embed catalog rows with sentence-transformers and store in Chroma."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb.errors
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog_db.connection import connect  # noqa: E402
from catalog_db.index_queries import count_index_rows, iter_index_rows  # noqa: E402
from catalog_search.catalog import (  # noqa: E402
    COLLECTION_NAME,
    DEFAULT_CHROMA_PATH,
    EMBEDDING_MODEL,
    catalog_row_to_document,
    catalog_row_to_metadata,
    chroma_id_from_repo,
)
from catalog_search.store import get_chroma_client, get_collection  # noqa: E402

BATCH_SIZE = 64


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the vector index build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=DEFAULT_CHROMA_PATH,
        help=f"Chroma persist directory (default: {DEFAULT_CHROMA_PATH})",
    )
    parser.add_argument(
        "--require-description",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows with empty description (default: true)",
    )
    parser.add_argument(
        "--min-downloads",
        type=int,
        default=None,
        help="Only index rows with at least this many downloads",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to index (for testing)",
    )
    return parser.parse_args()


def _index_batch(collection, ids, documents, metadatas, indexed: int, total: int) -> int:
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    indexed += len(ids)
    print(f"  Indexed {indexed}/{total}")
    return indexed


def build_from_postgres(
    collection,
    *,
    require_description: bool,
    min_downloads: int | None,
    limit: int | None,
) -> int:
    """Stream Postgres rows into Chroma. Returns total indexed."""
    with connect() as conn:
        total = count_index_rows(
            conn,
            require_description=require_description,
            min_downloads=min_downloads,
        )
        if limit is not None:
            total = min(total, limit)
        print(f"Postgres index candidates: {total}")

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        indexed = 0

        for row in iter_index_rows(
            conn,
            require_description=require_description,
            min_downloads=min_downloads,
            limit=limit,
        ):
            repo_id = str(row["id"])
            ids.append(chroma_id_from_repo(repo_id))
            documents.append(catalog_row_to_document(row))
            metadatas.append(catalog_row_to_metadata(row))

            if len(ids) >= BATCH_SIZE:
                indexed = _index_batch(collection, ids, documents, metadatas, indexed, total)
                ids, documents, metadatas = [], [], []

        if ids:
            indexed = _index_batch(collection, ids, documents, metadatas, indexed, total)
        return indexed


def main() -> int:
    """Rebuild the Chroma collection from PostgreSQL."""
    load_dotenv(ROOT / ".env")
    args = parse_args()
    client = get_chroma_client(args.chroma_path)

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Removed existing collection {COLLECTION_NAME!r}")
    except (chromadb.errors.NotFoundError, ValueError):
        pass

    collection = get_collection(client, create=True)
    count = build_from_postgres(
        collection,
        require_description=args.require_description,
        min_downloads=args.min_downloads,
        limit=args.limit,
    )

    print(
        f"Done: {count} datasets in Chroma @ {args.chroma_path} "
        f"(model={EMBEDDING_MODEL!r}, source=postgres)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
