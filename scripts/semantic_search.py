#!/usr/bin/env python3
"""Semantic search over the local Hugging Face dataset catalog (Chroma + MiniLM)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog_search.catalog import DEFAULT_CHROMA_PATH  # noqa: E402
from catalog_search.search import search_model_name, semantic_search  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for semantic search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural language search query")
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=DEFAULT_CHROMA_PATH,
        help=f"Chroma persist directory (default: {DEFAULT_CHROMA_PATH})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of human-readable lines",
    )
    return parser.parse_args()


def main() -> int:
    """Run a single semantic search query and print results."""
    args = parse_args()
    if args.k < 1:
        print("-k must be at least 1", file=sys.stderr)
        return 1

    try:
        hits = semantic_search(args.query, k=args.k, chroma_path=args.chroma_path)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {"query": args.query, "model": search_model_name(), "results": hits},
                indent=2,
            )
        )
        return 0

    print(f'Query: "{args.query}"  (model={search_model_name()})\n')
    for hit in hits:
        sim = hit.get("similarity")
        sim_str = f"{sim:.3f}" if sim is not None else "n/a"
        print(f"{hit.get('rank')}. {hit.get('repo_id')}  (similarity≈{sim_str})")
        print(f"   {hit.get('url')}")
        snippet = (hit.get("description_snippet") or "").strip()
        if snippet:
            line = snippet.replace("\n", " ")
            if len(line) > 220:
                line = line[:217] + "..."
            print(f"   {line}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
