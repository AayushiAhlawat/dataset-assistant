#!/usr/bin/env python3
"""Ingest public dataset metadata from Hugging Face Hub into a local catalog."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

DEFAULT_SEEDS = [
    "employment",
    "salary",
    "unemployment",
    "housing",
    "weather",
    "retail",
    "health",
    "education",
    "finance",
    "population",
]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "dataset_catalog.csv"
DEFAULT_SQLITE = ROOT / "data" / "dataset_catalog.db"


def normal_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_tags(raw_tags: list[str] | None) -> list[str]:
    tags = sorted(
        {normal_text(t).lower() for t in (raw_tags or []) if normal_text(t)}
    )
    return tags


def fetch_catalog(
    seeds: list[str],
    limit_per_seed: int,
    request_delay_s: float,
) -> list[dict]:
    api = HfApi()
    seen: set[str] = set()
    rows: list[dict] = []

    for term in seeds:
        print(f"Searching Hub: {term!r} (limit={limit_per_seed})")
        try:
            datasets = list(api.list_datasets(search=term, limit=limit_per_seed))
        except HfHubHTTPError as exc:
            print(f"  Warning: search failed for {term!r}: {exc}", file=sys.stderr)
            continue

        for listing in datasets:
            repo_id = listing.id
            if repo_id in seen:
                continue
            seen.add(repo_id)

            try:
                info = api.dataset_info(repo_id)
            except HfHubHTTPError as exc:
                print(f"  Skip {repo_id}: {exc}", file=sys.stderr)
                continue

            if getattr(info, "private", False):
                continue

            description = normal_text(info.description or "")
            if not description and info.card_data:
                description = normal_text(
                    getattr(info.card_data, "description", "") or ""
                )

            rows.append(
                {
                    "id": repo_id,
                    "title": normal_text(getattr(info, "id", repo_id) or repo_id),
                    "description": description[:4000],
                    "source": "Hugging Face",
                    "tags": normalize_tags(list(info.tags or [])),
                    "url": f"https://huggingface.co/datasets/{repo_id}",
                    "license": normal_text(getattr(info, "license", "") or ""),
                    "downloads": int(getattr(info, "downloads", 0) or 0),
                }
            )

            if request_delay_s > 0:
                time.sleep(request_delay_s)

        print(f"  Catalog size so far: {len(rows)} unique datasets")

    return rows


def to_dataframe(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "title",
                "description",
                "source",
                "tags",
                "url",
                "license",
                "downloads",
            ]
        )
    frame = pd.DataFrame(rows).drop_duplicates(subset="id").reset_index(drop=True)
    frame["tags"] = frame["tags"].apply(json.dumps)
    return frame


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {len(frame)} rows to {path}")


def save_sqlite(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        load_frame = frame.copy()
        load_frame.to_sql("datasets", conn, index=False)
    finally:
        conn.close()
    print(f"Wrote {len(frame)} rows to sqlite table datasets @ {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Hub search terms (default: curated list)",
    )
    parser.add_argument(
        "--limit-per-seed",
        type=int,
        default=40,
        help="Max datasets per search term (default: 40)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Seconds to sleep between dataset_info calls (default: 0.05)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Output CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=None,
        help="Optional SQLite path (e.g. data/dataset_catalog.db)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = fetch_catalog(args.seeds, args.limit_per_seed, args.delay)
    frame = to_dataframe(rows)

    if frame.empty:
        print("No datasets collected. Check network or HF API status.", file=sys.stderr)
        return 1

    save_csv(frame, args.csv)
    if args.sqlite:
        save_sqlite(frame, args.sqlite)

    described = (frame["description"].str.len() > 0).sum()
    print(f"Rows with description: {described}/{len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
