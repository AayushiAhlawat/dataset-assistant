#!/usr/bin/env python3
"""Create PostgreSQL tables for the dataset catalog."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog_db.repository import apply_schema, count_datasets  # noqa: E402
from catalog_db.connection import connect  # noqa: E402


def main() -> int:
    """Apply schema.sql to the database in DATABASE_URL."""
    load_dotenv(ROOT / ".env")
    try:
        with connect() as conn:
            apply_schema(conn)
            total = count_datasets(conn)
    except Exception as exc:
        print(f"Schema init failed: {exc}", file=sys.stderr)
        return 1
    print(f"PostgreSQL schema ready. datasets row count: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
