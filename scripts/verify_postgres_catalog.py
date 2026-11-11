#!/usr/bin/env python3
"""Print PostgreSQL catalog health stats."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog_db.connection import connect  # noqa: E402
from catalog_db.repository import apply_schema, catalog_stats, last_sync_run  # noqa: E402


def main() -> int:
    """Show dataset counts and the latest sync run."""
    load_dotenv(ROOT / ".env")
    try:
        with connect() as conn:
            apply_schema(conn)
            stats = catalog_stats(conn)
            sync = last_sync_run(conn)
            cur = conn.execute(
                """
                SELECT repo_id, downloads
                FROM datasets
                ORDER BY downloads DESC NULLS LAST
                LIMIT 3
                """
            )
            top = cur.fetchall()
    except Exception as exc:
        print(f"Verify failed: {exc}", file=sys.stderr)
        return 1

    print("PostgreSQL catalog OK")
    print(f"  Total datasets:      {stats['total']}")
    print(f"  With description:    {stats['with_description']}")
    print(f"  Private (in DB):     {stats['private_count']}")
    if sync:
        print(f"  Last sync:           #{sync['id']} mode={sync['mode']} status={sync['status']}")
        print(f"                       upserted={sync['rows_upserted']} finished={sync['finished_at']}")
    else:
        print("  Last sync:           (none)")
    if top:
        print("  Top by downloads:")
        for repo_id, downloads in top:
            print(f"    - {repo_id} ({downloads})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
