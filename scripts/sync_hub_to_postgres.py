#!/usr/bin/env python3
"""Sync public Hugging Face dataset metadata into PostgreSQL."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog_db.connection import connect  # noqa: E402
from catalog_db.hub_row import dataset_info_to_row  # noqa: E402
from catalog_db.repository import (  # noqa: E402
    apply_schema,
    count_datasets,
    finish_sync_run,
    last_successful_sync_finished_at,
    start_sync_run,
    upsert_dataset,
)

MODE_PRESETS: dict[str, dict[str, int | str | bool]] = {
    # First-time bootstrap: popular datasets first.
    "initial": {"max_rows": 500, "sort": "downloads", "direction": -1, "incremental": False},
    # Daily incremental: all Hub datasets with last_modified after last successful sync.
    "daily": {"max_rows": 0, "sort": "last_modified", "direction": -1, "incremental": True},
    # Background fill: entire public catalog (can take hours/days).
    "full": {"max_rows": 0, "sort": "last_modified", "direction": -1, "incremental": False},
}

DEFAULT_INCREMENTAL_LOOKBACK = timedelta(hours=24)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Hub → Postgres sync."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_PRESETS.keys()),
        default="initial",
        help="Preset: initial (500 popular), daily (incremental since last sync), full (no limit)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Override preset row limit (0 = no limit)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between upserts (default: 0)",
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include private repos (default: skip)",
    )
    parser.add_argument(
        "--sort",
        default=None,
        help="Override Hub sort field (e.g. downloads, last_modified)",
    )
    parser.add_argument(
        "--direction",
        type=int,
        default=None,
        help="Sort direction: -1 descending (default), 1 ascending",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Incremental watermark (ISO datetime). Daily mode uses last successful sync if omitted.",
    )
    return parser.parse_args()


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_since(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return _as_utc(datetime.fromisoformat(text))


def resolve_sync_params(
    args: argparse.Namespace,
) -> tuple[int | None, str, int, str, bool]:
    """Merge CLI args with mode presets into limit, sort, direction, mode label, incremental."""
    preset = MODE_PRESETS[args.mode]
    incremental = bool(preset.get("incremental"))
    max_rows = args.max_rows if args.max_rows is not None else int(preset["max_rows"])
    if incremental and args.max_rows is None:
        limit = None
    else:
        limit = None if max_rows == 0 else max_rows
    sort = args.sort or str(preset["sort"])
    direction = args.direction if args.direction is not None else int(preset["direction"])
    if direction not in (-1, 1):
        direction = -1
    if incremental:
        mode_label = f"{args.mode}|incremental|sort={sort}"
    else:
        mode_label = f"{args.mode}|sort={sort}|limit={limit or 'all'}"
    return limit, sort, direction, mode_label, incremental


def resolve_incremental_watermark(
    conn,
    *,
    since_arg: str | None,
) -> datetime:
    """Timestamp: only sync datasets with Hub last_modified strictly after this."""
    if since_arg:
        return _parse_since(since_arg)
    finished = last_successful_sync_finished_at(conn)
    if finished is not None:
        return _as_utc(finished)
    return datetime.now(timezone.utc) - DEFAULT_INCREMENTAL_LOOKBACK


def main() -> int:
    """List datasets from the Hub and upsert them into PostgreSQL."""
    load_dotenv(ROOT / ".env")
    args = parse_args()
    limit, sort, direction, mode_label, incremental = resolve_sync_params(args)

    api = HfApi()
    rows_upserted = 0
    run_id = 0
    total = 0

    try:
        with connect() as conn:
            apply_schema(conn)
            watermark: datetime | None = None
            if incremental:
                watermark = resolve_incremental_watermark(conn, since_arg=args.since)
                print(
                    f"Incremental sync: upserting datasets with last_modified "
                    f"after {watermark.isoformat()}"
                )
            run_id = start_sync_run(conn, mode_label)

            listings = api.list_datasets(
                full=True,
                limit=limit,
                sort=sort,
                direction=direction,
            )

            for info in listings:
                if incremental and watermark is not None:
                    modified = getattr(info, "last_modified", None)
                    if modified is not None:
                        if _as_utc(modified) <= watermark:
                            print(
                                f"  Reached watermark after {rows_upserted} upserts "
                                f"(stopped at {info.id})"
                            )
                            break

                if not args.include_private and getattr(info, "private", False):
                    continue
                if getattr(info, "disabled", False):
                    continue

                row = dataset_info_to_row(info)
                upsert_dataset(conn, row)
                rows_upserted += 1

                if rows_upserted % 50 == 0:
                    conn.commit()
                    print(f"  Upserted {rows_upserted} datasets...")

                if args.delay > 0:
                    time.sleep(args.delay)

            conn.commit()
            total = count_datasets(conn)
            finish_sync_run(
                conn,
                run_id,
                rows_upserted=rows_upserted,
                status="success",
            )
    except HfHubHTTPError as exc:
        print(f"Hub API error: {exc}", file=sys.stderr)
        if run_id:
            with connect() as conn:
                finish_sync_run(
                    conn,
                    run_id,
                    rows_upserted=rows_upserted,
                    status="failed",
                    error_message=str(exc),
                )
        return 1
    except Exception as exc:
        print(f"Sync failed: {exc}", file=sys.stderr)
        if run_id:
            with connect() as conn:
                finish_sync_run(
                    conn,
                    run_id,
                    rows_upserted=rows_upserted,
                    status="failed",
                    error_message=str(exc),
                )
        return 1

    print(f"Done. Upserted {rows_upserted} datasets this run. Total in DB: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
