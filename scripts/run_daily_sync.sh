#!/usr/bin/env bash
# Manual daily Hub → Postgres sync (metadata only).
# Upserts every public dataset whose Hub last_modified is after the last successful sync.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
python scripts/sync_hub_to_postgres.py --mode daily
