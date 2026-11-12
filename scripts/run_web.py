#!/usr/bin/env python3
"""Start Phase 4 web UI and API (uvicorn)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Dev auto-reload")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import uvicorn
    except ImportError:
        print("Install web deps: pip install fastapi uvicorn", file=sys.stderr)
        return 1

    uvicorn.run(
        "catalog_api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(ROOT)] if args.reload else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
