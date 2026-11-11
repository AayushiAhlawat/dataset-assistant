"""PostgreSQL connection helpers."""

from __future__ import annotations

import os
from typing import Any

import psycopg


def get_database_url() -> str:
    """Return DATABASE_URL from the environment or raise a clear error."""
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Example: "
            "postgresql://postgres:postgres@localhost:5432/hf_datasets"
        )
    return url


def connect() -> psycopg.Connection[Any]:
    """Open a new PostgreSQL connection."""
    return psycopg.connect(get_database_url())
