"""Map Hugging Face DatasetInfo objects to PostgreSQL row dicts."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from huggingface_hub.hf_api import DatasetInfo


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return str(value)


def _description_and_license(info: DatasetInfo) -> tuple[str, str]:
    description = str(getattr(info, "description", "") or "").strip()
    license_str = str(getattr(info, "license", "") or "").strip()
    card = info.card_data
    if card is not None:
        if not description:
            description = str(getattr(card, "description", "") or "").strip()
        if not license_str:
            license_str = str(getattr(card, "license", "") or "").strip()
    return description, license_str


def _siblings_json(info: DatasetInfo) -> list[dict[str, Any]]:
    if not info.siblings:
        return []
    out: list[dict[str, Any]] = []
    for sibling in info.siblings:
        out.append(
            {
                "rfilename": sibling.rfilename,
                "size": sibling.size,
            }
        )
    return out


def dataset_info_to_row(info: DatasetInfo) -> dict[str, Any]:
    """Convert Hub DatasetInfo to columns for the datasets table."""
    repo_id = info.id
    description, license_str = _description_and_license(info)
    tags = sorted({str(t).lower() for t in (info.tags or []) if str(t).strip()})
    card_json = _json_safe(info.card_data.to_dict()) if info.card_data else None
    siblings_json = _siblings_json(info)
    raw_hub_json = {
        "id": info.id,
        "author": info.author,
        "sha": info.sha,
        "private": info.private,
        "gated": info.gated,
        "disabled": info.disabled,
        "downloads": info.downloads,
        "downloads_all_time": info.downloads_all_time,
        "likes": info.likes,
        "trending_score": info.trending_score,
        "paperswithcode_id": info.paperswithcode_id,
        "tags": info.tags,
    }

    return {
        "repo_id": repo_id,
        "title": repo_id,
        "description": description[:8000],
        "author": info.author,
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "license": license_str,
        "source": "Hugging Face",
        "downloads": info.downloads,
        "downloads_all_time": info.downloads_all_time,
        "likes": info.likes,
        "trending_score": info.trending_score,
        "private": bool(info.private),
        "gated": str(info.gated) if info.gated is not None else None,
        "disabled": info.disabled,
        "sha": info.sha,
        "paperswithcode_id": info.paperswithcode_id,
        "created_at": info.created_at,
        "last_modified": info.last_modified,
        "tags": json.dumps(tags),
        "card_json": json.dumps(card_json) if card_json is not None else None,
        "siblings_json": json.dumps(siblings_json),
        "raw_hub_json": json.dumps(_json_safe(raw_hub_json)),
    }
