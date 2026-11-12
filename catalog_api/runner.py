"""Shared agent and search execution for CLI and API."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from catalog_agent.graph import AgentSettings, build_agent
from catalog_search.catalog import DEFAULT_CHROMA_PATH
from catalog_search.health import chroma_index_ready
from catalog_search.search import SearchHit, semantic_search
from catalog_api.messages import latest_assistant_text

ROOT = Path(__file__).resolve().parents[1]
_agent_lock = Lock()
_agent_bundle: dict[str, Any] | None = None


@dataclass
class ChatResult:
    thread_id: str
    reply: str
    hits: list[SearchHit]


def _ensure_env() -> None:
    load_dotenv(ROOT / ".env")


def openai_configured() -> bool:
    _ensure_env()
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def get_chroma_status(chroma_path: Path = DEFAULT_CHROMA_PATH) -> tuple[bool, str]:
    _ensure_env()
    return chroma_index_ready(chroma_path)


def _get_agent(top_k: int = 8, chroma_path: Path = DEFAULT_CHROMA_PATH):
    global _agent_bundle
    with _agent_lock:
        if _agent_bundle is not None:
            if (
                _agent_bundle["top_k"] == top_k
                and _agent_bundle["chroma_path"] == chroma_path
            ):
                return _agent_bundle["agent"], _agent_bundle["catalog_note"]

        ok, chroma_msg = chroma_index_ready(chroma_path)
        if not ok:
            raise RuntimeError(chroma_msg)

        catalog_note = (
            f"The catalog is a local semantic index over Hugging Face dataset metadata "
            f"(PostgreSQL + Chroma). {chroma_msg}"
        )
        settings = AgentSettings(
            top_k=top_k,
            chroma_path=chroma_path,
            catalog_note=catalog_note,
            verbose_retrieval=False,
        )
        agent = build_agent(settings)
        _agent_bundle = {
            "agent": agent,
            "catalog_note": catalog_note,
            "top_k": top_k,
            "chroma_path": chroma_path,
        }
        return agent, catalog_note


def run_semantic_search(
    query: str,
    *,
    k: int = 8,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
) -> list[SearchHit]:
    _ensure_env()
    ok, msg = chroma_index_ready(chroma_path)
    if not ok:
        raise RuntimeError(msg)
    return semantic_search(query, k=k, chroma_path=chroma_path)


def run_chat_turn(
    message: str,
    *,
    thread_id: str | None = None,
    top_k: int = 8,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
) -> ChatResult:
    _ensure_env()
    if not openai_configured():
        raise RuntimeError("OPENAI_API_KEY is not set.")

    message = message.strip()
    if not message:
        raise ValueError("message must not be empty")

    agent, _ = _get_agent(top_k=top_k, chroma_path=chroma_path)
    tid = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}
    result = agent.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    reply = latest_assistant_text(result.get("messages", []))
    hits = list(result.get("hits") or result.get("last_hits") or [])
    return ChatResult(thread_id=tid, reply=reply, hits=hits)


def hit_to_api_dict(hit: SearchHit) -> dict[str, Any]:
    return {
        "rank": hit.get("rank"),
        "repo_id": hit.get("repo_id"),
        "url": hit.get("url"),
        "title": hit.get("title"),
        "license": hit.get("license"),
        "downloads": hit.get("downloads"),
        "similarity": hit.get("similarity"),
        "description_snippet": hit.get("description_snippet"),
    }
