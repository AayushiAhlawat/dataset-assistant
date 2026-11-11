#!/usr/bin/env python3
"""Interactive LangGraph agent: clarify → retrieve → answer."""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog_agent.graph import AgentSettings, build_agent  # noqa: E402
from catalog_search.catalog import DEFAULT_CHROMA_PATH  # noqa: E402
from catalog_search.health import chroma_index_ready  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the chat agent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--message",
        help="Single-turn message (default: interactive REPL)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=8,
        help="Number of datasets to retrieve (default: 8)",
    )
    parser.add_argument(
        "--thread-id",
        default="",
        help="Conversation thread id for multi-turn memory (default: random per run)",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=None,
        help="Override Chroma directory (default: data/chroma)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print retrieval query and hit counts",
    )
    return parser.parse_args()


def print_assistant(messages: list) -> None:
    """Print the most recent assistant message from an invoke result."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            text = content if isinstance(content, str) else str(content)
            print(text)
            return


def main() -> int:
    """Start the REPL or handle a single ``-m`` message through the LangGraph agent."""
    load_dotenv(ROOT / ".env")
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Add it to .env or your environment.",
            file=sys.stderr,
        )
        return 1

    chroma_path = args.chroma_path or DEFAULT_CHROMA_PATH
    ok, chroma_msg = chroma_index_ready(chroma_path)
    if not ok:
        print(chroma_msg, file=sys.stderr)
        return 1

    catalog_note = (
        f"The catalog is a local semantic index over Hugging Face dataset metadata "
        f"(PostgreSQL + Chroma). {chroma_msg}"
    )
    settings = AgentSettings(
        top_k=args.k,
        chroma_path=chroma_path,
        catalog_note=catalog_note,
        verbose_retrieval=args.verbose,
    )

    try:
        agent = build_agent(settings)
    except Exception as exc:
        print(f"Failed to start agent: {exc}", file=sys.stderr)
        return 1

    thread_id = args.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    def run_turn(user_text: str) -> int:
        """Invoke the agent for one user message and print the assistant reply."""
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_text)]},
                config=config,
            )
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Agent error: {exc}", file=sys.stderr)
            return 1
        print_assistant(result.get("messages", []))
        return 0

    if args.message:
        return run_turn(args.message)

    print(f"Dataset discovery agent (Phase 3). {chroma_msg}")
    print("Type 'quit' or Ctrl-D to exit.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit", "q"}:
            break
        print("\nAssistant:")
        if run_turn(user_text) != 0:
            return 1
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
