"""Extract text from LangChain message lists."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage


def latest_assistant_text(messages: list[BaseMessage]) -> str:
    """Return the content of the most recent assistant message."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""
