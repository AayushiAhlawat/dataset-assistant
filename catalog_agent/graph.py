"""LangGraph flow: plan (clarify or search) → retrieve → answer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from catalog_search.catalog import DEFAULT_CHROMA_PATH
from catalog_search.hydrate import format_hits_for_llm, hydrate_hits
from catalog_search.search import SearchHit, semantic_search

_PLAN_RULES = """Rules:
- If the user greets you or asks what you can do, set action=clarify with a short helpful reply.
- If the request is too vague to search (e.g. "data" or "ML dataset" with no topic), ask ONE short clarifying question (action=clarify).
- If the user refers to datasets already recommended in this thread ("which one", "which is best", "compare those", "the third one", "for machine learning" as a follow-up), and previous datasets are listed below, use action=compare_previous (do NOT run a new broad search).
- If you need a fresh discovery search (new topic or no prior results to compare), set action=search with search_query: a concise English phrase for semantic search (include region, domain, modality, format when the user mentioned them).
- Do not invent dataset names; you only decide clarify vs search vs compare_previous.
"""


def _plan_system(catalog_note: str) -> str:
    return (
        "You route user messages for a Hugging Face dataset discovery assistant.\n"
        f"{catalog_note}\n\n"
        + _PLAN_RULES
    )


ANSWER_SYSTEM = """You recommend Hugging Face datasets ONLY from the retrieved list below.
Be concise and practical. For each good match include: repo_id, why it fits, license if known, and the URL.
Prefer higher-download datasets when relevance is similar. Mention gated access if listed.
If none fit well, say so and suggest how the user could refine their ask.
Never cite datasets that are not in the retrieved list."""

COMPARE_ANSWER_SYSTEM = """You compare Hugging Face datasets ONLY from the retrieved list below.
The user is asking a follow-up about datasets already shown in this conversation (e.g. which is best for machine learning, NLP, or fine-tuning).
Rank or contrast them using only the metadata provided (description, tags, license, downloads, gated).
State which is the best fit and why; mention trade-offs honestly.
You do not have column names, label fields, or sample rows — say when ML suitability cannot be verified from metadata alone.
Never cite datasets that are not in the retrieved list."""


class PlanDecision(BaseModel):
    """Structured output from the planner: clarify, search, or compare prior results."""

    action: Literal["clarify", "search", "compare_previous"] = Field(
        description=(
            "clarify = ask user one question; "
            "search = new semantic retrieval; "
            "compare_previous = follow-up about datasets already recommended in this thread"
        )
    )
    message: str = Field(
        default="",
        description="When action=clarify, the full assistant reply to the user",
    )
    search_query: str = Field(
        default="",
        description="When action=search, query string for vector search",
    )


class AgentState(TypedDict, total=False):
    """LangGraph state passed between plan, retrieve, and answer nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    search_query: str
    hits: list[SearchHit]
    last_hits: list[SearchHit]
    route: str
    answer_mode: str


@dataclass
class AgentSettings:
    """Runtime configuration for retrieval and OpenAI models."""

    chroma_path: Path = DEFAULT_CHROMA_PATH
    top_k: int = 8
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    catalog_note: str = (
        "The catalog is a local semantic index over Hugging Face dataset metadata "
        "(PostgreSQL + Chroma), not a live Hub API search."
    )
    verbose_retrieval: bool = False


def _latest_user_text(messages: list[BaseMessage]) -> str:
    """Return the content of the most recent human message, or an empty string."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            return str(content)
    return ""


def _conversation_transcript(messages: list[BaseMessage], limit: int = 12) -> str:
    """Format recent user/assistant turns as plain text for the planner prompt."""
    lines: list[str] = []
    for msg in messages[-limit:]:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            continue
        content = msg.content
        text = content if isinstance(content, str) else str(content)
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _format_last_hits_for_planner(last_hits: list[SearchHit]) -> str:
    if not last_hits:
        return "(none — first turn or nothing retrieved yet)"
    lines = []
    for hit in last_hits:
        repo = hit.get("repo_id") or ""
        if repo:
            lines.append(f"- {repo}")
    return "\n".join(lines) if lines else "(none)"


def build_agent(settings: AgentSettings | None = None):
    """Build and compile the LangGraph agent with in-memory conversation checkpointing."""
    settings = settings or AgentSettings()
    llm_cache: dict[str, object] = {}

    def planner_structured():
        """Return a cached OpenAI client that outputs structured ``PlanDecision`` objects."""
        if "planner" not in llm_cache:
            llm_cache["planner"] = ChatOpenAI(
                model=settings.openai_model, temperature=0
            ).with_structured_output(PlanDecision)
        return llm_cache["planner"]

    def answer_llm():
        """Return a cached OpenAI client used to write the final dataset recommendations."""
        if "answer" not in llm_cache:
            llm_cache["answer"] = ChatOpenAI(
                model=settings.openai_model, temperature=0.2
            )
        return llm_cache["answer"]

    plan_system = _plan_system(settings.catalog_note)

    def plan_node(state: AgentState) -> dict:
        """Decide whether to clarify with the user or proceed with semantic search."""
        transcript = _conversation_transcript(state["messages"])
        last_hits = state.get("last_hits") or []
        prior = _format_last_hits_for_planner(last_hits)
        decision: PlanDecision = planner_structured().invoke(
            [
                SystemMessage(content=plan_system),
                HumanMessage(
                    content=(
                        f"Conversation:\n{transcript}\n\n"
                        f"Datasets recommended in the last search turn (for follow-ups):\n{prior}\n\n"
                        "Decide the next step."
                    )
                ),
            ]
        )
        if decision.action == "clarify":
            text = decision.message.strip() or "What kind of dataset are you looking for?"
            return {
                "messages": [AIMessage(content=text)],
                "route": "end",
                "search_query": "",
                "hits": [],
                "answer_mode": "recommend",
            }
        if decision.action == "compare_previous" and last_hits:
            if settings.verbose_retrieval:
                print(
                    "Follow-up: comparing previous retrieval "
                    f"({len(last_hits)} datasets), no new Chroma search"
                )
            return {
                "route": "compare",
                "search_query": "",
                "hits": list(last_hits),
                "answer_mode": "compare",
            }
        if decision.action == "compare_previous" and not last_hits:
            if settings.verbose_retrieval:
                print("compare_previous requested but no last_hits; falling back to search")
        query = decision.search_query.strip() or _latest_user_text(state["messages"])
        return {
            "route": "retrieve",
            "search_query": query,
            "hits": [],
            "answer_mode": "recommend",
        }

    def retrieve_node(state: AgentState) -> dict:
        """Run Phase 2 vector search and store top-k hits in state."""
        query = (state.get("search_query") or _latest_user_text(state["messages"])).strip()
        if settings.verbose_retrieval:
            print(f"Retrieving datasets for query: {query!r}")
        hits = semantic_search(
            query,
            k=settings.top_k,
            chroma_path=settings.chroma_path,
        )
        if settings.verbose_retrieval:
            print(f"  Retrieved {len(hits)} hit(s)")
        return {"hits": hits, "search_query": query}

    def answer_node(state: AgentState) -> dict:
        """Generate a grounded reply using only retrieved catalog entries."""
        hits = state.get("hits") or []
        context = format_hits_for_llm(hits, hydrate_hits(hits))
        user_ask = _latest_user_text(state["messages"])
        compare = state.get("answer_mode") == "compare"
        system = COMPARE_ANSWER_SYSTEM if compare else ANSWER_SYSTEM
        query_note = (
            "(comparison of datasets from the previous search in this thread)"
            if compare
            else state.get("search_query", "")
        )
        reply = answer_llm().invoke(
            [
                SystemMessage(content=system),
                HumanMessage(
                    content=(
                        f"User request: {user_ask}\n\n"
                        f"Search query used: {query_note}\n\n"
                        f"Retrieved datasets:\n{context}"
                    )
                ),
            ]
        )
        content = reply.content if isinstance(reply.content, str) else str(reply.content)
        return {
            "messages": [AIMessage(content=content)],
            "last_hits": list(hits),
        }

    def route_after_plan(state: AgentState) -> str:
        """Route to END, compare (skip retrieve), or new retrieval."""
        route = state.get("route")
        if route == "end":
            return END
        if route == "compare":
            return "answer"
        return "retrieve"

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"retrieve": "retrieve", "answer": "answer", END: END},
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
