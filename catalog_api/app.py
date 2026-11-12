"""FastAPI application: health, semantic search, chat."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from catalog_api.runner import (
    hit_to_api_dict,
    get_chroma_status,
    openai_configured,
    run_chat_turn,
    run_semantic_search,
)
from catalog_search.catalog import DEFAULT_CHROMA_PATH

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=8, ge=1, le=50)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str | None = None
    top_k: int = Field(default=8, ge=1, le=50)


def create_app() -> FastAPI:
    app = FastAPI(
        title="HF Dataset Discovery API",
        description="Phase 4: web UI + search + LangGraph chat",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        ok, chroma_msg = get_chroma_status()
        return {
            "chroma_ok": ok,
            "chroma_message": chroma_msg,
            "openai_configured": openai_configured(),
            "chat_available": ok and openai_configured(),
        }

    @app.post("/api/search")
    def search(body: SearchRequest) -> dict:
        try:
            hits = run_semantic_search(body.query.strip(), k=body.k)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "query": body.query,
            "results": [hit_to_api_dict(h) for h in hits],
        }

    @app.post("/api/chat")
    def chat(body: ChatRequest) -> dict:
        if not openai_configured():
            raise HTTPException(
                status_code=503,
                detail="OPENAI_API_KEY is not set. Chat is unavailable.",
            )
        try:
            result = run_chat_turn(
                body.message,
                thread_id=body.thread_id,
                top_k=body.top_k,
                chroma_path=DEFAULT_CHROMA_PATH,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc
        return {
            "thread_id": result.thread_id,
            "reply": result.reply,
            "hits": [hit_to_api_dict(h) for h in result.hits],
        }

    @app.get("/")
    def index() -> FileResponse:
        index_path = WEB_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=404, detail="Web UI not found")
        return FileResponse(index_path)

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    return app


app = create_app()
