# Hugging Face Dataset Discovery

Conversational assistant for finding datasets on the [Hugging Face Hub](https://huggingface.co/datasets) using metadata sync, semantic search, and RAG chat (Phases 1–3).

**Deep dive (architecture, every file/function, flows):** [PROJECT_COMPLETE_STUDY_GUIDE.md](./PROJECT_COMPLETE_STUDY_GUIDE.md)

## Phase 1: Catalog ingest

Sync metadata from the full Hub into **PostgreSQL** (`sync_hub_to_postgres.py`).

### Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `DATABASE_URL`. Optional: `HF_TOKEN` for higher Hub rate limits.

**Start Postgres (local)**

```bash
brew services start postgresql@14   # or your installed version
createdb hf_datasets
```

Set `DATABASE_URL` in `.env` to match your local user and database (see `.env.example`).

```bash
python scripts/init_postgres_schema.py
python scripts/verify_postgres_catalog.py
```

### PostgreSQL sync modes (metadata only — no dataset file downloads)

| Mode | Command | Purpose |
|------|---------|---------|
| **initial** | `python scripts/sync_hub_to_postgres.py --mode initial` | First 500 by **downloads** (you already did this) |
| **daily** | `python scripts/sync_hub_to_postgres.py --mode daily` | **All** datasets **new or updated** since the last **successful** sync |
| **full** | `python scripts/sync_hub_to_postgres.py --mode full` | **Entire** public catalog (`--max-rows 0`) — long background job |

Daily mode walks the Hub **newest-first** and stops when it reaches datasets older than the watermark (`finished_at` of the last successful sync). First run with no prior sync uses the last **24 hours**. Override watermark: `--since 2025-11-01T00:00:00Z`. Cap for testing: `--max-rows 500`.

**Manual daily sync** (schedule on a server later):

```bash
source .venv/bin/activate
./scripts/run_daily_sync.sh
# or: python scripts/sync_hub_to_postgres.py --mode daily
```

### Phase 1 checklist

1. Postgres running, `DATABASE_URL` in `.env`
2. `init_postgres_schema.py` — tables created
3. `sync_hub_to_postgres.py --mode initial` — bootstrap catalog
4. `verify_postgres_catalog.py` — counts look right
5. (Optional) `--mode full` when you want the whole Hub in the background
6. (Optional) run `--mode daily` manually until server cron is set up

### Catalog schema (`datasets` table)

| Column        | Description                                      |
|---------------|--------------------------------------------------|
| `repo_id`     | Hub dataset id (e.g. `owner/name`)               |
| `title`       | Display title (currently same as id)             |
| `description` | Card / API description                           |
| `source`      | Always `Hugging Face`                            |
| `tags`        | JSON array of lowercase tags                     |
| `url`         | `https://huggingface.co/datasets/{id}`           |
| `license`     | License string when available                    |
| `downloads`   | Hub download count when available                |

## Phase 2: Semantic search (Chroma + MiniLM)

Embeds catalog rows with local **[sentence-transformers](https://www.sbert.net/)** (`all-MiniLM-L6-v2`) and stores vectors in **[ChromaDB](https://www.trychroma.com/)** under `data/chroma/` (regenerable; gitignored).

Indexes **PostgreSQL** rows (public datasets with a non-empty description by default). Rebuild after Hub syncs.

```bash
source .venv/bin/activate
python scripts/build_vector_index.py --limit 500      # smoke test
python scripts/build_vector_index.py                  # ~438k rows, hours on CPU
python scripts/build_vector_index.py --min-downloads 1000
```

Search:

```bash
python scripts/semantic_search.py "youth unemployment in Africa"
python scripts/semantic_search.py "housing price time series" -k 8
python scripts/semantic_search.py "medical imaging" --json
```

Phase 3 hydrates top-k hits from Postgres (description, tags, stats) before the LLM answers.

## Phase 3: LangGraph agent (OpenAI + retrieval)

Flow: **plan** (clarify | new **search** | **compare_previous**) → **retrieve** (optional) → **hydrate** (Postgres) → **answer**. Follow-ups like “which is best for ML?” reuse **`last_hits`** from the prior turn (no new Chroma search) when the planner chooses `compare_previous`.

| File | Role |
|------|------|
| `catalog_agent/graph.py` | LangGraph: `plan` → `retrieve` → `answer`, in-memory thread memory |
| `catalog_search/hydrate.py` | Load Postgres rows for top-k hits; format context for the LLM |
| `catalog_search/health.py` | Verify Chroma index exists before chat starts |
| `scripts/chat_agent.py` | CLI REPL or single `-m` message |

**Requires:** `OPENAI_API_KEY` in `.env`, built Chroma index (`build_vector_index.py`), Postgres up (`DATABASE_URL` for hydration).

Copy `.env.example` to `.env`. Optional: `OPENAI_MODEL` (default `gpt-4o-mini`).

```bash
python scripts/chat_agent.py
python scripts/chat_agent.py -m "datasets about youth unemployment in Africa"
python scripts/chat_agent.py -m "housing time series" -k 10 -v
```

### Roadmap

1. **Phase 1** — HF metadata ETL (this repo)
2. **Phase 2** — Embeddings + semantic search (Chroma) — **implemented**
3. **Phase 3** — LangGraph agent (clarify → retrieve → answer) — **implemented**
4. **Phase 4+** — FastAPI + React UI (optional)
