# DataGurus — Hugging Face Dataset Discovery

Conversational assistant (later phases) for finding datasets on the [Hugging Face Hub](https://huggingface.co/datasets) using metadata search and RAG. **Phase 1** builds a local catalog from public Hub metadata only (no dataset file downloads).

## Phase 1: Catalog ingest

### Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional: set `HF_TOKEN` for higher API rate limits ([HF settings](https://huggingface.co/settings/tokens)).

### Run ingest

```bash
python scripts/ingest_hf_catalog.py
```

Defaults write `data/dataset_catalog.csv` (~400 unique datasets from 10 search seeds, 40 hits each).

Custom output:

```bash
python scripts/ingest_hf_catalog.py \
  --seeds employment salary housing \
  --limit-per-seed 50 \
  --csv data/dataset_catalog.csv \
  --sqlite data/dataset_catalog.db
```

### Catalog schema

| Column        | Description                                      |
|---------------|--------------------------------------------------|
| `id`          | Hub dataset repo id (e.g. `owner/name`)            |
| `title`       | Display title (currently same as id)             |
| `description` | Card / API description                           |
| `source`      | Always `Hugging Face`                            |
| `tags`        | JSON array of lowercase tags                     |
| `url`         | `https://huggingface.co/datasets/{id}`           |
| `license`     | License string when available                    |
| `downloads`   | Hub download count when available                |

### Roadmap

1. **Phase 1** — HF metadata ETL (this repo)
2. **Phase 2** — Embeddings + semantic search (Chroma)
3. **Phase 3** — LangGraph agent (clarify → retrieve → answer)
4. **Phase 4+** — FastAPI + React UI (optional)

Reference notebook: `uploads/Mini_DataGurus_LangGraph_With_OpenAI_4638.ipynb`
