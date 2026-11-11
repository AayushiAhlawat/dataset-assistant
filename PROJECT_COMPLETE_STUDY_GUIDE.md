# Hugging Face Dataset Discovery Assistant

## Full project study guide (Phases 1–3)

> **Purpose:** This is the document to reread when you want to remember **everything you built**, **why you built it**, **what every file and function does**, and **exactly how data flows** from Hugging Face → PostgreSQL → MiniLM → ChromaDB → **LangGraph + OpenAI** chat answers.
>
> **Scope:** Phase 1 (Hub metadata → PostgreSQL), Phase 2 (Postgres → MiniLM → Chroma semantic search), and **Phase 3** (LangGraph agent: plan → retrieve → hydrate → answer).
>
> **Source note:** This guide is based on the supplied Phase 1–2 architecture description and file/function notes. The *current* PostgreSQL-era Python and SQL source files were not included in the materials provided for this guide. Function behavior below follows your notes; illustrative pseudocode is explicitly marked. Consult the actual repository files for exact signatures, SQL, exception handling, API parameters, and implementation details.
>
> **Update:** Legacy CSV ingest (`ingest_hf_catalog.py`), Postgres→CSV export, and `--source csv` indexing were **removed**. The catalog is **PostgreSQL-only**; Phase 2 always indexes from Postgres.

---

## Table of contents

1. [The project in 60 seconds](#1-the-project-in-60-seconds)
2. [Your completed milestones](#2-your-completed-milestones)
3. [The architecture: three separate workflows](#3-the-architecture-three-separate-workflows)
4. [Every technology: what and why](#4-every-technology-what-and-why)
5. [Directory layout and configuration](#5-directory-layout-and-configuration)
6. [Phase 1 deep dive — Hugging Face to PostgreSQL](#6-phase-1-deep-dive--hugging-face-to-postgresql)
7. [Phase 1 file and function reference](#7-phase-1-file-and-function-reference)
8. [Phase 1 sync modes and the watermark](#8-phase-1-sync-modes-and-the-watermark)
9. [Phase 2 deep dive — PostgreSQL to ChromaDB](#9-phase-2-deep-dive--postgresql-to-chromadb)
10. [How MiniLM makes embeddings internally](#10-how-minilm-makes-embeddings-internally)
11. [Phase 2 file and function reference](#11-phase-2-file-and-function-reference)
12. [How semantic search works, step by step](#12-how-semantic-search-works-step-by-step)
13. [The full journey of one dataset](#13-the-full-journey-of-one-dataset)
14. [Command-by-command operating guide](#14-command-by-command-operating-guide)
15. [Important design decisions and limitations](#15-important-design-decisions-and-limitations)
16. [Troubleshooting and debugging](#16-troubleshooting-and-debugging)
17. [Definitions and common confusion](#17-definitions-and-common-confusion)
18. [What could replace each component?](#18-what-could-replace-each-component)
19. [Quick revision sheet](#19-quick-revision-sheet)
20. [Self-test and interview-style questions](#20-self-test-and-interview-style-questions)
21. [Phase 3 deep dive — conversational discovery](#21-phase-3-deep-dive--conversational-discovery)
22. [Phase 3 file and function reference](#22-phase-3-file-and-function-reference)
23. [How one chat turn works, step by step](#23-how-one-chat-turn-works-step-by-step)
24. [Phase 3 commands and operating notes](#24-phase-3-commands-and-operating-notes)
25. [Phase 3 design decisions, costs, and limits](#25-phase-3-design-decisions-costs-and-limits)
26. [Phase 3 troubleshooting](#26-phase-3-troubleshooting)
27. [Phase 3 self-test questions](#27-phase-3-self-test-questions)

---

# 1. The project in 60 seconds

**What are we building?** A search engine that lets a person describe the *kind* of public dataset they want and find relevant dataset repositories on [Hugging Face](https://huggingface.co/datasets).

**Example user search:** “Find datasets about employee compensation in Canada.”

The system searches **information about datasets** (names, descriptions, tags, licenses, etc.). **It does not download the datasets' underlying training files, images, or data rows.**

Three completed phases:

- **Phase 1 = Build a catalog.** Ask the Hugging Face API for public dataset metadata; clean it; save records into **PostgreSQL**.
- **Phase 2 = Build semantic search.** Read eligible descriptions from PostgreSQL; turn text into numerical **embeddings** using local **MiniLM**; store/search those embeddings with **ChromaDB**.
- **Phase 3 = Conversational assistant.** **LangGraph** orchestrates: **OpenAI** decides clarify vs search → **Chroma** retrieves top-k → **PostgreSQL hydrates** full metadata → **OpenAI** writes grounded recommendations.

**One sentence to memorize:** **PostgreSQL keeps the facts; MiniLM + Chroma find similar datasets; OpenAI explains only what retrieval actually returned (RAG).**

---

# 2. Your completed milestones

| Milestone | Recorded result | What the number means |
|---|---:|---|
| Full Hub catalog sync | ~1,113,540 PostgreSQL rows | Dataset repository metadata records, not downloaded datasets |
| Full semantic-search index | **438,350** Chroma records | Eligible datasets indexed as embeddings |
| Search entry point | `scripts/semantic_search.py` | Accepts a natural-language query from the terminal (Phase 2 only) |
| Chat entry point | `scripts/chat_agent.py` | Interactive or single-turn **Phase 3** agent |
| Default LLM | `gpt-4o-mini` (configurable via `OPENAI_MODEL`) | Planner + answer writer; **requires API key** |
| Embedding model | `all-MiniLM-L6-v2` | A locally run, pretrained Sentence Transformer |
| Embedding size | **384 dimensions** | Each indexed text and query becomes a vector with 384 numbers |
| Chroma collection | `hf_datasets` | Named vector collection stored on disk under `data/chroma/` |

**Why are the record counts different?** The index builder selects eligible records. Your notes say the full index included **public datasets with descriptions**. Consequently, PostgreSQL has more records than Chroma. The precise reasons for every exclusion require checking actual index-query filters and database counts.

**Status versus proof:** These counts are results reported in your project notes, not a live verification of your local PostgreSQL or Chroma files.

---

# 3. The architecture: three separate workflows

This is the **single most important distinction**. Collection, index building, and searching happen at *different times*.

## 3.1 Workflow A — Collect metadata (Phase 1)

```mermaid
flowchart TD
    A["Public Hugging Face Hub API"] --> B["sync_hub_to_postgres.py"]
    B --> C["hub_row.py: normalize fields"]
    C --> D["repository.py: UPSERT"]
    D --> E[("PostgreSQL datasets table")]
    B --> F[("PostgreSQL sync_runs table")]
```

**When?** On initial or full synchronization, and later when you run a daily sync. **Output:** PostgreSQL records.

## 3.2 Workflow B — Create semantic index (Phase 2 build)

```mermaid
flowchart TD
    A[("PostgreSQL datasets")] --> B["index_queries.py: choose eligible rows"]
    B --> C["catalog.py: make embeddable text + metadata"]
    C --> D["store.py: Chroma collection with MiniLM embedding function"]
    D --> E["MiniLM: text to 384-number vector"]
    E --> F[("ChromaDB: hf_datasets")]
```

**When?** Whenever you explicitly build or rebuild the index. **Output:** Persistent vectors, identifying metadata, and searchable Chroma collection.

## 3.3 Workflow C — Search (Phase 2 query)

```mermaid
flowchart TD
    A["User: Find salary datasets"] --> B["scripts/semantic_search.py"]
    B --> C["catalog_search.search.semantic_search"]
    C --> D["Chroma embeds query using MiniLM"]
    D --> E["Compare query vector to stored dataset vectors"]
    E --> F["Return closest IDs, metadata, distances"]
    F --> G["Print results in terminal"]
```

**When?** Every search request. **Output:** Relevant dataset repository IDs/URLs/descriptions and similarity-like scores. **No OpenAI API call and no generative answer in Phase 2.**

## 3.4 Workflow D — Chat (Phase 3)

```mermaid
flowchart TD
    A["User message"] --> B["scripts/chat_agent.py"]
    B --> C["chroma_index_ready() preflight"]
    C --> D["LangGraph: plan_node — OpenAI structured PlanDecision"]
    D -->|clarify| E["AIMessage to user — END"]
    D -->|search| F["retrieve_node — semantic_search()"]
    F --> G[("Chroma: top-k hits")]
    G --> H["answer_node — hydrate_hits()"]
    H --> I[("PostgreSQL: full rows for repo_ids")]
    I --> J["format_hits_for_llm()"]
    J --> K["OpenAI answer — grounded reply"]
    K --> L["AIMessage to user — END"]
```

**When?** Each user turn in the chat CLI (or each `-m` message). **Output:** Natural-language dataset recommendations with Hub links. **Costs:** OpenAI tokens for plan + answer; **no** OpenAI for embeddings (still MiniLM locally).

> **Critical:** A daily Hub sync updates **PostgreSQL**, **not Chroma**. New metadata is searchable semantically only after the Chroma index is updated/rebuilt. Hydration reads **live Postgres**, so descriptions/tags can be newer than what was embedded until you **rebuild the index**.

---

# 4. Every technology: what and why

| Technology | Layperson's description | Actual responsibility |
|---|---|---|
| **Hugging Face Hub** | Huge online library of datasets | Source of the public metadata |
| **`huggingface_hub`** | Python messenger to Hugging Face | Lists/reads dataset repository metadata |
| **API** | Way one program requests information from another | Gives Python access to Hub metadata |
| **Python** | Language connecting all pieces | Runs sync, transformation, indexing, searching |
| **PostgreSQL** | Well-organized library catalog | Main *source of truth* for dataset metadata |
| **SQL** | Language for reading/writing catalog tables | Insert, update, select, filter, count |
| **`psycopg`** | Connector between Python and Postgres | Creates DB connections and executes SQL |
| **`.env` / `python-dotenv`** | Private configuration notes | Loads DB URL and optional Hugging Face token |
| **Sentence Transformers** | Text-to-number meaning translator | Provides pretrained embedding models |
| **LangGraph** | Stateful workflow graph for agents | Runs plan → retrieve → answer with memory |
| **LangChain OpenAI** | OpenAI chat client wrapper | Structured planner output + final answer text |
| **OpenAI API** | Cloud language model service | Clarification, search-query rewriting, grounded recommendations (**Phase 3 only**) |
| **MiniLM** | Specific small Transformer model | Converts text into 384-dimensional vectors |
| **PyTorch** | Neural-network engine | Runs the local embedding model |
| **ChromaDB** | Nearby-meaning finder | Stores/indexes vectors and returns nearest matches |
| **NumPy** | Number-array library | Supports local numerical/ML computation |
| **pgAdmin / `psql`** | Ways to inspect a DB | Useful for browsing/testing, not necessary for Python pipeline |

## 4.1 PostgreSQL vs ChromaDB

**PostgreSQL stores facts:** complete metadata, document-card JSON, file list metadata, synchronization times, sorting/filtering fields.

**Chroma stores a search copy:** vectors + document text + lightweight metadata such as ID, title, URL, tags and snippets.

**MiniLM isn't a database:** it converts text into vectors. **Chroma isn't an LLM:** it searches numerical vectors. **Postgres isn't automatically semantic search:** it stores structured information, though PostgreSQL could perform vector search with extensions such as pgvector in a different architecture.

---

# 5. Directory layout and configuration

Your architecture notes describe this organization:

```text
hf-dataset-assistant/
├── .env                         # DATABASE_URL and optional HF_TOKEN
├── requirements.txt             # Python dependencies
├── catalog_db/                  # Phase 1: PostgreSQL helpers
│   ├── __init__.py
│   ├── schema.sql
│   ├── connection.py
│   ├── hub_row.py
│   ├── repository.py
│   └── index_queries.py          # Phase 2 index + Phase 3 hydration SQL
├── catalog_search/              # Phase 2 search + Phase 3 hydration/health
│   ├── __init__.py
│   ├── catalog.py
│   ├── store.py
│   ├── search.py
│   ├── postgres_catalog.py
│   ├── hydrate.py                # Postgres lookup + LLM context formatting
│   └── health.py                 # Chroma preflight before chat
├── catalog_agent/               # Phase 3: LangGraph agent
│   ├── __init__.py
│   └── graph.py
├── scripts/
│   ├── init_postgres_schema.py
│   ├── sync_hub_to_postgres.py
│   ├── run_daily_sync.sh
│   ├── verify_postgres_catalog.py
│   ├── build_vector_index.py
│   ├── semantic_search.py
│   └── chat_agent.py             # Phase 3 CLI
└── data/
    └── chroma/                  # Persistent Chroma vector index
```

## 5.1 Example `.env`

```dotenv
DATABASE_URL=postgresql://YOUR_USER:YOUR_PASSWORD@localhost:5432/YOUR_DATABASE
HF_TOKEN=YOUR_OPTIONAL_HUGGING_FACE_TOKEN
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

- **`DATABASE_URL`**: Phase 1 sync, Phase 2 index build, **Phase 3 hydration** after Chroma retrieval.
- **`HF_TOKEN`**: optional authentication for permitted Hub API calls and rate limits.
- **`OPENAI_API_KEY`**: **required for Phase 3** (`chat_agent.py`).
- **`OPENAI_MODEL`**: optional; defaults to `gpt-4o-mini` in `AgentSettings`.

Do not commit `.env` or put real passwords/tokens in screenshots, examples, or GitHub.

## 5.2 What does `requirements.txt` do?

It lists Python libraries to install. The dependency names recorded for this expanded project include `huggingface_hub`, `psycopg`, `chromadb`, `sentence-transformers`, `torch`, and supporting packages. See the actual current `requirements.txt` for exact version bounds.

---

# 6. Phase 1 deep dive — Hugging Face to PostgreSQL

## 6.1 What is ETL?

**ETL = Extract → Transform → Load.**

- **Extract:** Ask Hugging Face for dataset metadata using the Hub API.
- **Transform:** Convert Hub objects to database-friendly strings, numbers, dates, and JSON values.
- **Load:** Insert/update records in PostgreSQL.

You are collecting **metadata only**. `siblings_json` represents a *file list*, not downloaded file contents. `card_json` represents dataset card metadata, not the actual data rows.

## 6.2 What exactly is stored?

Your `datasets` table stores one row per repository, with fields noted in your materials such as:

| Field | Meaning |
|---|---|
| `repo_id` | Unique Hugging Face `owner/name` repository ID |
| `description` | Text explaining the dataset |
| `tags` | Hub tags/classifications |
| `downloads` | Hub-reported download count |
| `card_json` | Available dataset card metadata in JSON form |
| `siblings_json` | Filenames and available file metadata |
| `raw_hub_json` | Extra Hub metadata retained in JSON form |
| `last_modified` | Last modification time reported by Hub |
| `hub_synced_at` | When this record was synchronized locally |

Other fields may exist. Check `catalog_db/schema.sql` for the exact authoritative column list and data types.

The `sync_runs` table records jobs with fields including `mode`, `status`, `rows_upserted`, `started_at`, `finished_at`, and `error_message`.

## 6.3 What does UPSERT mean?

**UPSERT = insert a record if new; update the existing record if the same key already exists.**

Illustrative SQL:

```sql
INSERT INTO datasets (repo_id, downloads)
VALUES ('example/salary-dataset', 12000)
ON CONFLICT (repo_id)
DO UPDATE SET downloads = EXCLUDED.downloads;
```

If `example/salary-dataset` already exists, update it; don't create a duplicate. This is *conceptual SQL*, not the entire actual project's `UPSERT_DATASET` statement.

## 6.4 Why do you have two tables?

- **`datasets`:** *What information do we currently know about each dataset?*
- **`sync_runs`:** *When did we attempt to refresh this information, and did the job succeed?*

The second helps you detect interrupted/failed jobs and determine the cutoff for later incremental syncs.

---

# 7. Phase 1 file and function reference

For every function, remember **input → job → output**.

## 7.1 `catalog_db/schema.sql`

| Item | Input | Job | Output |
|---|---|---|---|
| `datasets` table | SQL schema | Stores one row per Hub dataset | Relational catalog |
| DB indexes | Table fields | Speed up certain searches/sorts | More efficient queries |
| `sync_runs` table | SQL schema | Logs sync execution | Synchronization history |

Your notes specify indexes used for `downloads`, `last_modified` and tags (including a GIN index for tags).

## 7.2 `catalog_db/connection.py`

| Function | Input | Output | One-line meaning |
|---|---|---|---|
| `get_database_url()` | `DATABASE_URL` environment value | Connection URL string | Find the address and credentials for Postgres |
| `connect()` | Connection URL from helper | `psycopg` connection | Open a communication channel to Postgres |

**Mini-flow:** `.env` → `get_database_url()` → `connect()` → PostgreSQL.

## 7.3 `catalog_db/hub_row.py`

| Function | Input | Output | One-line meaning |
|---|---|---|---|
| `_json_safe(value)` | Nested Hub/Python value | JSON-safe value | Convert dates and other values to serializable form |
| `_description_and_license(info)` | Hub dataset info | Description and license | Extract these fields; fall back to card if needed |
| `_siblings_json(info)` | Hub dataset info | JSON-compatible file-list data | Capture repo file metadata, not file contents |
| `dataset_info_to_row(info)` | `DatasetInfo` object | Dictionary matching `datasets` columns | The main Hub-to-Postgres mapper |

**Example (illustrative):**

```text
Hub object: DatasetInfo(id='example/salary-data', downloads=12000, ...)
        ↓ dataset_info_to_row(info)
DB-ready dict: {repo_id: 'example/salary-data', downloads: 12000, ...}
```

Your notes say descriptions are capped at **8,000 characters** at this mapping stage. This is **different** from the text limit applied later during embedding.

## 7.4 `catalog_db/repository.py`

| Function/constant | Input | Output | One-line meaning |
|---|---|---|---|
| `UPSERT_DATASET` | SQL constant | SQL command | Insert new repository row or update existing one |
| `apply_schema(conn)` | PostgreSQL connection | Database schema changes | Create missing tables/indexes |
| `upsert_dataset(conn, row)` | Connection + one DB-ready dict | Insert/update executed | Save one dataset's metadata |
| `start_sync_run(conn, mode)` | Connection + sync mode | `run_id` | Start a job log with `running` status |
| `finish_sync_run(conn, run_id, ...)` | Connection + run ID + results | Updated log | Mark success/failure and counts |
| `count_datasets(conn)` | Connection | Count | How many catalog records exist? |
| `catalog_stats(conn)` | Connection | Statistics dictionary | Counts of total/described/private rows |
| `last_successful_sync_finished_at(conn)` | Connection | Timestamp or missing value | Watermark for daily sync |
| `last_sync_run(conn)` | Connection | Last sync log dictionary | Inspect most recent attempt |

## 7.5 `catalog_db/index_queries.py` — the bridge to Phase 2 and Phase 3

| Function | Input | Output | Why it matters |
|---|---|---|---|
| `iter_index_rows(conn, ...)` | Connection + filters such as `require_description`, `min_downloads`, `limit` | Row iterator | Streams only records you want to embed |
| `count_index_rows(conn, ...)` | Same filters | Integer count | Lets you know/print index-build progress |
| `fetch_datasets_by_repo_ids(conn, repo_ids)` | DB connection + list of repo IDs | Full matching DB records | **Phase 3:** `hydrate_hits()` loads rich LLM context after Chroma retrieval |

An **iterator** lets you process records one at a time instead of loading hundreds of thousands into one Python list. This is valuable for memory use.

## 7.6 `catalog_db/__init__.py`

Re-exports commonly used helpers such as `connect`, `get_database_url`, `apply_schema`, `upsert_dataset`, `start_sync_run`, and `finish_sync_run`. It makes package imports shorter; it isn't a new pipeline stage.

## 7.7 `scripts/sync_hub_to_postgres.py` — main Phase 1 entry point

| Function or object | Input | Output / role |
|---|---|---|
| `MODE_PRESETS` | Mode name | Preset settings for initial/daily/full |
| `parse_args()` | CLI flags | Parsed argument object |
| `_as_utc(dt)` | Date/time | UTC-normalized date/time |
| `_parse_since(value)` | ISO date string | Parsed UTC date/time |
| `resolve_sync_params(args)` | CLI arguments + presets | Limit, sort, direction, mode name, incremental flag |
| `resolve_incremental_watermark(conn, ...)` | Connection + optional `--since` | Timestamp cutoff for daily mode |
| `main()` | Environment + CLI | Runs the complete sync and records its outcome |

**Function execution/call chain from your notes:**

```text
sync_hub_to_postgres.main()
  ├─ load_dotenv / parse_args / resolve_sync_params
  ├─ connect() / apply_schema()
  ├─ [daily] resolve_incremental_watermark()
  │    └─ last_successful_sync_finished_at()
  ├─ start_sync_run()
  ├─ HfApi().list_datasets(full=True, ...)
  │    └─ for each eligible dataset:
  │         ├─ dataset_info_to_row(info)
  │         └─ upsert_dataset(conn, row)
  └─ finish_sync_run() / count_datasets()
```

**Why `full=True`?** In your supplied notes, the full-sync listing requests fuller metadata from the Hub. This is not the same as downloading the underlying dataset files.

## 7.8 Other Phase 1 commands

| File | Main call | Purpose |
|---|---|---|
| `scripts/init_postgres_schema.py` | `main()` → `apply_schema()` | Create/verify tables once |
| `scripts/verify_postgres_catalog.py` | `catalog_stats()` + `last_sync_run()` | Check catalog health, recent job, top records |
| `scripts/run_daily_sync.sh` | Shell wrapper | Activate environment and run `--mode daily` |

---

# 8. Phase 1 sync modes and the watermark

## 8.1 The three modes

| Mode | What it does | When to use |
|---|---|---|
| `initial` | First approximately 500 popular records | Fast bootstrap/test |
| `full` | Entire available public listing (no preset row limit) | First comprehensive catalog build or full reconciliation |
| `daily` | Incremental new/modified records since cutoff | Regular refresh |

Examples:

```bash
python scripts/sync_hub_to_postgres.py --mode initial
python scripts/sync_hub_to_postgres.py --mode full
python scripts/sync_hub_to_postgres.py --mode daily
```

## 8.2 What is a watermark?

**Watermark = the date/time of your last successful synchronization, used to decide which newer changes to process.**

Example:

```text
Last successful sync finished: Nov 1 at 10:00 AM
Current daily sync:          Nov 2 at 10:00 AM
Target:                       repositories modified after cutoff
```

Your notes specify this logic: explicit `--since` if supplied; otherwise last successful sync's finished time; otherwise a fallback to the preceding 24 hours. The daily loop stops when `last_modified <= watermark`, so this relies on the Hub listing being sorted appropriately.

**Caveats:** A time-based listing can miss late-arriving updates or deletions if sort/filter semantics differ from assumptions. Periodic full reconciliation and explicit deleted-repository handling are good production improvements; your notes do **not** establish that they already exist.

## 8.3 Why track sync status?

A typical successful run log is conceptually:

```text
mode: full
status: success
rows_upserted: ...
started_at: ...
finished_at: ...
error_message: null
```

If the process fails, the error is recorded and a failed run should not be treated as the source of the next successful watermark.

---

# 9. Phase 2 deep dive — PostgreSQL to ChromaDB

## 9.1 Why Phase 2 exists

Plain SQL keyword filtering can match exact words, but isn't automatically aware of paraphrases.

Example:

- User: “employee compensation data”
- Dataset description: “salary records for technology workers”

A semantic embedding model may represent these phrases as similar, enabling retrieval even without identical wording.

## 9.2 What is a vector?

A vector is a list of numbers, such as `[0.21, -0.15, 0.63, ...]`. Those numbers act like coordinates in a high-dimensional meaning space.

The numbers themselves are *not* human-readable labels. Their **relative arrangement** matters: related text tends to have embeddings close to each other under a chosen similarity measure.

## 9.3 Why only 438,350 out of ~1.1M?

The build described in your notes selected **eligible public rows with descriptions**. It did **not** embed every database record. This is an indexing rule, not a sign that the unindexed records were downloaded or deleted.

## 9.4 What exactly gets stored in Chroma?

Conceptually, each Chroma entry has:

```text
ID:          example__salary-data
Document:    "Dataset: example/salary-data\nTitle: ...\nDescription: ..."
Embedding:   384-number vector produced by MiniLM
Metadata:    repo_id, title, URL, tags, license, downloads, short snippet
```

**Document vs metadata:** The document is what the embedding model reads; metadata is stored alongside the vector so you can identify and display the match.

---

# 10. How MiniLM makes embeddings internally

Model: **`all-MiniLM-L6-v2`**, a pretrained Sentence Transformer that runs **locally** on your Mac using the ML stack.

## 10.1 In everyday language

**MiniLM is a meaning translator.** It takes a piece of text and converts it to a numerical fingerprint. That fingerprint lets a search engine compare text meaning.

## 10.2 Internal pipeline

```mermaid
flowchart TD
    A["Text: Employee compensation data"] --> B["Tokenizer: split into model tokens"]
    B --> C["Map tokens to token IDs"]
    C --> D["Embedding layer: ID to initial vectors"]
    D --> E["Transformer: tokens influence one another"]
    E --> F["Pooling: combine contextual token vectors"]
    F --> G["One 384-dimensional sentence embedding"]
```

- **Tokenizer:** Converts text into pieces (tokens), then numerical token IDs.
- **Embedding layer:** Converts token IDs into *initial token vectors* using learned model weights.
- **Transformer:** Uses attention and feed-forward layers to incorporate context. “Bank” near “loan” can be represented differently from “bank” near “river.”
- **Pooling:** Combines processed token vectors into one vector for the whole input (for this model, mean pooling is used).
- **Output:** A vector of **384 numbers**. **Not** a generated sentence; **not** a search result.

## 10.3 The 3,500-character and 40-tag limits

Your `catalog_row_to_document()` prepares a concise text document:

```text
Dataset: owner/name
Title: owner/name
Description: <at most 3,500 characters>
Tags: <up to 40 tags>
License: ...
```

**Why limit it?** To reduce unnecessary text and make indexing manageable.

**Important:** MiniLM also has a **token limit** (the commonly used `all-MiniLM-L6-v2` Sentence Transformers configuration limits model input to 256 tokens). **3,500 characters is not the same as 256 tokens.** A long document may exceed the model's input window, and the trailing information may get truncated.

**Effect:** A key idea near the end of a description might never influence the embedding. Future improvement: select meaningful fields more carefully or chunk longer texts into multiple embeddings.

## 10.4 MiniLM input/output vs Chroma input/output

| Component | Input | Output |
|---|---|---|
| MiniLM at **index time** | Catalog document text | 384-dimensional dataset vector |
| Chroma at **index time** | Vector + ID + text + metadata | Persistent indexed entry |
| MiniLM at **search time** | User query text | 384-dimensional query vector |
| Chroma at **search time** | Query vector (or text that it passes to MiniLM) | Closest entries, metadata, distance scores |

**Golden rule:** Always use a **compatible embedding model** for *both* the stored documents and user queries. If you switch from MiniLM to OpenAI embeddings, rebuild the index and query with the new model.

---

# 11. Phase 2 file and function reference

## 11.1 `catalog_search/catalog.py`

Constants recorded in your notes:

| Constant | Value | Meaning |
|---|---|---|
| `COLLECTION_NAME` | `hf_datasets` | Chroma collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `DEFAULT_CHROMA_PATH` | `data/chroma/` | Persistent storage folder |
| `DESCRIPTION_SNIPPET_LEN` | 400 characters | Short displayed description |
| `EMBED_DESCRIPTION_MAX_LEN` | 3,500 characters | Limit used to build embeddable description text |
| `LLM_DESCRIPTION_MAX_LEN` | 2,800 characters | Separate later-phase formatting limit |

| Function | Input | Output | Job |
|---|---|---|---|
| `chroma_id_from_repo(repo_id)` | `owner/name` | `owner__name` | Stable index entry ID |
| `parse_tags(tags_cell)` | Tags from JSON string or list | `list[str]` | Normalize tags for text/metadata |
| `_repo_id(row)` | Postgres row dict | Repo ID | Supports either `repo_id` or `id` key |
| `catalog_row_to_document(row)` | Postgres row | String | Build text MiniLM embeds: ID, title, description, tags, license |
| `catalog_row_to_metadata(row)` | Postgres row | Dict | Build Chroma sidecar metadata (URL, downloads, snippet, etc.) |

**Example (fictional dataset):**

```text
Postgres row
   ├─ catalog_row_to_document() ─> "Title: ... Description: ... Tags: ..."
   └─ catalog_row_to_metadata() ─> {repo_id, url, downloads, ...}
                                      ↓
                            stored together in Chroma
```

## 11.2 `catalog_search/store.py`

| Function | Input | Output | Job |
|---|---|---|---|
| `get_embedding_function()` | Model constant | Chroma-compatible embedding function | Wrap Sentence Transformer MiniLM |
| `get_chroma_client(persist_path)` | `data/chroma/` or override | Persistent Chroma client | Connect to on-disk vector DB; telemetry off |
| `get_collection(client, create=...)` | Client + create flag | Chroma collection | Open/create `hf_datasets` with **cosine** distance and embedding function |

**Hidden automation:** `collection.add(documents=...)` can call the embedding function automatically; `collection.query(query_texts=...)` can use the **same** embedding function to embed the user question.

## 11.3 `catalog_search/search.py`

| Name | Input | Output | Job |
|---|---|---|---|
| `SearchHit` | N/A (type definition) | Dict shape | Describes returned fields: rank, repo_id, URL, title, snippet, distance, similarity, etc. |
| `distance_to_similarity(distance)` | Numeric distance | Clipped `1 - distance` | A convenient display score, not probability |
| `semantic_search(query, k, chroma_path)` | Query text + result limit + index path | `list[SearchHit]` | Embed query, search Chroma, format top matches |
| `search_model_name()` | None | Model name string | Display which embedding model is configured |

## 11.4 `catalog_search/postgres_catalog.py`

| Function | Input | Output |
|---|---|---|
| `load_catalog_from_postgres(...)` | Connection/settings | List of rows; can be memory-heavy |
| `iter_catalog_from_postgres(...)` | Connection/settings | Iterator of rows |
| `postgres_index_count(...)` | Filters | Count of matching rows |

Your notes specify that the *main full build* uses `build_from_postgres()` and `iter_index_rows()` directly, rather than loading everything into a list.

## 11.5 `catalog_search/hydrate.py` (Phase 2 search uses Chroma only; **Phase 3 uses this**)

| Function | Input | Output | Job |
|---|---|---|---|
| `hydrate_hits(hits)` | `list[SearchHit]` from Chroma | `dict[repo_id → row]` | `fetch_datasets_by_repo_ids()` in Postgres |
| `format_hits_for_llm(hits, details)` | Hits + hydrated rows | Multi-dataset plain text | Full description (up to `LLM_DESCRIPTION_MAX_LEN`), tags, downloads, likes, gated, URL |

**Hydration analogy:** Chroma gives you a library-card number; hydration uses that number to bring back the **full catalog record** so the LLM does not rely on a 400-character snippet alone.

## 11.6 `catalog_search/__init__.py`

Convenience re-exports of constants and helper functions. It doesn't itself embed or search documents.

## 11.7 `scripts/build_vector_index.py`

| Function | Input | Output | Job |
|---|---|---|---|
| `parse_args()` | CLI flags | Build configuration | Limit, min downloads, description requirement, Chroma path |
| `_index_batch(collection, ...)` | Collection + ≤64 records | Chroma additions/progress | Send one batch to Chroma for embedding and storage |
| `build_from_postgres(...)` | DB connection/settings + collection | Populated collection | Iterate Postgres rows and index in batches |
| `main()` | CLI/environment | Rebuilt index + completion summary | Delete old collection, recreate, build from Postgres only |

**Main call chain:**

```text
build_vector_index.main()
  ├─ parse_args() / load environment
  ├─ get_chroma_client()
  ├─ delete old collection + get_collection(create=True)
  └─ build_from_postgres()
       ├─ count_index_rows()
       ├─ iter_index_rows()
       ├─ catalog_row_to_document(row)
       ├─ catalog_row_to_metadata(row)
       ├─ chroma_id_from_repo(repo_id)
       └─ _index_batch()  [up to 64 items]
            └─ collection.add(documents=..., ids=..., metadatas=...)
                 └─ Chroma calls MiniLM and saves vectors
```

**Why batches?** You process manageable groups rather than asking the model to embed hundreds of thousands of texts at once.

**Very important:** According to your notes, `main()` **deletes the previous collection before recreating it**. That means this is currently a **full rebuild**, not a changed-record-only update. Plan for build time and avoid accidentally deleting your only working index without a recovery strategy.

## 11.8 `scripts/semantic_search.py`

| Function | Input | Output |
|---|---|---|
| `parse_args()` | Query string, `-k`, `--chroma-path`, `--json` | Parsed arguments |
| `main()` | Parsed arguments | Search results printed as readable text or JSON |

**Call chain:**

```text
semantic_search.py main()
  └─ catalog_search.search.semantic_search(query, k)
       ├─ get_chroma_client() + get_collection()
       ├─ collection.query(query_texts=[query], n_results=k)
       │    ├─ MiniLM embeds the question
       │    └─ Chroma retrieves the nearest vectors
       └─ distance_to_similarity() + SearchHit list
```

## 11.9 `catalog_search/health.py` (Phase 3 preflight)

| Function | Input | Output | Job |
|---|---|---|---|
| `chroma_index_count(chroma_path)` | Path | `int` | Number of vectors in `hf_datasets` |
| `chroma_index_ready(chroma_path)` | Path | `(bool, str)` | User-facing readiness message |

---

# 12. How semantic search works, step by step

**Example question:** “Find compensation datasets.”

**Suppose** an indexed dataset description contains “employee salary records.” The two texts aren't identical, but may be close in the semantic embedding space.

1. **Accept query:** `"Find compensation datasets"`.
2. **Open collection:** `get_chroma_client()` → `get_collection()`.
3. **Embed question:** Chroma uses MiniLM on the query text, producing **384 numbers**.
4. **Compare vectors:** Chroma finds close embeddings among the **438,350** indexed dataset records.
5. **Get matches:** Chroma returns their IDs, stored metadata, and vector distances.
6. **Format:** `semantic_search()` creates a `SearchHit` dictionary per result.
7. **Display:** CLI prints repo IDs, links, snippets, and similarity-like scores.

Illustrative result:

```json
{
  "rank": 1,
  "repo_id": "example/salary-data",
  "url": "https://huggingface.co/datasets/example/salary-data",
  "description_snippet": "Employee salary statistics across industries",
  "distance": 0.09,
  "similarity": 0.91
}
```

**These are made-up example IDs and scores, not output from your installed index.**

## 12.1 What is cosine similarity?

It looks at the **direction** of two vectors rather than their literal text. If the vectors point in similar directions, the texts are often semantically related.

Your Chroma collection uses **cosine distance**. Lower distance typically means a better vector match. Your UI/helper turns a distance into a display score using `max(0, min(1, 1 - distance))`.

**Warning:** This score is **not** “91% chance this is a correct dataset.” Semantic similarity is only one relevance signal and does not verify license, date range, row counts, or columns.

## 12.2 Does Chroma understand language?

Not independently. **MiniLM learned a mapping from text to embeddings.** Chroma finds nearby vectors using numerical search. Chroma may *call* the MiniLM function automatically, but the jobs remain separate.

---

# 13. The full journey of one dataset

This example connects *everything*.

**Fictional repository:** `example/salary-data` — “Employee salary records across industries.”

### When collecting metadata

```text
1. Hugging Face API returns DatasetInfo for example/salary-data
2. dataset_info_to_row(info) creates the database dictionary
3. upsert_dataset(conn, row) saves/updates Postgres datasets row
4. sync_runs records the synchronization job
```

### When building the index

```text
5. iter_index_rows(conn) yields the row if it meets indexing criteria
6. catalog_row_to_document(row) creates descriptive text
7. catalog_row_to_metadata(row) creates sidecar metadata
8. _index_batch() passes document + metadata + ID to Chroma
9. MiniLM turns document into 384 numbers
10. Chroma saves vector, ID, text and metadata
```

### When a user searches

```text
11. User types: "Find employee compensation datasets"
12. semantic_search.py passes query to semantic_search()
13. MiniLM converts query text to a 384-number vector
14. Chroma compares to indexed dataset vectors
15. Chroma returns closest IDs, URLs, snippets and distances
16. CLI prints matching datasets
```

**The Hub API is not queried on every search.** The user searches your **local saved index**.

---

# 14. Command-by-command operating guide

> Run commands from the repository root. Exact dependencies, DB setup and OS-specific shell steps depend on your machine.

## 14.1 Set up the Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure PostgreSQL is installed, running, and reachable using `DATABASE_URL`. The virtual environment separates this project's dependencies from other Python projects.

## 14.2 Initialize schema (once per new DB)

```bash
python scripts/init_postgres_schema.py
```

**Main effect:** `apply_schema()` creates the catalog and sync-log tables/indexes.

## 14.3 Small test sync

```bash
python scripts/sync_hub_to_postgres.py --mode initial
```

**Main effect:** Populate a starter catalog with roughly 500 records.

## 14.4 Full catalog sync

```bash
python scripts/sync_hub_to_postgres.py --mode full
```

**Main effect:** Iterate the accessible public Hub listing and upsert metadata. The reported result for your completed run was **~1.11 million** records.

## 14.5 Verify catalog

```bash
python scripts/verify_postgres_catalog.py
```

**Main effect:** Show statistics and most recent sync status.

## 14.6 Daily sync

```bash
./scripts/run_daily_sync.sh
# Or:
python scripts/sync_hub_to_postgres.py --mode daily
```

**Main effect:** Refresh eligible new/modified metadata in PostgreSQL.

## 14.7 Build the full vector index

```bash
python scripts/build_vector_index.py
```

**Main effect:** Read eligible Postgres rows, embed them in batches, and store them in Chroma. Your recorded full build indexed **438,350** records.

## 14.8 Test semantic search

```bash
python scripts/semantic_search.py "medical imaging" -k 5
python scripts/semantic_search.py "housing price time series" -k 8
python scripts/semantic_search.py "employment data" --json
```

**Main effect:** MiniLM embeds the query and Chroma retrieves the nearest matching dataset descriptions.

## 14.9 Optional smaller rebuilds

```bash
python scripts/build_vector_index.py --limit 500
python scripts/build_vector_index.py --min-downloads 1000
```

**Warning:** These options select *what goes into the newly rebuilt collection*. If your builder deletes the old collection first, `--limit 500` **replaces your large searchable index with a 500-record index**. Don't use a small test rebuild against your main Chroma path unless that is intentional. Prefer a separate `--chroma-path` for experiments.

## 14.10 Phase 3 chat agent

```bash
python scripts/chat_agent.py
python scripts/chat_agent.py -m "datasets for fine-tuning on Python code" -k 10 -v
```

Requires `OPENAI_API_KEY`, running Postgres (`DATABASE_URL`), and a built Chroma index. See [§24 Phase 3 commands](#24-phase-3-commands-and-operating-notes).

---

# 15. Important design decisions and limitations

| Decision | Reason or consequence |
|---|---|
| **Metadata only** | Avoid downloading huge underlying dataset files |
| **PostgreSQL as source of truth** | Keep richer structured fields and sync history |
| **Chroma as separate index** | Efficient semantic similarity search |
| **Local MiniLM** | No per-document OpenAI embedding API fees; model runs on Mac after initial download |
| **384-dimensional embeddings** | Fixed output of selected MiniLM model |
| **Index only eligible public records with descriptions** | Smaller searchable subset; other DB records won't appear in Chroma results |
| **Limit embedded description and tags** | Control input size, but MiniLM token truncation can lose trailing content |
| **Index in batches of 64** | Avoid processing entire corpus at once |
| **Full Chroma rebuild** | Simple but resource-intensive; potential downtime/temporary missing index during rebuild |
| **Search Chroma, not Hub live** | Fast local retrieval but depends on index freshness |
| **No generative model in Phases 1–2** | Search produces hits, not conversational recommendations |

## 15.1 Gaps to remember

**A. Metadata completeness:** Missing description, license, or tags should be treated as unknown, not silently invented.

**B. Coverage:** “Full public Hub listing” means repositories returned and successfully processed by the sync, not proof that every reachable, gated, deleted, or newly created repository was captured at one instant.

**C. Freshness:** New Postgres records aren't automatically reflected in the existing Chroma collection.

**D. Retrieval quality:** Nearest vectors aren't always useful matches; a relevance threshold or reranking may help later.

**E. Structured constraints:** Semantic similarity alone cannot reliably enforce “after 2022,” “commercial use,” “at least 100,000 rows,” or “contains an age column.” You need reliable structured metadata and filtering for that.

**F. Text truncation:** A 3,500-character stored description doesn't guarantee the model sees all 3,500 characters after tokenization.

**G. Full vs incremental sync:** Timestamp watermarks can miss some changes; repository deletions require additional handling not established in the notes.

---

# 16. Troubleshooting and debugging

| Problem | Likely area to check | Why |
|---|---|---|
| `DATABASE_URL` missing | `.env`, `load_dotenv`, `connection.py` | Postgres connection needs DB URL |
| Can't connect to PostgreSQL | Postgres service, URL, credentials, port | `psycopg` requires running reachable DB |
| No tables | `init_postgres_schema.py`, `schema.sql` | Schema may not be initialized |
| Very few records after sync | Hub errors, filters, sync mode, rate limits | `initial` intentionally limits records |
| Daily sync finds no rows | Watermark, Hub sorting, `--since`, sync log | Incremental cutoff determines which records qualify |
| Index count smaller than Postgres count | `count_index_rows`, description requirement, filters | Not every DB record is eligible |
| Model downloads weights on first run | Sentence Transformer model cache | Initial local model setup requires downloading weights |
| Chroma index missing | `data/chroma/`, collection name, build script | Search requires an existing built index |
| Search results stale | Last index build vs latest Postgres sync | Postgres and Chroma refresh separately |
| Search results don't match a strict date/license rule | Semantic search lacks exact filtering | Similarity isn't constraint validation |
| Search output looks irrelevant | Missing/short descriptions, truncation, nearest-neighbor ranking | Vector proximity doesn't guarantee relevance |
| Huge rebuild is slow | Batch embedding hundreds of thousands of rows | Local CPU/GPU workload; full collection recreated |
| Error messages about telemetry | Chroma configuration/version | Telemetry can be disabled without disabling search |

**Practical debugging strategy:** Test one layer at a time: **Can I connect to Postgres? → Are there dataset records? → Can I build a small isolated Chroma index? → Can I search that index? → Can I inspect an actual returned repo ID in Postgres?**

---

# 17. Definitions and common confusion

| Term | Simple definition |
|---|---|
| **Metadata** | Data *about* a dataset: name, description, tags, license, etc. |
| **Repository** | Named set of files and metadata on Hugging Face |
| **Dataset card** | Documentation describing a dataset, where provided |
| **ETL** | Extract, transform, load |
| **API** | Interface that lets software request data from another service |
| **`DatasetInfo`** | Python object describing a Hugging Face repository |
| **Schema** | Database tables, fields, and types |
| **Primary key / unique key** | Identifier preventing multiple rows for same repo ID |
| **UPSERT** | Insert if new, update if existing |
| **Synchronization** | Refresh local stored data from an external source |
| **Watermark** | Timestamp cutoff used to choose newer changes |
| **Iterator / streaming** | Process records incrementally, not all at once |
| **Batch** | Small group processed together, e.g. 64 rows |
| **Tokenizer** | Converts text into model tokens and token IDs |
| **Embedding model** | Converts text into a semantic vector |
| **Sentence Transformer** | Transformer-based embedding model/framework |
| **MiniLM** | Your chosen compact local embedding neural network |
| **Vector** | List of numbers representing position in a meaning space |
| **Dimension** | One number/coordinate in a vector; here there are 384 |
| **Vector database** | Stores/indexes vectors and retrieves nearby ones |
| **Chroma collection** | Named group of embeddings, documents and metadata |
| **Semantic search** | Find conceptually related text, not just exact words |
| **Cosine distance** | Numerical distance based on differences in vector directions |
| **HNSW** | Algorithm/data structure for fast approximate neighbor search |
| **Source of truth** | Authoritative place from which derived copies are made |
| **Hydration** | Load detailed Postgres records using IDs returned by Chroma |
| **RAG** | Retrieve (Chroma) + augment context (Postgres) + generate (OpenAI) — **Phase 3** |
| **LangGraph** | Graph of nodes with state; routes clarify vs search vs answer |
| **Structured output** | LLM returns a typed object (`PlanDecision`) instead of free text |
| **Checkpointer / thread_id** | Keeps conversation history across turns in one CLI session |

## 17.1 Four common confusions

**“Is MiniLM the search engine?”** No. MiniLM creates vector representations; Chroma retrieves similar vectors.

**“Does Chroma create embeddings?”** It can *call the embedding function* on your behalf, but MiniLM is the model generating the vectors.

**“Does Postgres have the vectors?”** In your design, your primary rich metadata is in PostgreSQL and your vector search index is in Chroma. Postgres could be used for embeddings with pgvector, but that is a different design.

**“Is Chroma's answer an AI answer?”** No. In Phase 2, Chroma produces matching repository records. In **Phase 3**, OpenAI generates the conversational explanation **using only hydrated retrieval context**.

---

# 18. What could replace each component?

| Current component | Alternative | What would change? |
|---|---|---|
| Local MiniLM | OpenAI `text-embedding-3-small` | Cloud API, API-key/billing, default 1536-dimensional vectors instead of 384; rebuild embeddings |
| Local MiniLM | Another Sentence Transformer | Different embedding quality/speed/dimensions; rebuild index |
| ChromaDB | FAISS | Search library instead of full vector DB; you'd manage more document metadata yourself |
| ChromaDB | Pinecone | Managed cloud vector database |
| ChromaDB | PostgreSQL + pgvector | Keep vector search in Postgres, simplifying some infrastructure |
| PostgreSQL | Another relational DB | Different SQL driver/schema/sync operations |

**Why MiniLM was sensible here:** You have hundreds of thousands of documents to embed. Local MiniLM has no per-request embedding API charges (though it uses your Mac's compute), and you've already built the index.

**If you switch embedding models:** Re-embed **both** indexed documents and future user queries with the same model. Do not compare 384-dimensional MiniLM vectors directly with 1536-dimensional OpenAI vectors.

---

# 19. Quick revision sheet

## 19.1 The four workflows you must be able to explain

```text
PHASE 1 — CATALOG SYNC
Hugging Face Hub API
    → sync_hub_to_postgres.main()
    → dataset_info_to_row(info)
    → upsert_dataset(conn, row)
    → PostgreSQL datasets
    + sync_runs logs

PHASE 2 — INDEX BUILD
PostgreSQL datasets
    → iter_index_rows(conn)
    → catalog_row_to_document(row)
    → collection.add(...)
    → MiniLM generates vector
    → Chroma hf_datasets collection

PHASE 2 — SEARCH
User search text
    → semantic_search(query)
    → Chroma calls MiniLM to embed query
    → Chroma compares stored vectors
    → top-k SearchHit records
    → terminal output

PHASE 3 — CHAT (one turn)
User message
    → chat_agent.main() → build_agent()
    → plan_node (OpenAI PlanDecision)
    → retrieve_node → semantic_search()
    → answer_node → hydrate_hits() + format_hits_for_llm()
    → OpenAI grounded answer → AIMessage
```

## 19.2 One line for each key file

| File | One-line memory aid |
|---|---|
| `schema.sql` | Defines catalog and sync-log tables |
| `connection.py` | Connect Python to PostgreSQL |
| `hub_row.py` | Convert Hub info to DB-ready data |
| `repository.py` | Insert/update and query PostgreSQL |
| `index_queries.py` | Select records for embedding |
| `sync_hub_to_postgres.py` | Collect and sync all public Hub metadata |
| `catalog_search/catalog.py` | Make text and metadata from a DB row |
| `catalog_search/store.py` | Configure MiniLM and Chroma |
| `build_vector_index.py` | Embed and index eligible datasets |
| `catalog_search/search.py` | Query Chroma and return matches |
| `semantic_search.py` | Search from the terminal |
| `catalog_search/health.py` | Preflight Chroma before chat |
| `catalog_search/hydrate.py` | Postgres enrichment for LLM context |
| `catalog_agent/graph.py` | LangGraph plan → retrieve → answer |
| `scripts/chat_agent.py` | Interactive Phase 3 CLI |

## 19.3 Most important input/output pairs

| Operation | Input | Output |
|---|---|---|
| Hub listing | API request | DatasetInfo objects |
| `dataset_info_to_row` | DatasetInfo | DB-ready dict |
| `upsert_dataset` | DB row dict | Stored/updated Postgres row |
| `catalog_row_to_document` | Postgres row | Searchable text |
| MiniLM encode | Text | 384-number vector |
| Chroma `collection.add` | IDs, text, metadata | Stored/indexed entries (embedding generated through configured function) |
| Chroma `collection.query` | Query text (or vector) + `k` | Nearest IDs/metadata/distances |
| `semantic_search` | Query text + result limit | SearchHit list |

**Ultimate memory aid:** **Hub supplies facts → Postgres stores facts → MiniLM creates coordinates → Chroma finds nearby coordinates → Python returns matching dataset information.**

---

# 20. Self-test and interview-style questions

Use these to check whether you can explain the project without looking at the code.

1. **Why aren't you downloading all Hugging Face dataset files?** Because the project is discovering and recommending datasets; repository metadata is enough for catalog-level search, and underlying data files can be huge.
2. **Why PostgreSQL?** It's the structured source of truth for the rich catalog and synchronization history.
3. **What's the difference between `datasets` and `sync_runs`?** One holds dataset facts; the other logs synchronization jobs.
4. **What is an upsert?** Insert new rows; update existing ones based on a unique identifier.
5. **What happens during `--mode full`?** The script iterates the accessible Hub listing, maps API entries, and upserts them to Postgres.
6. **What happens during `--mode daily`?** The script uses an incremental watermark to target new/updated records.
7. **Why have `iter_index_rows()`?** To stream eligible DB records for indexing instead of loading the entire corpus into memory.
8. **Why do PostgreSQL and Chroma have different counts?** The Chroma index includes only eligible records, such as public rows with descriptions.
9. **What is `catalog_row_to_document()` for?** To turn database columns into one text representation for MiniLM.
10. **What's the difference between document text and Chroma metadata?** Text is embedded; metadata identifies and describes matches.
11. **What is MiniLM's input and output?** Text in; a 384-dimensional vector out.
12. **Does MiniLM search the database?** No. Chroma searches stored vectors.
13. **Why does `collection.add()` work without `embeddings=`?** The Chroma collection uses a configured MiniLM embedding function.
14. **What happens at query time?** Same MiniLM converts the question into a vector; Chroma finds nearby indexed vectors.
15. **What does cosine distance mean?** A numerical measure of how different vector directions are; lower is usually closer.
16. **Is `similarity=0.9` 90% confidence?** No. It's a transformed similarity/distance measure.
17. **If you sync Postgres today, will Chroma automatically refresh?** No. Your current pipeline requires an explicit index rebuild.
18. **What's dangerous about running `build_vector_index.py --limit 500`?** With the documented delete-and-rebuild behavior, it can replace the large index with only 500 entries.
19. **Could you use OpenAI embeddings instead of MiniLM?** Yes, but it requires API use and rebuilding/querying the index with the replacement model.
20. **Where does the LLM fit?** In Phase 3: plan (clarify vs search) and answer (recommendations). Phases 1–2 work without OpenAI.

---

# 21. Phase 3 deep dive — conversational discovery

## 21.1 What problem Phase 3 solves

**Phase 2** returns a **list** of similar datasets. Users must read snippets and decide themselves.

**Phase 3** adds a **conversation layer**:

- Ask a **clarifying question** when the request is too vague.
- Turn messy chat into a **focused search query** for Chroma.
- **Retrieve** top-k matches (same as Phase 2).
- **Hydrate** full metadata from Postgres (longer descriptions, tags, stats).
- **Answer** in plain English with repo IDs, reasons, and URLs — **only** for retrieved datasets.

This is **RAG** (Retrieval-Augmented Generation): retrieval is not optional; the LLM must not invent dataset names.

## 21.2 What Phase 3 does *not* do

- Does **not** call Hugging Face live search on every message (uses **local index**).
- Does **not** download dataset files or show sample rows from Parquet.
- Does **not** re-embed queries with OpenAI (still **MiniLM** via Chroma).
- Does **not** automatically refresh Chroma when Postgres syncs.

## 21.3 The LangGraph graph (mental model)

Three **nodes**, one **conditional edge** after plan:

| Node | Model | Purpose |
|------|--------|---------|
| **plan** | OpenAI (structured `PlanDecision`, temp 0) | `clarify`, **`search`** (new Chroma), or **`compare_previous`** (reuse `last_hits`) |
| **retrieve** | MiniLM + Chroma (local) | `semantic_search(query, k)` → `hits` (skipped on compare path) |
| **answer** | OpenAI (temp 0.2) | Hydrate + `format_hits_for_llm` → recommend or compare; sets **`last_hits`** |

**Routes:**

- `plan` → if clarify → **END** (assistant message already added).
- `plan` → if **compare_previous** and `last_hits` present → **answer** → **END** (no retrieve).
- `plan` → if search → **retrieve** → **answer** → **END** (`last_hits` updated).

**State (`AgentState`):**

- `messages` — conversation (accumulated with `add_messages`).
- `search_query` — string used for Chroma.
- `hits` — `list[SearchHit]` for the current turn.
- `last_hits` — `list[SearchHit]` from the **last answer** (for follow-ups like “which is best for ML?”).
- `route` — `"end"`, `"retrieve"`, or `"compare"`.
- `answer_mode` — `"recommend"` or `"compare"` (picks answer system prompt).

**Memory:** `MemorySaver` checkpointer + `thread_id` in `chat_agent.py` so multi-turn REPL keeps context.

## 21.4 Planner vs answer LLM (two calls per search turn)

1. **Planner** uses `.with_structured_output(PlanDecision)` — forces JSON-like fields: `action`, `message`, `search_query`.
2. **Answer** uses normal chat — system prompt + user block containing **retrieved datasets text**.

Why two steps? **Separation of concerns:** routing/clarification vs grounded writing. One conflated prompt is harder to tune and easier to hallucinate.

## 21.5 Grounding rules (what keeps answers honest)

- `ANSWER_SYSTEM` instructs: recommend **only** from retrieved list.
- Context string is built from **actual** Chroma hits + Postgres rows.
- If retrieval is weak, model should say so and suggest refining the ask.

## 21.6 Follow-up questions (“Which one is best for machine learning?”)

**Problem (old behavior):** Every search turn ran Chroma again, so turn 2 might return **different** datasets than turn 1.

**Fix (implemented):**

1. After each answer, `answer_node` saves **`last_hits`** (the same `SearchHit` list used for that reply).
2. `MemorySaver` + same `thread_id` keep **`last_hits`** and **messages** across turns.
3. Planner sees transcript **plus** `Datasets recommended in the last search turn: …`.
4. For follow-ups, planner returns **`compare_previous`** → graph routes **plan → answer** (skips `retrieve`).
5. **`COMPARE_ANSWER_SYSTEM`** asks OpenAI to rank/compare using hydrated Postgres metadata only.

**Example:** User asks for Canadian salary datasets → search → answer lists A, B, C. User asks “Which is best for ML?” → compare path re-hydrates A, B, C → compares without a new vector search.

If `compare_previous` is chosen but `last_hits` is empty, the graph **falls back** to a normal `search`.

---

# 22. Phase 3 file and function reference

## 22.1 `catalog_agent/graph.py`

| Name | Type | Job |
|------|------|-----|
| `_PLAN_RULES` | string | Shared routing rules for the planner |
| `_plan_system(catalog_note)` | function | Builds planner system prompt with live catalog size note |
| `ANSWER_SYSTEM` | string | Grounded recommendation instructions |
| `PlanDecision` | Pydantic model | `action: clarify\|search\|compare_previous`, `message`, `search_query` |
| `COMPARE_ANSWER_SYSTEM` | string | Follow-up comparison prompt (rank/trade-offs, no invented columns) |
| `_format_last_hits_for_planner(last_hits)` | function | Repo id list shown to planner for follow-up routing |
| `AgentState` | TypedDict | Graph state fields |
| `AgentSettings` | dataclass | `chroma_path`, `top_k`, `openai_model`, `catalog_note`, `verbose_retrieval` |
| `_latest_user_text(messages)` | function | Last `HumanMessage` content |
| `_conversation_transcript(messages, limit=12)` | function | Plain-text history for planner |
| `build_agent(settings)` | function | Wires nodes, compiles graph with `MemorySaver` |
| `planner_structured()` | inner | Cached `ChatOpenAI` + structured output |
| `answer_llm()` | inner | Cached `ChatOpenAI` for final reply |
| `plan_node(state)` | node | Invoke planner; clarify → END or set `search_query` |
| `retrieve_node(state)` | node | `semantic_search()` with `settings.top_k` |
| `answer_node(state)` | node | `hydrate_hits` + `format_hits_for_llm` + answer LLM |
| `route_after_plan(state)` | router | END if `route == "end"`, else `"retrieve"` |

**Call chain inside `build_agent` (search path):**

```text
START → plan_node
  → route_after_plan → retrieve_node
  → answer_node → END
```

## 22.2 `catalog_agent/__init__.py`

Re-exports `AgentSettings`, `build_agent`.

## 22.3 `catalog_search/health.py`

| Function | Input | Output | Job |
|----------|--------|--------|-----|
| `chroma_index_count(chroma_path)` | Path | `int` | `collection.count()` |
| `chroma_index_ready(chroma_path)` | Path | `(bool, str)` | Fail with helpful message or `"Chroma index ready (N datasets)."` |

Used by **`chat_agent.py`** before `build_agent()` so users do not hit obscure Chroma errors mid-chat.

## 22.4 `catalog_search/hydrate.py` (Phase 3 critical path)

See §11.5 for function table. Depends on `fetch_datasets_by_repo_ids()` in `index_queries.py`.

## 22.5 `catalog_db/index_queries.py` — hydration SQL

| Function | Phase 3 use |
|----------|-------------|
| `fetch_datasets_by_repo_ids(conn, repo_ids)` | Single SQL `WHERE repo_id = ANY(...)` for top-k IDs |

Returns full `description`, `tags`, `downloads`, `likes`, `gated`, etc.

## 22.6 `scripts/chat_agent.py`

| Function | Input | Output | Job |
|----------|--------|--------|-----|
| `parse_args()` | CLI | Namespace | `-m`, `-k`, `--thread-id`, `--chroma-path`, `-v` |
| `print_assistant(messages)` | Message list | stdout | Print latest `AIMessage` |
| `main()` | env + CLI | exit code | Load `.env`, check `OPENAI_API_KEY`, `chroma_index_ready`, `build_agent`, REPL or single turn |
| `run_turn(user_text)` | inner | int | `agent.invoke({"messages": [HumanMessage(...)]}, config)` |

**Config:** `{"configurable": {"thread_id": thread_id}}` — same `thread_id` preserves multi-turn state.

---

# 23. How one chat turn works, step by step

**User:** “I need EEG brain datasets for sleep staging.”

1. **`chat_agent.main()`** loads `.env`, verifies `OPENAI_API_KEY`, runs **`chroma_index_ready()`** (e.g. 438,350 datasets).
2. **`build_agent(AgentSettings(...))`** returns compiled graph.
3. **`agent.invoke`** with new `HumanMessage`.
4. **`plan_node`:**
   - Builds transcript from last ≤12 messages.
   - OpenAI returns `PlanDecision(action="search", search_query="EEG sleep staging brain datasets")`.
5. **`route_after_plan`** → `"retrieve"`.
6. **`retrieve_node`:**
   - `semantic_search("EEG sleep staging...", k=8)` → 8 `SearchHit` dicts (MiniLM + Chroma).
7. **`answer_node`:**
   - `hydrate_hits(hits)` → up to 8 full Postgres rows.
   - `format_hits_for_llm(hits, details)` → long context block.
   - OpenAI writes answer citing only those repo IDs.
8. **`print_assistant`** shows the reply.

**Clarify path example:** User says “hi” → planner `action=clarify` → `AIMessage` “What kind of datasets…” → **END** (no Chroma, no second LLM call).

---

# 24. Phase 3 commands and operating notes

```bash
source .venv/bin/activate

# Single question
python scripts/chat_agent.py -m "medical imaging chest x-ray datasets" -v

# Interactive (same thread_id for the session unless you pass --thread-id)
python scripts/chat_agent.py

# Retrieve more candidates
python scripts/chat_agent.py -m "French NLP corpora" -k 12
```

| Flag | Meaning |
|------|---------|
| `-m` / `--message` | One shot; no REPL |
| `-k` | Top-k for Chroma (default 8) |
| `-v` / `--verbose` | Print search query + hit count in retrieve |
| `--thread-id` | Stable conversation memory key |
| `--chroma-path` | Override index directory |

**Prerequisites checklist:**

1. Postgres up, `DATABASE_URL` set (hydration).
2. Chroma built (`build_vector_index.py`).
3. `OPENAI_API_KEY` set.

---

# 25. Phase 3 design decisions, costs, and limits

## 25.1 Why LangGraph instead of one big prompt?

- Explicit **branching** (clarify vs search).
- **Testable nodes** (retrieve without answer).
- **Checkpointed** multi-turn memory.

## 25.2 Why hydrate from Postgres after Chroma?

Chroma metadata holds **snippets** (400 chars). Postgres holds **full description** (up to 2800 chars sent to LLM). Retrieval quality still depends on what was **embedded** at index time.

## 25.3 Cost model (per search turn)

| Step | Paid API? |
|------|-----------|
| Plan (OpenAI) | Yes |
| Retrieve (MiniLM/Chroma) | No (local) |
| Hydrate (Postgres) | No |
| Answer (OpenAI) | Yes |

Clarify-only turns: **one** OpenAI call.

## 25.4 Limits / honesty

- Recommendations quality ≤ retrieval quality (garbage in → garbage out).
- Index may be **stale** vs Postgres until rebuild.
- Model may still **over-explain**; grounding reduces but does not eliminate all errors.
- No access control for **gated** datasets beyond mentioning `gated` in context.

---

# 26. Phase 3 troubleshooting

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| `OPENAI_API_KEY is not set` | Missing env | Add to `.env` |
| `Chroma directory not found` | No index | `python scripts/build_vector_index.py` |
| `Agent error: ... connection` | Postgres down / bad `DATABASE_URL` | Start Postgres, fix URL (hydration) |
| Always clarifies, never searches | Vague prompts | Add domain/modality in your message |
| Weird dataset suggestions | Weak semantic match | Rephrase; increase `-k`; rebuild index |
| Answer cites wrong license | Stale or thin metadata | Sync Hub; rebuild index |

---

# 27. Phase 3 self-test questions

21. **What are the three LangGraph nodes?** plan, retrieve, answer.
22. **When does the graph END without searching?** When planner chooses `action=clarify`.
23. **What is `PlanDecision`?** Structured planner output: clarify vs search + fields.
24. **What does `hydrate_hits` do?** Postgres lookup by `repo_id` for Chroma top-k.
25. **Why two OpenAI calls on a search turn?** One to route/rewrite query; one to write grounded answer.
26. **Does Phase 3 use OpenAI embeddings?** No. Still MiniLM through Chroma.
27. **What does `thread_id` do?** Names the checkpointed conversation for multi-turn REPL.
28. **What does `chroma_index_ready` check?** Directory exists, collection exists, count > 0.
29. **Can the LLM recommend a dataset not in `hits`?** It should not; prompts forbid it; enforcement is prompt-based not hard-coded.
30. **After daily Postgres sync, are answers automatically fresher?** Hydration yes for text fields; semantic recall unchanged until Chroma rebuild.
31. **What is `compare_previous`?** Planner action for follow-ups; reuses `last_hits` instead of a new Chroma query.
32. **What is `last_hits`?** SearchHit list saved after each answer for the next compare turn.
33. **When is `retrieve` skipped?** When route is `compare` after `compare_previous`.

---

## Final 30-second explanation (practice saying this aloud)

> “I built a three-stage Hugging Face dataset discovery assistant. Phase 1 synced about 1.1 million public dataset metadata records into PostgreSQL with upsert and sync-run tracking. Phase 2 embedded about 438,000 described datasets with local MiniLM into Chroma for semantic search. Phase 3 is a LangGraph agent: OpenAI plans whether to clarify or search, Chroma retrieves top-k matches with the same local embeddings, PostgreSQL hydrates full metadata for those IDs, and OpenAI writes grounded recommendations with links. PostgreSQL is the source of truth, Chroma is the semantic index, and the LLM only explains what retrieval actually returned.”

---

**End of Phase 1–3 study guide.** Consult repository source files for exact signatures and error handling when implementing changes.
