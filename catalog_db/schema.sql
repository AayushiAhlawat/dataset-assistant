-- Phase 1 (PostgreSQL): Hub dataset metadata catalog

CREATE TABLE IF NOT EXISTS datasets (
    repo_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    author TEXT,
    url TEXT NOT NULL,
    license TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'Hugging Face',
    downloads BIGINT,
    downloads_all_time BIGINT,
    likes BIGINT,
    trending_score BIGINT,
    private BOOLEAN NOT NULL DEFAULT FALSE,
    gated TEXT,
    disabled BOOLEAN,
    sha TEXT,
    paperswithcode_id TEXT,
    created_at TIMESTAMPTZ,
    last_modified TIMESTAMPTZ,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    card_json JSONB,
    siblings_json JSONB,
    raw_hub_json JSONB,
    hub_synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_datasets_downloads ON datasets (downloads DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_datasets_last_modified ON datasets (last_modified DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_datasets_tags ON datasets USING GIN (tags);

CREATE TABLE IF NOT EXISTS sync_runs (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    mode TEXT NOT NULL,
    rows_upserted INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    error_message TEXT
);
