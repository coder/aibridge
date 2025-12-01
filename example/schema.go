package main

import "database/sql"

// initSchema creates the aibridge tables following the coder/coder data model.
// See: https://github.com/coder/coder/blob/c87c33f7dda82eb91ee8ba9504f749101bb367d6/coderd/database/dump.sql#L1052-L1115
func initSchema(db *sql.DB) error {
	schema := `
	CREATE TABLE IF NOT EXISTS aibridge_interceptions (
		id TEXT PRIMARY KEY,
		initiator_id TEXT NOT NULL,
		provider TEXT NOT NULL,
		model TEXT NOT NULL,
		started_at DATETIME NOT NULL,
		ended_at DATETIME,
		metadata TEXT
	);

	CREATE TABLE IF NOT EXISTS aibridge_token_usages (
		id TEXT PRIMARY KEY,
		interception_id TEXT NOT NULL,
		provider_response_id TEXT NOT NULL,
		input_tokens INTEGER NOT NULL,
		output_tokens INTEGER NOT NULL,
		metadata TEXT,
		created_at DATETIME NOT NULL,
		FOREIGN KEY (interception_id) REFERENCES aibridge_interceptions(id)
	);

	CREATE TABLE IF NOT EXISTS aibridge_user_prompts (
		id TEXT PRIMARY KEY,
		interception_id TEXT NOT NULL,
		provider_response_id TEXT NOT NULL,
		prompt TEXT NOT NULL,
		metadata TEXT,
		created_at DATETIME NOT NULL,
		FOREIGN KEY (interception_id) REFERENCES aibridge_interceptions(id)
	);

	CREATE TABLE IF NOT EXISTS aibridge_tool_usages (
		id TEXT PRIMARY KEY,
		interception_id TEXT NOT NULL,
		provider_response_id TEXT NOT NULL,
		server_url TEXT,
		tool TEXT NOT NULL,
		input TEXT NOT NULL,
		injected BOOLEAN NOT NULL DEFAULT FALSE,
		invocation_error TEXT,
		metadata TEXT,
		created_at DATETIME NOT NULL,
		FOREIGN KEY (interception_id) REFERENCES aibridge_interceptions(id)
	);

	CREATE INDEX IF NOT EXISTS idx_interceptions_initiator ON aibridge_interceptions(initiator_id);
	CREATE INDEX IF NOT EXISTS idx_token_usages_interception ON aibridge_token_usages(interception_id);
	CREATE INDEX IF NOT EXISTS idx_user_prompts_interception ON aibridge_user_prompts(interception_id);
	CREATE INDEX IF NOT EXISTS idx_tool_usages_interception ON aibridge_tool_usages(interception_id);
	`
	_, err := db.Exec(schema)
	return err
}
