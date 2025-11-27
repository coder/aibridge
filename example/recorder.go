package main

import (
	"context"
	"database/sql"
	"encoding/json"

	"cdr.dev/slog"
	"github.com/coder/aibridge"
	"github.com/google/uuid"
)

// SQLiteRecorder implements aibridge.Recorder and persists usage data to SQLite.
type SQLiteRecorder struct {
	db     *sql.DB
	logger slog.Logger

	stmtInsertInterception    *sql.Stmt
	stmtUpdateInterception    *sql.Stmt
	stmtInsertTokenUsage      *sql.Stmt
	stmtInsertPromptUsage     *sql.Stmt
	stmtInsertToolUsage       *sql.Stmt
}

func NewSQLiteRecorder(db *sql.DB, logger slog.Logger) (*SQLiteRecorder, error) {
	r := &SQLiteRecorder{db: db, logger: logger}

	var err error
	r.stmtInsertInterception, err = db.Prepare(`
		INSERT INTO aibridge_interceptions (id, initiator_id, provider, model, started_at, metadata)
		VALUES (?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return nil, err
	}

	r.stmtUpdateInterception, err = db.Prepare(`
		UPDATE aibridge_interceptions SET ended_at = ? WHERE id = ?`)
	if err != nil {
		return nil, err
	}

	r.stmtInsertTokenUsage, err = db.Prepare(`
		INSERT INTO aibridge_token_usages (id, interception_id, provider_response_id, input_tokens, output_tokens, metadata, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return nil, err
	}

	r.stmtInsertPromptUsage, err = db.Prepare(`
		INSERT INTO aibridge_user_prompts (id, interception_id, provider_response_id, prompt, metadata, created_at)
		VALUES (?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return nil, err
	}

	r.stmtInsertToolUsage, err = db.Prepare(`
		INSERT INTO aibridge_tool_usages (id, interception_id, provider_response_id, server_url, tool, input, injected, invocation_error, metadata, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return nil, err
	}

	return r, nil
}

func (r *SQLiteRecorder) Close() error {
	r.stmtInsertInterception.Close()
	r.stmtUpdateInterception.Close()
	r.stmtInsertTokenUsage.Close()
	r.stmtInsertPromptUsage.Close()
	r.stmtInsertToolUsage.Close()
	return nil
}

func (r *SQLiteRecorder) RecordInterception(ctx context.Context, req *aibridge.InterceptionRecord) error {
	metadata, _ := json.Marshal(req.Metadata)
	_, err := r.stmtInsertInterception.ExecContext(ctx,
		req.ID, req.InitiatorID, req.Provider, req.Model, req.StartedAt, string(metadata),
	)
	if err != nil {
		r.logger.Warn(ctx, "failed to record interception", slog.Error(err))
	}
	return err
}

func (r *SQLiteRecorder) RecordInterceptionEnded(ctx context.Context, req *aibridge.InterceptionRecordEnded) error {
	_, err := r.stmtUpdateInterception.ExecContext(ctx, req.EndedAt, req.ID)
	if err != nil {
		r.logger.Warn(ctx, "failed to record interception end", slog.Error(err))
	}
	return err
}

func (r *SQLiteRecorder) RecordTokenUsage(ctx context.Context, req *aibridge.TokenUsageRecord) error {
	// Build metadata, merging extra token types.
	merged := make(map[string]any)
	for k, v := range req.Metadata {
		merged[k] = v
	}
	for k, v := range req.ExtraTokenTypes {
		merged[k] = v
	}
	metadata, _ := json.Marshal(merged)

	_, err := r.stmtInsertTokenUsage.ExecContext(ctx,
		uuid.NewString(), req.InterceptionID, req.MsgID, req.Input, req.Output, string(metadata), req.CreatedAt,
	)
	if err != nil {
		r.logger.Warn(ctx, "failed to record token usage", slog.Error(err))
	}
	return err
}

func (r *SQLiteRecorder) RecordPromptUsage(ctx context.Context, req *aibridge.PromptUsageRecord) error {
	metadata, _ := json.Marshal(req.Metadata)
	_, err := r.stmtInsertPromptUsage.ExecContext(ctx,
		uuid.NewString(), req.InterceptionID, req.MsgID, req.Prompt, string(metadata), req.CreatedAt,
	)
	if err != nil {
		r.logger.Warn(ctx, "failed to record prompt usage", slog.Error(err))
	}
	return err
}

func (r *SQLiteRecorder) RecordToolUsage(ctx context.Context, req *aibridge.ToolUsageRecord) error {
	metadata, _ := json.Marshal(req.Metadata)
	input, _ := json.Marshal(req.Args)

	var serverURL *string
	if req.ServerURL != nil {
		serverURL = req.ServerURL
	}

	var invocationError *string
	if req.InvocationError != nil {
		errStr := req.InvocationError.Error()
		invocationError = &errStr
	}

	_, err := r.stmtInsertToolUsage.ExecContext(ctx,
		uuid.NewString(), req.InterceptionID, req.MsgID, serverURL, req.Tool, string(input), req.Injected, invocationError, string(metadata), req.CreatedAt,
	)
	if err != nil {
		r.logger.Warn(ctx, "failed to record tool usage", slog.Error(err))
	}
	return err
}
