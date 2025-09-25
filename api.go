package aibridge

import (
	"context"
	"time"
)

type ToolArgs any

type Metadata map[string]any

type InterceptionRecord struct {
	ID                           string
	InitiatorID, Provider, Model string
	Metadata                     Metadata
	StartedAt                    time.Time
}

type TokenUsageRecord struct {
	InterceptionID string
	MsgID          string
	Input, Output  int64
	Metadata       Metadata
	CreatedAt      time.Time
}

type PromptUsageRecord struct {
	InterceptionID string
	MsgID, Prompt  string
	Metadata       Metadata
	CreatedAt      time.Time
}

type ToolUsageRecord struct {
	InterceptionID  string
	MsgID, Tool     string
	ServerURL       *string
	Args            ToolArgs
	Injected        bool
	InvocationError error
	Metadata        Metadata
	CreatedAt       time.Time
}

// Recorder describes all the possible usage information we need to capture during interactions with AI providers.
// Additionally, it introduces the concept of an "Interception", which includes information about which provider/model was
// used and by whom. All usage records should reference this Interception by ID.
type Recorder interface {
	// RecordInterception records metadata about an interception with an upstream AI provider.
	RecordInterception(ctx context.Context, req *InterceptionRecord) error
	// RecordTokenUsage records the tokens used in an interception with an upstream AI provider.
	RecordTokenUsage(ctx context.Context, req *TokenUsageRecord) error
	// RecordPromptUsage records the prompts used in an interception with an upstream AI provider.
	RecordPromptUsage(ctx context.Context, req *PromptUsageRecord) error
	// RecordToolUsage records the tools used in an interception with an upstream AI provider.
	RecordToolUsage(ctx context.Context, req *ToolUsageRecord) error
}
