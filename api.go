package aibridge

import (
	"context"

	"cdr.dev/slog"
	"github.com/coder/aibridge/config"
	aibcontext "github.com/coder/aibridge/context"
	"github.com/coder/aibridge/metrics"
	"github.com/coder/aibridge/provider"
	"github.com/coder/aibridge/recorder"
	"github.com/prometheus/client_golang/prometheus"
	"go.opentelemetry.io/otel/trace"
)

// Type + function aliases for backwards compatibility.
type (
	Metrics = metrics.Metrics

	Provider = provider.Provider

	InterceptionRecord      = recorder.InterceptionRecord
	InterceptionRecordEnded = recorder.InterceptionRecordEnded
	TokenUsageRecord        = recorder.TokenUsageRecord
	PromptUsageRecord       = recorder.PromptUsageRecord
	ToolUsageRecord         = recorder.ToolUsageRecord
	Recorder                = recorder.Recorder
	RecorderWrapper         = recorder.RecorderWrapper
	AsyncRecorder           = recorder.AsyncRecorder
	Metadata                = recorder.Metadata

	AnthropicConfig  = config.AnthropicConfig
	AWSBedrockConfig = config.AWSBedrockConfig
	OpenAIConfig     = config.OpenAIConfig
)

func AsActor(ctx context.Context, actorID string, metadata recorder.Metadata) context.Context {
	return aibcontext.AsActor(ctx, actorID, metadata)
}

func NewAnthropicProvider(cfg config.AnthropicConfig, bedrockCfg *config.AWSBedrockConfig) provider.Provider {
	return provider.NewAnthropic(cfg, bedrockCfg)
}

func NewOpenAIProvider(cfg config.OpenAIConfig) provider.Provider {
	return provider.NewOpenAI(cfg)
}

func NewMetrics(reg prometheus.Registerer) *metrics.Metrics {
	return metrics.NewMetrics(reg)
}

func NewRecorder(logger slog.Logger, tracer trace.Tracer, clientFn func() (Recorder, error)) *RecorderWrapper {
	return recorder.NewRecorder(logger, tracer, clientFn)
}
