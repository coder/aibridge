package aibridge

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var baseLabels []string = []string{"provider", "model"}

const (
	InterceptionCountStatusFailed    = "failed"
	InterceptionCountStatusCompleted = "completed"
)

type Metrics struct {
	// Interception-related metrics.
	InterceptionDuration *prometheus.HistogramVec
	InterceptionCount    *prometheus.CounterVec
	PassthroughCount     *prometheus.CounterVec

	// Prompt-related metrics.
	PromptCount *prometheus.CounterVec

	// Token-related metrics.
	TokenUseCount *prometheus.CounterVec

	// Tool-related metrics.
	InjectedToolUseCount    *prometheus.CounterVec
	NonInjectedToolUseCount *prometheus.CounterVec
}

// NewMetrics creates AND registers metrics. It will panic if a collector has already been registered.
// Note: we are not specifying namespace in the metrics; the provided registerer may specify a "namespace"
// using [prometheus.WrapRegistererWithPrefix].
func NewMetrics(reg prometheus.Registerer) *Metrics {
	return &Metrics{
		// Interception-related metrics.
		InterceptionCount: promauto.With(reg).NewCounterVec(prometheus.CounterOpts{
			Subsystem: "interceptions",
			Name:      "total",
			Help:      "The count of intercepted requests",
		}, append(baseLabels, "status", "route", "method")),
		InterceptionDuration: promauto.With(reg).NewHistogramVec(prometheus.HistogramOpts{
			Subsystem: "interceptions",
			Name:      "duration",
			Help:      "The total duration of intercepted requests",
			// We can't control the duration (it's up to the provider), so this is just illustrative.
			Buckets: []float64{1, 5, 10, 20, 30, 45, 60, 120},
		}, baseLabels),
		PassthroughCount: promauto.With(reg).NewCounterVec(prometheus.CounterOpts{
			Subsystem: "passthrough",
			Name:      "total",
			Help:      "The count of requests which were not intercepted but passed through to the upstream",
		}, []string{"provider", "route", "method"}),

		// Prompt-related metrics.
		PromptCount: promauto.With(reg).NewCounterVec(prometheus.CounterOpts{
			Subsystem: "prompts",
			Name:      "total",
			Help:      "The number of prompts issued by users (initiators)",
		}, append(baseLabels, "initiator_id")),

		// Token-related metrics.
		TokenUseCount: promauto.With(reg).NewCounterVec(prometheus.CounterOpts{
			Subsystem: "token_usage",
			Name:      "total",
			Help:      "The number of tokens used by intercepted requests",
		}, append(baseLabels, "type", "initiator_id")),

		// Tool-related metrics.
		InjectedToolUseCount: promauto.With(reg).NewCounterVec(prometheus.CounterOpts{
			Subsystem: "injected_tool_usage",
			Name:      "total",
			Help:      "The number of times an injected MCP tool was invoked by aibridge",
		}, append(baseLabels, "server", "name", "failed", "initiator_id")),
		NonInjectedToolUseCount: promauto.With(reg).NewCounterVec(prometheus.CounterOpts{
			Subsystem: "non_injected_tool_usage",
			Name:      "total",
			Help:      "The number of times an AI model determined a tool must be invoked by the client",
		}, append(baseLabels, "name", "initiator_id")),
	}
}
