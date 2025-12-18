package aibridge_test

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"cdr.dev/slog"
	"cdr.dev/slog/sloggers/slogtest"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/mcp"
	"github.com/prometheus/client_golang/prometheus"
	promtest "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
)

func TestCircuitBreaker_WithNewRequestBridge(t *testing.T) {
	t.Parallel()

	var upstreamCalls atomic.Int32

	// Mock upstream that returns 429 in Anthropic error format.
	// x-should-retry: false is required to disable SDK automatic retries (default MaxRetries=2).
	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("x-should-retry", "false")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}`))
	}))
	defer mockUpstream.Close()

	metrics := aibridge.NewMetrics(prometheus.NewRegistry())

	// Create provider with circuit breaker config
	provider := aibridge.NewAnthropicProvider(aibridge.AnthropicConfig{
		BaseURL: mockUpstream.URL,
		Key:     "test-key",
		CircuitBreaker: &aibridge.CircuitBreakerConfig{
			FailureThreshold: 2,
			Interval:         time.Minute,
			Timeout:          50 * time.Millisecond,
			MaxRequests:      1,
		},
	}, nil)

	ctx := t.Context()
	tracer := otel.Tracer("forTesting")
	logger := slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)
	bridge, err := aibridge.NewRequestBridge(ctx,
		[]aibridge.Provider{provider},
		&mockRecorderClient{},
		mcp.NewServerProxyManager(nil, tracer),
		logger,
		metrics,
		tracer,
	)
	require.NoError(t, err)

	mockSrv := httptest.NewUnstartedServer(bridge)
	t.Cleanup(mockSrv.Close)
	mockSrv.Config.BaseContext = func(_ net.Listener) context.Context {
		return aibridge.AsActor(ctx, "test-user-id", nil)
	}
	mockSrv.Start()

	makeRequest := func() *http.Response {
		body := `{"model":"claude-sonnet-4-20250514","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}`
		req, _ := http.NewRequest("POST", mockSrv.URL+"/anthropic/v1/messages", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", "test")
		req.Header.Set("anthropic-version", "2023-06-01")
		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		_, _ = io.ReadAll(resp.Body)
		resp.Body.Close()
		return resp
	}

	// First 2 requests hit upstream, get 429
	for i := 0; i < 2; i++ {
		resp := makeRequest()
		assert.Equal(t, http.StatusTooManyRequests, resp.StatusCode)
	}
	assert.Equal(t, int32(2), upstreamCalls.Load())

	// Third request should be blocked by circuit breaker
	resp := makeRequest()
	assert.Equal(t, http.StatusServiceUnavailable, resp.StatusCode)
	assert.Equal(t, int32(2), upstreamCalls.Load()) // No new upstream call

	// Verify metrics were recorded via NewRequestBridge's onChange callback
	trips := promtest.ToFloat64(metrics.CircuitBreakerTrips.WithLabelValues(aibridge.ProviderAnthropic, "/v1/messages"))
	assert.Equal(t, 1.0, trips, "CircuitBreakerTrips should be 1")

	state := promtest.ToFloat64(metrics.CircuitBreakerState.WithLabelValues(aibridge.ProviderAnthropic, "/v1/messages"))
	assert.Equal(t, 1.0, state, "CircuitBreakerState should be 1 (open)")

	rejects := promtest.ToFloat64(metrics.CircuitBreakerRejects.WithLabelValues(aibridge.ProviderAnthropic, "/v1/messages"))
	assert.Equal(t, 1.0, rejects, "CircuitBreakerRejects should be 1")
}
