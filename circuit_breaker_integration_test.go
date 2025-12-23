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

	type testCase struct {
		name           string
		providerName   string
		endpoint       string
		errorBody      string
		successBody    string
		requestBody    string
		setupHeaders   func(req *http.Request)
		createProvider func(baseURL string, cbConfig *aibridge.CircuitBreakerConfig) aibridge.Provider
	}

	tests := []testCase{
		{
			name:         "Anthropic",
			providerName: aibridge.ProviderAnthropic,
			endpoint:     "/v1/messages",
			errorBody:    `{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}`,
			successBody:  `{"id":"msg_01","type":"message","role":"assistant","content":[{"type":"text","text":"Hello!"}],"model":"claude-sonnet-4-20250514","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`,
			requestBody:  `{"model":"claude-sonnet-4-20250514","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}`,
			setupHeaders: func(req *http.Request) {
				req.Header.Set("x-api-key", "test")
				req.Header.Set("anthropic-version", "2023-06-01")
			},
			createProvider: func(baseURL string, cbConfig *aibridge.CircuitBreakerConfig) aibridge.Provider {
				return aibridge.NewAnthropicProvider(aibridge.AnthropicConfig{
					BaseURL:        baseURL,
					Key:            "test-key",
					CircuitBreaker: cbConfig,
				}, nil)
			},
		},
		{
			name:         "OpenAI",
			providerName: aibridge.ProviderOpenAI,
			endpoint:     "/v1/chat/completions",
			errorBody:    `{"error":{"type":"rate_limit_error","message":"rate limited","code":"rate_limit_exceeded"}}`,
			successBody:  `{"id":"chatcmpl-123","object":"chat.completion","created":1677652288,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":12,"total_tokens":21}}`,
			requestBody:  `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`,
			setupHeaders: func(req *http.Request) {
				req.Header.Set("Authorization", "Bearer test-key")
			},
			createProvider: func(baseURL string, cbConfig *aibridge.CircuitBreakerConfig) aibridge.Provider {
				return aibridge.NewOpenAIProvider(aibridge.OpenAIConfig{
					BaseURL:        baseURL,
					Key:            "test-key",
					CircuitBreaker: cbConfig,
				})
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var upstreamCalls atomic.Int32
			var shouldFail atomic.Bool
			shouldFail.Store(true)

			// Mock upstream that returns 429 or 200 based on shouldFail flag.
			// x-should-retry: false is required to disable SDK automatic retries (default MaxRetries=2).
			mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				upstreamCalls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("x-should-retry", "false")
				if shouldFail.Load() {
					w.WriteHeader(http.StatusTooManyRequests)
					_, _ = w.Write([]byte(tc.errorBody))
				} else {
					w.WriteHeader(http.StatusOK)
					_, _ = w.Write([]byte(tc.successBody))
				}
			}))
			defer mockUpstream.Close()

			metrics := aibridge.NewMetrics(prometheus.NewRegistry())

			// Create provider with circuit breaker config
			cbConfig := &aibridge.CircuitBreakerConfig{
				FailureThreshold: 2,
				Interval:         time.Minute,
				Timeout:          50 * time.Millisecond,
				MaxRequests:      1,
			}
			provider := tc.createProvider(mockUpstream.URL, cbConfig)

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
				req, _ := http.NewRequest("POST", mockSrv.URL+"/"+tc.providerName+tc.endpoint, strings.NewReader(tc.requestBody))
				req.Header.Set("Content-Type", "application/json")
				tc.setupHeaders(req)
				resp, err := http.DefaultClient.Do(req)
				require.NoError(t, err)
				_, _ = io.ReadAll(resp.Body)
				resp.Body.Close()
				return resp
			}

			// Phase 1: Trip the circuit breaker
			// First FailureThreshold requests hit upstream, get 429
			for i := uint32(0); i < cbConfig.FailureThreshold; i++ {
				resp := makeRequest()
				assert.Equal(t, http.StatusTooManyRequests, resp.StatusCode)
			}
			assert.Equal(t, int32(cbConfig.FailureThreshold), upstreamCalls.Load())

			// Phase 2: Verify circuit is open
			// Request should be blocked by circuit breaker (no upstream call)
			resp := makeRequest()
			assert.Equal(t, http.StatusServiceUnavailable, resp.StatusCode)
			assert.Equal(t, int32(cbConfig.FailureThreshold), upstreamCalls.Load(), "No new upstream call when circuit is open")

			// Verify metrics show circuit is open
			trips := promtest.ToFloat64(metrics.CircuitBreakerTrips.WithLabelValues(tc.providerName, tc.endpoint))
			assert.Equal(t, 1.0, trips, "CircuitBreakerTrips should be 1")

			state := promtest.ToFloat64(metrics.CircuitBreakerState.WithLabelValues(tc.providerName, tc.endpoint))
			assert.Equal(t, 1.0, state, "CircuitBreakerState should be 1 (open)")

			rejects := promtest.ToFloat64(metrics.CircuitBreakerRejects.WithLabelValues(tc.providerName, tc.endpoint))
			assert.Equal(t, 1.0, rejects, "CircuitBreakerRejects should be 1")

			// Phase 3: Wait for timeout to transition to half-open
			time.Sleep(cbConfig.Timeout + 10*time.Millisecond)

			// Switch upstream to return success
			shouldFail.Store(false)

			// Phase 4: Recovery - request in half-open state should succeed and close circuit
			upstreamCallsBefore := upstreamCalls.Load()
			resp = makeRequest()
			assert.Equal(t, http.StatusOK, resp.StatusCode, "Request should succeed in half-open state")
			assert.Equal(t, upstreamCallsBefore+1, upstreamCalls.Load(), "Request should reach upstream in half-open state")

			// Verify circuit is now closed
			state = promtest.ToFloat64(metrics.CircuitBreakerState.WithLabelValues(tc.providerName, tc.endpoint))
			assert.Equal(t, 0.0, state, "CircuitBreakerState should be 0 (closed) after recovery")

			// Phase 5: Verify circuit is fully functional again
			// Multiple requests should all succeed and reach upstream
			for i := 0; i < 3; i++ {
				resp = makeRequest()
				assert.Equal(t, http.StatusOK, resp.StatusCode, "Request should succeed after circuit closes")
			}

			// All requests should have reached upstream
			assert.Equal(t, upstreamCallsBefore+4, upstreamCalls.Load(), "All requests should reach upstream after circuit closes")

			// Rejects count should not have increased
			rejects = promtest.ToFloat64(metrics.CircuitBreakerRejects.WithLabelValues(tc.providerName, tc.endpoint))
			assert.Equal(t, 1.0, rejects, "CircuitBreakerRejects should still be 1 (no new rejects)")
		})
	}
}

func TestCircuitBreaker_HalfOpenFailure(t *testing.T) {
	t.Parallel()

	var upstreamCalls atomic.Int32

	// Mock upstream that always returns 429.
	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("x-should-retry", "false")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"type":"rate_limit_error","message":"rate limited","code":"rate_limit_exceeded"}}`))
	}))
	defer mockUpstream.Close()

	metrics := aibridge.NewMetrics(prometheus.NewRegistry())

	cbConfig := &aibridge.CircuitBreakerConfig{
		FailureThreshold: 2,
		Interval:         time.Minute,
		Timeout:          50 * time.Millisecond,
		MaxRequests:      1,
	}
	provider := aibridge.NewOpenAIProvider(aibridge.OpenAIConfig{
		BaseURL:        mockUpstream.URL,
		Key:            "test-key",
		CircuitBreaker: cbConfig,
	})

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
		req, _ := http.NewRequest("POST", mockSrv.URL+"/openai/v1/chat/completions",
			strings.NewReader(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer test-key")
		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		_, _ = io.ReadAll(resp.Body)
		resp.Body.Close()
		return resp
	}

	// Phase 1: Trip the circuit
	for i := uint32(0); i < cbConfig.FailureThreshold; i++ {
		resp := makeRequest()
		assert.Equal(t, http.StatusTooManyRequests, resp.StatusCode)
	}

	// Verify circuit is open
	resp := makeRequest()
	assert.Equal(t, http.StatusServiceUnavailable, resp.StatusCode)

	trips := promtest.ToFloat64(metrics.CircuitBreakerTrips.WithLabelValues("openai", "/v1/chat/completions"))
	assert.Equal(t, 1.0, trips, "CircuitBreakerTrips should be 1")

	// Phase 2: Wait for half-open state
	time.Sleep(cbConfig.Timeout + 10*time.Millisecond)

	// Phase 3: Request in half-open state fails, circuit should re-open
	upstreamCallsBefore := upstreamCalls.Load()
	resp = makeRequest()
	assert.Equal(t, http.StatusTooManyRequests, resp.StatusCode, "Request should fail in half-open state")
	assert.Equal(t, upstreamCallsBefore+1, upstreamCalls.Load(), "Request should reach upstream in half-open state")

	// Circuit should be open again - next request should be rejected immediately
	resp = makeRequest()
	assert.Equal(t, http.StatusServiceUnavailable, resp.StatusCode, "Circuit should be open again after half-open failure")
	assert.Equal(t, upstreamCallsBefore+1, upstreamCalls.Load(), "Request should NOT reach upstream when circuit re-opens")

	// Verify metrics: trips should be 2 now (tripped twice)
	trips = promtest.ToFloat64(metrics.CircuitBreakerTrips.WithLabelValues("openai", "/v1/chat/completions"))
	assert.Equal(t, 2.0, trips, "CircuitBreakerTrips should be 2 after half-open failure")

	state := promtest.ToFloat64(metrics.CircuitBreakerState.WithLabelValues("openai", "/v1/chat/completions"))
	assert.Equal(t, 1.0, state, "CircuitBreakerState should be 1 (open) after half-open failure")
}
