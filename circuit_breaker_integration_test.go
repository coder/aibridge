package aibridge_test

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"cdr.dev/slog/v3"
	"cdr.dev/slog/v3/sloggers/slogtest"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/internal/testutil"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/metrics"
	"github.com/coder/aibridge/provider"
	"github.com/prometheus/client_golang/prometheus"
	promtest "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
)

// TestCircuitBreaker_FullRecoveryCycle tests the complete circuit breaker lifecycle:
// closed → open (after consecutive failures) → half-open (after timeout) → closed (after successful request)
func TestCircuitBreaker_FullRecoveryCycle(t *testing.T) {
	t.Parallel()

	type testCase struct {
		name           string
		providerName   string
		endpoint       string
		errorBody      string
		successBody    string
		requestBody    string
		setupHeaders   func(req *http.Request)
		createProvider func(baseURL string, cbConfig *config.CircuitBreaker) provider.Provider
	}

	tests := []testCase{
		{
			name:         "Anthropic",
			providerName: config.ProviderAnthropic,
			endpoint:     "/v1/messages",
			errorBody:    `{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}`,
			successBody:  `{"id":"msg_01","type":"message","role":"assistant","content":[{"type":"text","text":"Hello!"}],"model":"claude-sonnet-4-20250514","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`,
			requestBody:  `{"model":"claude-sonnet-4-20250514","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}`,
			setupHeaders: func(req *http.Request) {
				req.Header.Set("x-api-key", "test")
				req.Header.Set("anthropic-version", "2023-06-01")
			},
			createProvider: func(baseURL string, cbConfig *config.CircuitBreaker) provider.Provider {
				return provider.NewAnthropic(config.Anthropic{
					BaseURL:        baseURL,
					Key:            "test-key",
					CircuitBreaker: cbConfig,
				}, nil)
			},
		},
		{
			name:         "OpenAI",
			providerName: config.ProviderOpenAI,
			endpoint:     "/v1/chat/completions",
			errorBody:    `{"error":{"type":"rate_limit_error","message":"rate limited","code":"rate_limit_exceeded"}}`,
			successBody:  `{"id":"chatcmpl-123","object":"chat.completion","created":1677652288,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":12,"total_tokens":21}}`,
			requestBody:  `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`,
			setupHeaders: func(req *http.Request) {
				req.Header.Set("Authorization", "Bearer test-key")
			},
			createProvider: func(baseURL string, cbConfig *config.CircuitBreaker) provider.Provider {
				return provider.NewOpenAI(config.OpenAI{
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

			metrics := metrics.NewMetrics(prometheus.NewRegistry())

			// Create provider with circuit breaker config
			cbConfig := &config.CircuitBreaker{
				FailureThreshold: 2,
				Interval:         time.Minute,
				Timeout:          50 * time.Millisecond,
				MaxRequests:      1,
			}
			prov := tc.createProvider(mockUpstream.URL, cbConfig)

			ctx := t.Context()
			tracer := otel.Tracer("forTesting")
			logger := slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)
			bridge, err := aibridge.NewRequestBridge(ctx,
				[]provider.Provider{prov},
				&testutil.MockRecorder{},
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
				req, err := http.NewRequest("POST", mockSrv.URL+"/"+tc.providerName+tc.endpoint, strings.NewReader(tc.requestBody))
				require.NoError(t, err)
				req.Header.Set("Content-Type", "application/json")
				tc.setupHeaders(req)
				resp, err := http.DefaultClient.Do(req)
				require.NoError(t, err)
				_, err = io.ReadAll(resp.Body)
				require.NoError(t, err)
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

// TestCircuitBreaker_HalfOpenFailure tests that a failed request in half-open state
// returns the circuit to open: closed → open → half-open → open
func TestCircuitBreaker_HalfOpenFailure(t *testing.T) {
	t.Parallel()

	type testCase struct {
		name           string
		providerName   string
		endpoint       string
		errorBody      string
		requestBody    string
		setupHeaders   func(req *http.Request)
		createProvider func(baseURL string, cbConfig *config.CircuitBreaker) provider.Provider
	}

	tests := []testCase{
		{
			name:         "Anthropic",
			providerName: config.ProviderAnthropic,
			endpoint:     "/v1/messages",
			errorBody:    `{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}`,
			requestBody:  `{"model":"claude-sonnet-4-20250514","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}`,
			setupHeaders: func(req *http.Request) {
				req.Header.Set("x-api-key", "test")
				req.Header.Set("anthropic-version", "2023-06-01")
			},
			createProvider: func(baseURL string, cbConfig *config.CircuitBreaker) provider.Provider {
				return provider.NewAnthropic(config.Anthropic{
					BaseURL:        baseURL,
					Key:            "test-key",
					CircuitBreaker: cbConfig,
				}, nil)
			},
		},
		{
			name:         "OpenAI",
			providerName: config.ProviderOpenAI,
			endpoint:     "/v1/chat/completions",
			errorBody:    `{"error":{"type":"rate_limit_error","message":"rate limited","code":"rate_limit_exceeded"}}`,
			requestBody:  `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`,
			setupHeaders: func(req *http.Request) {
				req.Header.Set("Authorization", "Bearer test-key")
			},
			createProvider: func(baseURL string, cbConfig *config.CircuitBreaker) provider.Provider {
				return provider.NewOpenAI(config.OpenAI{
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

			// Mock upstream that always returns 429.
			mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				upstreamCalls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("x-should-retry", "false")
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = w.Write([]byte(tc.errorBody))
			}))
			defer mockUpstream.Close()

			metrics := metrics.NewMetrics(prometheus.NewRegistry())

			cbConfig := &config.CircuitBreaker{
				FailureThreshold: 2,
				Interval:         time.Minute,
				Timeout:          50 * time.Millisecond,
				MaxRequests:      1,
			}
			prov := tc.createProvider(mockUpstream.URL, cbConfig)

			ctx := t.Context()
			tracer := otel.Tracer("forTesting")
			logger := slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)
			bridge, err := aibridge.NewRequestBridge(ctx,
				[]provider.Provider{prov},
				&testutil.MockRecorder{},
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
				req, err := http.NewRequest("POST", mockSrv.URL+"/"+tc.providerName+tc.endpoint, strings.NewReader(tc.requestBody))
				require.NoError(t, err)
				req.Header.Set("Content-Type", "application/json")
				tc.setupHeaders(req)
				resp, err := http.DefaultClient.Do(req)
				require.NoError(t, err)
				_, err = io.ReadAll(resp.Body)
				require.NoError(t, err)
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

			trips := promtest.ToFloat64(metrics.CircuitBreakerTrips.WithLabelValues(tc.providerName, tc.endpoint))
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
			trips = promtest.ToFloat64(metrics.CircuitBreakerTrips.WithLabelValues(tc.providerName, tc.endpoint))
			assert.Equal(t, 2.0, trips, "CircuitBreakerTrips should be 2 after half-open failure")

			state := promtest.ToFloat64(metrics.CircuitBreakerState.WithLabelValues(tc.providerName, tc.endpoint))
			assert.Equal(t, 1.0, state, "CircuitBreakerState should be 1 (open) after half-open failure")
		})
	}
}

// TestCircuitBreaker_HalfOpenMaxRequests tests that MaxRequests limits concurrent
// requests in half-open state. Requests beyond the limit should be rejected.
func TestCircuitBreaker_HalfOpenMaxRequests(t *testing.T) {
	t.Parallel()

	type testCase struct {
		name           string
		providerName   string
		endpoint       string
		errorBody      string
		successBody    string
		requestBody    string
		setupHeaders   func(req *http.Request)
		createProvider func(baseURL string, cbConfig *config.CircuitBreaker) provider.Provider
	}

	tests := []testCase{
		{
			name:         "Anthropic",
			providerName: config.ProviderAnthropic,
			endpoint:     "/v1/messages",
			errorBody:    `{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}`,
			successBody:  `{"id":"msg_01","type":"message","role":"assistant","content":[{"type":"text","text":"Hello!"}],"model":"claude-sonnet-4-20250514","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`,
			requestBody:  `{"model":"claude-sonnet-4-20250514","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}`,
			setupHeaders: func(req *http.Request) {
				req.Header.Set("x-api-key", "test")
				req.Header.Set("anthropic-version", "2023-06-01")
			},
			createProvider: func(baseURL string, cbConfig *config.CircuitBreaker) provider.Provider {
				return provider.NewAnthropic(config.Anthropic{
					BaseURL:        baseURL,
					Key:            "test-key",
					CircuitBreaker: cbConfig,
				}, nil)
			},
		},
		{
			name:         "OpenAI",
			providerName: config.ProviderOpenAI,
			endpoint:     "/v1/chat/completions",
			errorBody:    `{"error":{"type":"rate_limit_error","message":"rate limited","code":"rate_limit_exceeded"}}`,
			successBody:  `{"id":"chatcmpl-123","object":"chat.completion","created":1677652288,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":12,"total_tokens":21}}`,
			requestBody:  `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`,
			setupHeaders: func(req *http.Request) {
				req.Header.Set("Authorization", "Bearer test-key")
			},
			createProvider: func(baseURL string, cbConfig *config.CircuitBreaker) provider.Provider {
				return provider.NewOpenAI(config.OpenAI{
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

			// Upstream is slow to ensure concurrent requests overlap in half-open state.
			mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				upstreamCalls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("x-should-retry", "false")
				if shouldFail.Load() {
					w.WriteHeader(http.StatusTooManyRequests)
					_, _ = w.Write([]byte(tc.errorBody))
				} else {
					// Slow response to ensure requests overlap
					time.Sleep(100 * time.Millisecond)
					w.WriteHeader(http.StatusOK)
					_, _ = w.Write([]byte(tc.successBody))
				}
			}))
			defer mockUpstream.Close()

			metrics := metrics.NewMetrics(prometheus.NewRegistry())

			const maxRequests = 2
			cbConfig := &config.CircuitBreaker{
				FailureThreshold: 2,
				Interval:         time.Minute,
				Timeout:          50 * time.Millisecond,
				MaxRequests:      maxRequests, // Allow only 2 concurrent requests in half-open
			}
			prov := tc.createProvider(mockUpstream.URL, cbConfig)

			ctx := t.Context()
			tracer := otel.Tracer("forTesting")
			logger := slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)
			bridge, err := aibridge.NewRequestBridge(ctx,
				[]provider.Provider{prov},
				&testutil.MockRecorder{},
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
				req, err := http.NewRequest("POST", mockSrv.URL+"/"+tc.providerName+tc.endpoint, strings.NewReader(tc.requestBody))
				require.NoError(t, err)
				req.Header.Set("Content-Type", "application/json")
				tc.setupHeaders(req)
				resp, err := http.DefaultClient.Do(req)
				require.NoError(t, err)
				_, err = io.ReadAll(resp.Body)
				require.NoError(t, err)
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

			// Phase 2: Wait for half-open state and switch upstream to success
			time.Sleep(cbConfig.Timeout + 10*time.Millisecond)
			shouldFail.Store(false)
			upstreamCalls.Store(0)

			// Phase 3: Send concurrent requests (more than MaxRequests)
			const totalRequests = 5
			var wg sync.WaitGroup
			responses := make(chan int, totalRequests)

			for i := 0; i < totalRequests; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					resp := makeRequest()
					responses <- resp.StatusCode
				}()
			}

			wg.Wait()
			close(responses)

			// Count results
			var successCount, rejectedCount int
			for status := range responses {
				switch status {
				case http.StatusOK:
					successCount++
				case http.StatusServiceUnavailable:
					rejectedCount++
				}
			}

			// Verify only MaxRequests reached upstream
			assert.Equal(t, int32(maxRequests), upstreamCalls.Load(),
				"Only MaxRequests (%d) should reach upstream in half-open state", maxRequests)

			// Verify request counts
			assert.Equal(t, maxRequests, successCount,
				"Only %d requests should succeed (MaxRequests)", maxRequests)
			assert.Equal(t, totalRequests-maxRequests, rejectedCount,
				"%d requests should be rejected (ErrTooManyRequests)", totalRequests-maxRequests)

			// Verify rejects metric increased
			rejects := promtest.ToFloat64(metrics.CircuitBreakerRejects.WithLabelValues(tc.providerName, tc.endpoint))
			assert.Equal(t, float64(1+totalRequests-maxRequests), rejects,
				"CircuitBreakerRejects should include half-open rejections")
		})
	}
}
