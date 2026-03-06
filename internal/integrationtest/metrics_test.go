package integrationtest

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/coder/aibridge"
	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/fixtures"
	"github.com/coder/aibridge/metrics"
	"github.com/prometheus/client_golang/prometheus"
	promtest "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
)

func TestMetrics_Interception(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name           string
		fixture        []byte
		path           string
		expectStatus   string
		expectModel    string
		expectRoute    string
		expectProvider string
		allowOverflow  bool // error fixtures may cause retries
	}{
		{
			name:           "ant_simple",
			fixture:        fixtures.AntSimple,
			path:           pathAnthropicMessages,
			expectStatus:   metrics.InterceptionCountStatusCompleted,
			expectModel:    "claude-sonnet-4-0",
			expectRoute:    "/v1/messages",
			expectProvider: config.ProviderAnthropic,
		},
		{
			name:           "ant_error",
			fixture:        fixtures.AntNonStreamError,
			path:           pathAnthropicMessages,
			expectStatus:   metrics.InterceptionCountStatusFailed,
			expectModel:    "claude-sonnet-4-0",
			expectRoute:    "/v1/messages",
			expectProvider: config.ProviderAnthropic,
			allowOverflow:  true,
		},
		{
			name:           "oai_chat_simple",
			fixture:        fixtures.OaiChatSimple,
			path:           pathOpenAIChatCompletions,
			expectStatus:   metrics.InterceptionCountStatusCompleted,
			expectModel:    "gpt-4.1",
			expectRoute:    "/v1/chat/completions",
			expectProvider: config.ProviderOpenAI,
		},
		{
			name:           "oai_chat_error",
			fixture:        fixtures.OaiChatNonStreamError,
			path:           pathOpenAIChatCompletions,
			expectStatus:   metrics.InterceptionCountStatusFailed,
			expectModel:    "gpt-4.1",
			expectRoute:    "/v1/chat/completions",
			expectProvider: config.ProviderOpenAI,
			allowOverflow:  true,
		},
		{
			name:           "oai_responses_blocking_simple",
			fixture:        fixtures.OaiResponsesBlockingSimple,
			path:           pathOpenAIResponses,
			expectStatus:   metrics.InterceptionCountStatusCompleted,
			expectModel:    "gpt-4o-mini",
			expectRoute:    "/v1/responses",
			expectProvider: config.ProviderOpenAI,
		},
		{
			name:           "oai_responses_blocking_error",
			fixture:        fixtures.OaiResponsesBlockingHttpErr,
			path:           pathOpenAIResponses,
			expectStatus:   metrics.InterceptionCountStatusFailed,
			expectModel:    "gpt-4o-mini",
			expectRoute:    "/v1/responses",
			expectProvider: config.ProviderOpenAI,
			allowOverflow:  true,
		},
		{
			name:           "oai_responses_streaming_simple",
			fixture:        fixtures.OaiResponsesStreamingSimple,
			path:           pathOpenAIResponses,
			expectStatus:   metrics.InterceptionCountStatusCompleted,
			expectModel:    "gpt-4o-mini",
			expectRoute:    "/v1/responses",
			expectProvider: config.ProviderOpenAI,
		},
		{
			name:           "oai_responses_streaming_error",
			fixture:        fixtures.OaiResponsesStreamingHttpErr,
			path:           pathOpenAIResponses,
			expectStatus:   metrics.InterceptionCountStatusFailed,
			expectModel:    "gpt-4o-mini",
			expectRoute:    "/v1/responses",
			expectProvider: config.ProviderOpenAI,
			allowOverflow:  true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			fix := fixtures.Parse(t, tc.fixture)
			upstream := newMockUpstream(t, ctx, newFixtureResponse(fix))
			upstream.AllowOverflow = tc.allowOverflow

			m := aibridge.NewMetrics(prometheus.NewRegistry())
			bridgeServer := newBridgeTestServer(t, ctx, upstream.URL,
				withMetrics(m),
			)

			resp := bridgeServer.makeRequest(t, http.MethodPost, tc.path, fix.Request())
			_, _ = io.ReadAll(resp.Body)

			count := promtest.ToFloat64(m.InterceptionCount.WithLabelValues(
				tc.expectProvider, tc.expectModel, tc.expectStatus, tc.expectRoute, "POST", defaultActorID))
			require.Equal(t, 1.0, count)
			require.Equal(t, 1, promtest.CollectAndCount(m.InterceptionDuration))
			require.Equal(t, 1, promtest.CollectAndCount(m.InterceptionCount))
		})
	}
}

func TestMetrics_InterceptionsInflight(t *testing.T) {
	t.Parallel()

	fix := fixtures.Parse(t, fixtures.AntSimple)

	ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
	t.Cleanup(cancel)

	blockCh := make(chan struct{})

	// Setup a mock HTTP server which blocks until the request is marked as inflight then proceeds.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-blockCh
	}))
	t.Cleanup(srv.Close)

	m := aibridge.NewMetrics(prometheus.NewRegistry())
	bridgeServer := newBridgeTestServer(t, ctx, srv.URL,
		withMetrics(m),
	)

	// Make request in background.
	doneCh := make(chan struct{})
	go func() {
		defer close(doneCh)
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, bridgeServer.URL+pathAnthropicMessages, bytes.NewReader(fix.Request()))
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			defer resp.Body.Close()
			_, _ = io.ReadAll(resp.Body)
		}
	}()

	// Wait until request is detected as inflight.
	require.Eventually(t, func() bool {
		return promtest.ToFloat64(
			m.InterceptionsInflight.WithLabelValues(config.ProviderAnthropic, "claude-sonnet-4-0", "/v1/messages"),
		) == 1
	}, time.Second*10, time.Millisecond*50)

	// Unblock request, await completion.
	close(blockCh)
	select {
	case <-doneCh:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}

	// Metric is not updated immediately after request completes, so wait until it is.
	require.Eventually(t, func() bool {
		return promtest.ToFloat64(
			m.InterceptionsInflight.WithLabelValues(config.ProviderAnthropic, "claude-sonnet-4-0", "/v1/messages"),
		) == 0
	}, time.Second*10, time.Millisecond*50)
}

func TestMetrics_PassthroughCount(t *testing.T) {
	t.Parallel()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	t.Cleanup(upstream.Close)

	m := aibridge.NewMetrics(prometheus.NewRegistry())
	bridgeServer := newBridgeTestServer(t, t.Context(), upstream.URL,
		withMetrics(m),
	)

	resp := bridgeServer.makeRequest(t, http.MethodGet, "/openai/v1/models", nil)
	require.Equal(t, http.StatusOK, resp.StatusCode)

	count := promtest.ToFloat64(m.PassthroughCount.WithLabelValues(
		config.ProviderOpenAI, "/models", "GET"))
	require.Equal(t, 1.0, count)
}

func TestMetrics_PromptCount(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
	t.Cleanup(cancel)

	fix := fixtures.Parse(t, fixtures.OaiChatSimple)
	upstream := newMockUpstream(t, ctx, newFixtureResponse(fix))

	m := aibridge.NewMetrics(prometheus.NewRegistry())
	bridgeServer := newBridgeTestServer(t, ctx, upstream.URL,
		withMetrics(m),
	)

	resp := bridgeServer.makeRequest(t, http.MethodPost, pathOpenAIChatCompletions, fix.Request())
	require.Equal(t, http.StatusOK, resp.StatusCode)
	_, _ = io.ReadAll(resp.Body)

	prompts := promtest.ToFloat64(m.PromptCount.WithLabelValues(
		config.ProviderOpenAI, "gpt-4.1", defaultActorID))
	require.Equal(t, 1.0, prompts)
}

func TestMetrics_NonInjectedToolUseCount(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
	t.Cleanup(cancel)

	fix := fixtures.Parse(t, fixtures.OaiChatSingleBuiltinTool)
	upstream := newMockUpstream(t, ctx, newFixtureResponse(fix))

	m := aibridge.NewMetrics(prometheus.NewRegistry())
	bridgeServer := newBridgeTestServer(t, ctx, upstream.URL,
		withMetrics(m),
	)

	resp := bridgeServer.makeRequest(t, http.MethodPost, pathOpenAIChatCompletions, fix.Request())
	require.Equal(t, http.StatusOK, resp.StatusCode)
	_, _ = io.ReadAll(resp.Body)

	count := promtest.ToFloat64(m.NonInjectedToolUseCount.WithLabelValues(
		config.ProviderOpenAI, "gpt-4.1", "read_file"))
	require.Equal(t, 1.0, count)
}

func TestMetrics_InjectedToolUseCount(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
	t.Cleanup(cancel)

	// First request returns the tool invocation, the second returns the mocked response to the tool result.
	fix := fixtures.Parse(t, fixtures.AntSingleInjectedTool)
	upstream := newMockUpstream(t, ctx, newFixtureResponse(fix), newFixtureToolResponse(fix))

	m := aibridge.NewMetrics(prometheus.NewRegistry())

	// Setup mocked MCP server & tools.
	mockMCP := setupMCPForTest(t, defaultTracer)

	bridgeServer := newBridgeTestServer(t, ctx, upstream.URL,
		withMetrics(m),
		withMCP(mockMCP),
	)

	resp := bridgeServer.makeRequest(t, http.MethodPost, pathAnthropicMessages, fix.Request())
	require.Equal(t, http.StatusOK, resp.StatusCode)
	_, _ = io.ReadAll(resp.Body)

	// Wait until full roundtrip has completed.
	require.Eventually(t, func() bool {
		return upstream.Calls.Load() == 2
	}, time.Second*10, time.Millisecond*50)

	recorder := bridgeServer.Recorder
	require.Len(t, recorder.ToolUsages(), 1)
	require.True(t, recorder.ToolUsages()[0].Injected)
	require.NotNil(t, recorder.ToolUsages()[0].ServerURL)
	actualServerURL := *recorder.ToolUsages()[0].ServerURL

	count := promtest.ToFloat64(m.InjectedToolUseCount.WithLabelValues(
		config.ProviderAnthropic, "claude-sonnet-4-20250514", actualServerURL, mockToolName))
	require.Equal(t, 1.0, count)
}
