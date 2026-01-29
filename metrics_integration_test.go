package aibridge_test

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"cdr.dev/slog/v3"
	"cdr.dev/slog/v3/sloggers/slogtest"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/config"
	aibcontext "github.com/coder/aibridge/context"
	"github.com/coder/aibridge/fixtures"
	"github.com/coder/aibridge/internal/testutil"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/metrics"
	"github.com/coder/aibridge/provider"
	"github.com/prometheus/client_golang/prometheus"
	promtest "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/tools/txtar"
)

func TestMetrics_Interception(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name           string
		fixture        []byte
		reqFunc        func(*testing.T, string, []byte) *http.Request
		expectStatus   string
		expectModel    string
		expectRoute    string
		expectProvider string
	}{
		{
			name:           "ant_simple",
			fixture:        fixtures.AntSimple,
			reqFunc:        createAnthropicMessagesReq,
			expectStatus:   metrics.InterceptionCountStatusCompleted,
			expectModel:    "claude-sonnet-4-0",
			expectRoute:    "/v1/messages",
			expectProvider: config.ProviderAnthropic,
		},
		{
			name:           "ant_error",
			fixture:        fixtures.AntNonStreamError,
			reqFunc:        createAnthropicMessagesReq,
			expectStatus:   metrics.InterceptionCountStatusFailed,
			expectModel:    "claude-sonnet-4-0",
			expectRoute:    "/v1/messages",
			expectProvider: config.ProviderAnthropic,
		},
		{
			name:           "oai_chat_simple",
			fixture:        fixtures.OaiChatSimple,
			reqFunc:        createOpenAIChatCompletionsReq,
			expectStatus:   metrics.InterceptionCountStatusCompleted,
			expectModel:    "gpt-4.1",
			expectRoute:    "/v1/chat/completions",
			expectProvider: config.ProviderOpenAI,
		},
		{
			name:           "oai_chat_error",
			fixture:        fixtures.OaiChatNonStreamError,
			reqFunc:        createOpenAIChatCompletionsReq,
			expectStatus:   metrics.InterceptionCountStatusFailed,
			expectModel:    "gpt-4.1",
			expectRoute:    "/v1/chat/completions",
			expectProvider: config.ProviderOpenAI,
		},
		{
			name:           "oai_responses_blocking_simple",
			fixture:        fixtures.OaiResponsesBlockingSimple,
			reqFunc:        createOpenAIResponsesReq,
			expectStatus:   metrics.InterceptionCountStatusCompleted,
			expectModel:    "gpt-4o-mini",
			expectRoute:    "/v1/responses",
			expectProvider: config.ProviderOpenAI,
		},
		{
			name:           "oai_responses_blocking_error",
			fixture:        fixtures.OaiResponsesBlockingHttpErr,
			reqFunc:        createOpenAIResponsesReq,
			expectStatus:   metrics.InterceptionCountStatusFailed,
			expectModel:    "gpt-4o-mini",
			expectRoute:    "/v1/responses",
			expectProvider: config.ProviderOpenAI,
		},
		{
			name:           "oai_responses_streaming_simple",
			fixture:        fixtures.OaiResponsesStreamingSimple,
			reqFunc:        createOpenAIResponsesReq,
			expectStatus:   metrics.InterceptionCountStatusCompleted,
			expectModel:    "gpt-4o-mini",
			expectRoute:    "/v1/responses",
			expectProvider: config.ProviderOpenAI,
		},
		{
			name:           "oai_responses_streaming_error",
			fixture:        fixtures.OaiResponsesStreamingHttpErr,
			reqFunc:        createOpenAIResponsesReq,
			expectStatus:   metrics.InterceptionCountStatusFailed,
			expectModel:    "gpt-4o-mini",
			expectRoute:    "/v1/responses",
			expectProvider: config.ProviderOpenAI,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			arc := txtar.Parse(tc.fixture)
			files := filesMap(arc)

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			mockAPI := newMockServer(ctx, t, files, nil, nil)
			t.Cleanup(mockAPI.Close)

			metrics := aibridge.NewMetrics(prometheus.NewRegistry())
			var prov aibridge.Provider
			if tc.expectProvider == config.ProviderAnthropic {
				prov = provider.NewAnthropic(anthropicCfg(mockAPI.URL, apiKey), nil)
			} else {
				prov = provider.NewOpenAI(openaiCfg(mockAPI.URL, apiKey))
			}
			srv, _ := newTestSrv(t, ctx, prov, metrics, testTracer)

			req := tc.reqFunc(t, srv.URL, files[fixtureRequest])
			resp, err := http.DefaultClient.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()
			_, _ = io.ReadAll(resp.Body)

			count := promtest.ToFloat64(metrics.InterceptionCount.WithLabelValues(
				tc.expectProvider, tc.expectModel, tc.expectStatus, tc.expectRoute, "POST", userID))
			require.Equal(t, 1.0, count)
			require.Equal(t, 1, promtest.CollectAndCount(metrics.InterceptionDuration))
			require.Equal(t, 1, promtest.CollectAndCount(metrics.InterceptionCount))
		})
	}
}

func TestMetrics_InterceptionsInflight(t *testing.T) {
	t.Parallel()

	arc := txtar.Parse(fixtures.AntSimple)
	files := filesMap(arc)

	ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
	t.Cleanup(cancel)

	blockCh := make(chan struct{})

	// Setup a mock HTTP server which blocks until the request is marked as inflight then proceeds.
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-blockCh
		mock := newMockServer(ctx, t, files, nil, nil)
		defer mock.Close()
		mock.Server.Config.Handler.ServeHTTP(w, r)
	}))
	srv.Config.BaseContext = func(_ net.Listener) context.Context {
		return ctx
	}
	srv.Start()
	t.Cleanup(srv.Close)

	metrics := aibridge.NewMetrics(prometheus.NewRegistry())
	provider := provider.NewAnthropic(anthropicCfg(srv.URL, apiKey), nil)
	bridgeSrv, _ := newTestSrv(t, ctx, provider, metrics, testTracer)

	// Make request in background.
	doneCh := make(chan struct{})
	go func() {
		defer close(doneCh)
		req := createAnthropicMessagesReq(t, bridgeSrv.URL, files[fixtureRequest])
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			defer resp.Body.Close()
			_, _ = io.ReadAll(resp.Body)
		}
	}()

	// Wait until request is detected as inflight.
	require.Eventually(t, func() bool {
		return promtest.ToFloat64(
			metrics.InterceptionsInflight.WithLabelValues(config.ProviderAnthropic, "claude-sonnet-4-0", "/v1/messages"),
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
			metrics.InterceptionsInflight.WithLabelValues(config.ProviderAnthropic, "claude-sonnet-4-0", "/v1/messages"),
		) == 0
	}, time.Second*10, time.Millisecond*50)
}

func TestMetrics_PassthroughCount(t *testing.T) {
	t.Parallel()

	arc := txtar.Parse(fixtures.OaiChatFallthrough)
	files := filesMap(arc)

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(files[fixtureResponse])
	}))
	t.Cleanup(upstream.Close)

	metrics := aibridge.NewMetrics(prometheus.NewRegistry())
	provider := provider.NewOpenAI(openaiCfg(upstream.URL, apiKey))
	srv, _ := newTestSrv(t, t.Context(), provider, metrics, testTracer)

	req, err := http.NewRequestWithContext(t.Context(), "GET", srv.URL+"/openai/v1/models", nil)
	require.NoError(t, err)

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)

	count := promtest.ToFloat64(metrics.PassthroughCount.WithLabelValues(
		config.ProviderOpenAI, "/v1/models", "GET"))
	require.Equal(t, 1.0, count)
}

func TestMetrics_PromptCount(t *testing.T) {
	t.Parallel()

	arc := txtar.Parse(fixtures.OaiChatSimple)
	files := filesMap(arc)

	ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
	t.Cleanup(cancel)

	mockAPI := newMockServer(ctx, t, files, nil, nil)
	t.Cleanup(mockAPI.Close)

	metrics := aibridge.NewMetrics(prometheus.NewRegistry())
	provider := provider.NewOpenAI(openaiCfg(mockAPI.URL, apiKey))
	srv, _ := newTestSrv(t, ctx, provider, metrics, testTracer)

	req := createOpenAIChatCompletionsReq(t, srv.URL, files[fixtureRequest])
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode)
	defer resp.Body.Close()
	_, _ = io.ReadAll(resp.Body)

	prompts := promtest.ToFloat64(metrics.PromptCount.WithLabelValues(
		config.ProviderOpenAI, "gpt-4.1", userID))
	require.Equal(t, 1.0, prompts)
}

func TestMetrics_NonInjectedToolUseCount(t *testing.T) {
	t.Parallel()

	arc := txtar.Parse(fixtures.OaiChatSingleBuiltinTool)
	files := filesMap(arc)

	ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
	t.Cleanup(cancel)

	mockAPI := newMockServer(ctx, t, files, nil, nil)
	t.Cleanup(mockAPI.Close)

	metrics := aibridge.NewMetrics(prometheus.NewRegistry())
	provider := provider.NewOpenAI(openaiCfg(mockAPI.URL, apiKey))
	srv, _ := newTestSrv(t, ctx, provider, metrics, testTracer)

	req := createOpenAIChatCompletionsReq(t, srv.URL, files[fixtureRequest])
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode)
	defer resp.Body.Close()
	_, _ = io.ReadAll(resp.Body)

	count := promtest.ToFloat64(metrics.NonInjectedToolUseCount.WithLabelValues(
		config.ProviderOpenAI, "gpt-4.1", "read_file"))
	require.Equal(t, 1.0, count)
}

func TestMetrics_InjectedToolUseCount(t *testing.T) {
	t.Parallel()

	arc := txtar.Parse(fixtures.AntSingleInjectedTool)
	files := filesMap(arc)

	ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
	t.Cleanup(cancel)

	// First request returns the tool invocation, the second returns the mocked response to the tool result.
	mockAPI := newMockServer(ctx, t, files, nil, func(reqCount uint32, resp []byte) []byte {
		if reqCount == 1 {
			return resp
		}
		return files[fixtureNonStreamingToolResponse]
	})
	t.Cleanup(mockAPI.Close)

	recorder := &testutil.MockRecorder{}
	logger := slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)
	metrics := aibridge.NewMetrics(prometheus.NewRegistry())
	provider := provider.NewAnthropic(anthropicCfg(mockAPI.URL, apiKey), nil)

	// Setup mocked MCP server & tools.
	mcpProxiers, _ := setupMCPServerProxiesForTest(t, testTracer)
	mcpMgr := mcp.NewServerProxyManager(mcpProxiers, testTracer)
	require.NoError(t, mcpMgr.Init(ctx))

	bridge, err := aibridge.NewRequestBridge(ctx, []aibridge.Provider{provider}, recorder, mcpMgr, logger, metrics, testTracer)
	require.NoError(t, err)

	srv := httptest.NewUnstartedServer(bridge)
	srv.Config.BaseContext = func(_ net.Listener) context.Context {
		return aibcontext.AsActor(ctx, userID, nil)
	}
	srv.Start()
	t.Cleanup(srv.Close)

	req := createAnthropicMessagesReq(t, srv.URL, files[fixtureRequest])
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode)
	defer resp.Body.Close()
	_, _ = io.ReadAll(resp.Body)

	// Wait until full roundtrip has completed.
	require.Eventually(t, func() bool {
		return mockAPI.callCount.Load() == 2
	}, time.Second*10, time.Millisecond*50)

	require.Len(t, recorder.ToolUsages(), 1)
	require.True(t, recorder.ToolUsages()[0].Injected)
	require.NotNil(t, recorder.ToolUsages()[0].ServerURL)
	actualServerURL := *recorder.ToolUsages()[0].ServerURL

	count := promtest.ToFloat64(metrics.InjectedToolUseCount.WithLabelValues(
		config.ProviderAnthropic, "claude-sonnet-4-20250514", actualServerURL, mockToolName))
	require.Equal(t, 1.0, count)
}

func newTestSrv(t *testing.T, ctx context.Context, provider aibridge.Provider, metrics *metrics.Metrics, tracer trace.Tracer) (*httptest.Server, *testutil.MockRecorder) {
	t.Helper()

	logger := slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)
	mockRecorder := &testutil.MockRecorder{}
	clientFn := func() (aibridge.Recorder, error) {
		return mockRecorder, nil
	}
	wrappedRecorder := aibridge.NewRecorder(logger, tracer, clientFn)

	bridge, err := aibridge.NewRequestBridge(ctx, []aibridge.Provider{provider}, wrappedRecorder, mcp.NewServerProxyManager(nil, testTracer), logger, metrics, tracer)
	require.NoError(t, err)

	srv := httptest.NewUnstartedServer(bridge)
	srv.Config.BaseContext = func(_ net.Listener) context.Context {
		return aibcontext.AsActor(ctx, userID, nil)
	}
	srv.Start()
	t.Cleanup(srv.Close)

	return srv, mockRecorder
}
