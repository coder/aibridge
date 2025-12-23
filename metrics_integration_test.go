package aibridge_test

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"cdr.dev/slog"
	"cdr.dev/slog/sloggers/slogtest"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/testutil"
	"github.com/prometheus/client_golang/prometheus"
	promtest "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace"
)

func TestMetrics_Interception(t *testing.T) {
	t.Parallel()

	cases := []struct {
		fixture        []byte
		expectedStatus string
	}{
		{
			fixture:        antSimple,
			expectedStatus: aibridge.InterceptionCountStatusCompleted,
		},
		{
			fixture:        antNonStreamErr,
			expectedStatus: aibridge.InterceptionCountStatusFailed,
		},
	}

	for _, tc := range cases {
		fixture := testutil.MustParseTXTAR(t, tc.fixture)
		llm := testutil.MustLLMFixture(t, fixture)

		ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
		t.Cleanup(cancel)

		upstream := testutil.NewUpstreamServer(t, ctx, llm)

		metrics := aibridge.NewMetrics(prometheus.NewRegistry())
		provider := aibridge.NewAnthropicProvider(anthropicCfg(upstream.URL, apiKey), nil)
		bridgeSrv, _ := newTestSrv(t, ctx, provider, metrics, testTracer)

		req := bridgeSrv.NewProviderRequest(t, provider.Name(), fixture.MustFile(t, testutil.FixtureRequest))
		resp, err := bridgeSrv.Client.Do(req)
		require.NoError(t, err)
		defer resp.Body.Close()
		_, _ = io.ReadAll(resp.Body)

		count := promtest.ToFloat64(metrics.InterceptionCount.WithLabelValues(
			aibridge.ProviderAnthropic, "claude-sonnet-4-0", tc.expectedStatus, "/v1/messages", "POST", userID))
		require.Equal(t, 1.0, count)
		require.Equal(t, 1, promtest.CollectAndCount(metrics.InterceptionDuration))
	}
}

func TestMetrics_InterceptionsInflight(t *testing.T) {
	t.Parallel()

	fixture := testutil.MustParseTXTAR(t, antSimple)
	llm := testutil.MustLLMFixture(t, fixture)

	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	t.Cleanup(cancel)

	upstream := testutil.NewUpstreamServer(t, ctx, llm)

	blockCh := make(chan struct{})

	// Setup a mock HTTP server which blocks until the request is marked as inflight then proceeds.
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-blockCh
		upstream.Config.Handler.ServeHTTP(w, r)
	}))
	srv.Config.BaseContext = func(_ net.Listener) context.Context {
		return ctx
	}
	srv.Start()
	t.Cleanup(srv.Close)

	metrics := aibridge.NewMetrics(prometheus.NewRegistry())
	provider := aibridge.NewAnthropicProvider(anthropicCfg(srv.URL, apiKey), nil)
	bridgeSrv, _ := newTestSrv(t, ctx, provider, metrics, testTracer)

	req := bridgeSrv.NewProviderRequest(t, provider.Name(), fixture.MustFile(t, testutil.FixtureRequest))

	// Make request in background.
	doneCh := make(chan struct{})
	go func() {
		defer close(doneCh)
		resp, err := bridgeSrv.Client.Do(req)
		if err == nil {
			defer resp.Body.Close()
			_, _ = io.ReadAll(resp.Body)
		}
	}()

	// Wait until request is detected as inflight.
	require.Eventually(t, func() bool {
		return promtest.ToFloat64(
			metrics.InterceptionsInflight.WithLabelValues(aibridge.ProviderAnthropic, "claude-sonnet-4-0", "/v1/messages"),
		) == 1
	}, 10*time.Second, 50*time.Millisecond)

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
			metrics.InterceptionsInflight.WithLabelValues(aibridge.ProviderAnthropic, "claude-sonnet-4-0", "/v1/messages"),
		) == 0
	}, 10*time.Second, 50*time.Millisecond)
}

func TestMetrics_PassthroughCount(t *testing.T) {
	t.Parallel()

	fixture := testutil.MustParseTXTAR(t, oaiFallthrough)
	respBody := fixture.MustFile(t, testutil.FixtureResponse)

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(respBody)
	}))
	t.Cleanup(upstream.Close)

	metrics := aibridge.NewMetrics(prometheus.NewRegistry())
	provider := aibridge.NewOpenAIProvider(openaiCfg(upstream.URL, apiKey))
	bridgeSrv, _ := newTestSrv(t, t.Context(), provider, metrics, testTracer)

	req, err := http.NewRequestWithContext(t.Context(), "GET", bridgeSrv.URL+"/openai/v1/models", nil)
	require.NoError(t, err)

	resp, err := bridgeSrv.Client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)

	count := promtest.ToFloat64(metrics.PassthroughCount.WithLabelValues(
		aibridge.ProviderOpenAI, "/v1/models", "GET"))
	require.Equal(t, 1.0, count)
}

func TestMetrics_PromptCount(t *testing.T) {
	t.Parallel()

	fixture := testutil.MustParseTXTAR(t, oaiSimple)
	llm := testutil.MustLLMFixture(t, fixture)

	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	t.Cleanup(cancel)

	upstream := testutil.NewUpstreamServer(t, ctx, llm)

	metrics := aibridge.NewMetrics(prometheus.NewRegistry())
	provider := aibridge.NewOpenAIProvider(openaiCfg(upstream.URL, apiKey))
	bridgeSrv, _ := newTestSrv(t, ctx, provider, metrics, testTracer)

	req := bridgeSrv.NewProviderRequest(t, provider.Name(), fixture.MustFile(t, testutil.FixtureRequest))
	resp, err := bridgeSrv.Client.Do(req)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode)
	defer resp.Body.Close()
	_, _ = io.ReadAll(resp.Body)

	prompts := promtest.ToFloat64(metrics.PromptCount.WithLabelValues(
		aibridge.ProviderOpenAI, "gpt-4.1", userID))
	require.Equal(t, 1.0, prompts)
}

func TestMetrics_NonInjectedToolUseCount(t *testing.T) {
	t.Parallel()

	fixture := testutil.MustParseTXTAR(t, oaiSingleBuiltinTool)
	llm := testutil.MustLLMFixture(t, fixture)

	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	t.Cleanup(cancel)

	upstream := testutil.NewUpstreamServer(t, ctx, llm)

	metrics := aibridge.NewMetrics(prometheus.NewRegistry())
	provider := aibridge.NewOpenAIProvider(openaiCfg(upstream.URL, apiKey))
	bridgeSrv, _ := newTestSrv(t, ctx, provider, metrics, testTracer)

	req := bridgeSrv.NewProviderRequest(t, provider.Name(), fixture.MustFile(t, testutil.FixtureRequest))
	resp, err := bridgeSrv.Client.Do(req)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode)
	defer resp.Body.Close()
	_, _ = io.ReadAll(resp.Body)

	count := promtest.ToFloat64(metrics.NonInjectedToolUseCount.WithLabelValues(
		aibridge.ProviderOpenAI, "gpt-4.1", "read_file"))
	require.Equal(t, 1.0, count)
}

func TestMetrics_InjectedToolUseCount(t *testing.T) {
	t.Parallel()

	fixture := testutil.MustParseTXTAR(t, antSingleInjectedTool)
	llm := testutil.MustLLMFixture(t, fixture)

	ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
	t.Cleanup(cancel)

	upstream := testutil.NewUpstreamServer(t, ctx, llm)

	logger := slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)
	metrics := aibridge.NewMetrics(prometheus.NewRegistry())
	provider := aibridge.NewAnthropicProvider(anthropicCfg(upstream.URL, apiKey), nil)

	// Setup mocked MCP server & tools.
	mcpSrv := testutil.NewMCPServer(t, testutil.DefaultCoderToolNames())
	mcpProxiers := mcpSrv.Proxiers(t, "coder", logger, testTracer)

	recorder := &testutil.RecorderSpy{}
	bridge := testutil.NewBridgeServer(t, testutil.BridgeConfig{
		Ctx:         ctx,
		ActorID:     userID,
		Providers:   []aibridge.Provider{provider},
		Recorder:    recorder,
		MCPProxiers: mcpProxiers,
		Logger:      logger,
		Metrics:     metrics,
		Tracer:      testTracer,
	})

	reqBody := fixture.MustFile(t, testutil.FixtureRequest)
	req := bridge.NewProviderRequest(t, aibridge.ProviderAnthropic, reqBody)
	resp, err := bridge.Client.Do(req)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode)
	defer resp.Body.Close()
	_, _ = io.ReadAll(resp.Body)

	// Wait until full roundtrip has completed.
	upstream.RequireCallCountEventually(t, 2)

	toolUsages := recorder.RecordedToolUsages()
	require.Len(t, toolUsages, 1)
	require.True(t, toolUsages[0].Injected)
	require.NotNil(t, toolUsages[0].ServerURL)
	actualServerURL := *toolUsages[0].ServerURL

	count := promtest.ToFloat64(metrics.InjectedToolUseCount.WithLabelValues(
		aibridge.ProviderAnthropic, "claude-sonnet-4-20250514", actualServerURL, testutil.ToolCoderListWorkspaces))
	require.Equal(t, 1.0, count)
}

func newTestSrv(t *testing.T, ctx context.Context, provider aibridge.Provider, metrics *aibridge.Metrics, tracer trace.Tracer) (*testutil.BridgeServer, *testutil.RecorderSpy) {
	t.Helper()

	logger := slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)
	spy := &testutil.RecorderSpy{}
	clientFn := func() (aibridge.Recorder, error) {
		return spy, nil
	}
	wrappedRecorder := aibridge.NewRecorder(logger, tracer, clientFn)

	bridgeSrv := testutil.NewBridgeServer(t, testutil.BridgeConfig{
		Ctx:       ctx,
		ActorID:   userID,
		Providers: []aibridge.Provider{provider},
		Recorder:  wrappedRecorder,
		Logger:    logger,
		Metrics:   metrics,
		Tracer:    tracer,
	})

	return bridgeSrv, spy
}
