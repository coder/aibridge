package aibridge_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"cdr.dev/slog/v3"
	"cdr.dev/slog/v3/sloggers/slogtest"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/provider"
	"github.com/coder/aibridge/tracing"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"golang.org/x/tools/txtar"
)

// expect 'count' amount of traces named 'name' with status 'status'
type expectTrace struct {
	name   string
	count  int
	status codes.Code
}

func TestTraceAnthropic(t *testing.T) {
	expectNonStreaming := []expectTrace{
		{"Intercept", 1, codes.Unset},
		{"Intercept.CreateInterceptor", 1, codes.Unset},
		{"Intercept.RecordInterception", 1, codes.Unset},
		{"Intercept.ProcessRequest", 1, codes.Unset},
		{"Intercept.RecordInterceptionEnded", 1, codes.Unset},
		{"Intercept.RecordPromptUsage", 1, codes.Unset},
		{"Intercept.RecordTokenUsage", 1, codes.Unset},
		{"Intercept.RecordToolUsage", 1, codes.Unset},
		{"Intercept.ProcessRequest.Upstream", 1, codes.Unset},
	}

	expectStreaming := []expectTrace{
		{"Intercept", 1, codes.Unset},
		{"Intercept.CreateInterceptor", 1, codes.Unset},
		{"Intercept.RecordInterception", 1, codes.Unset},
		{"Intercept.ProcessRequest", 1, codes.Unset},
		{"Intercept.RecordInterceptionEnded", 1, codes.Unset},
		{"Intercept.RecordPromptUsage", 1, codes.Unset},
		{"Intercept.RecordTokenUsage", 2, codes.Unset},
		{"Intercept.RecordToolUsage", 1, codes.Unset},
		{"Intercept.ProcessRequest.Upstream", 1, codes.Unset},
	}

	cases := []struct {
		name      string
		streaming bool
		bedrock   bool
		expect    []expectTrace
	}{
		{
			name:   "trace_anthr_non_streaming",
			expect: expectNonStreaming,
		},
		{
			name:    "trace_bedrock_non_streaming",
			bedrock: true,
			expect:  expectNonStreaming,
		},
		{
			name:      "trace_anthr_streaming",
			streaming: true,
			expect:    expectStreaming,
		},
		{
			name:      "trace_bedrock_streaming",
			streaming: true,
			bedrock:   true,
			expect:    expectStreaming,
		},
	}

	arc := txtar.Parse(antSingleBuiltinTool)

	files := filesMap(arc)
	require.Contains(t, files, fixtureRequest)
	require.Contains(t, files, fixtureStreamingResponse)
	require.Contains(t, files, fixtureNonStreamingResponse)

	fixtureReqBody := files[fixtureRequest]

	for _, tc := range cases {
		t.Run(fmt.Sprintf("%s/streaming=%v", t.Name(), tc.streaming), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			sr := tracetest.NewSpanRecorder()
			tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
			tracer := tp.Tracer(t.Name())
			defer func() { _ = tp.Shutdown(t.Context()) }()

			reqBody, err := setJSON(fixtureReqBody, "stream", tc.streaming)
			require.NoError(t, err)

			mockAPI := newMockServer(ctx, t, files, nil)
			t.Cleanup(mockAPI.Close)

			var bedrockCfg *config.AWSBedrock
			if tc.bedrock {
				bedrockCfg = testBedrockCfg(mockAPI.URL)
			}
			provider := provider.NewAnthropic(anthropicCfg(mockAPI.URL, apiKey), bedrockCfg)
			srv, recorder := newTestSrv(t, ctx, provider, nil, tracer)

			req := createAnthropicMessagesReq(t, srv.URL, reqBody)
			client := &http.Client{}
			resp, err := client.Do(req)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, resp.StatusCode)
			defer resp.Body.Close()
			srv.Close()

			require.Equal(t, 1, len(recorder.interceptions))
			intcID := recorder.interceptions[0].ID

			model := gjson.Get(string(reqBody), "model").Str
			if tc.bedrock {
				model = "beddel"
			}

			totalCount := 0
			for _, e := range tc.expect {
				totalCount += e.count
			}

			attrs := []attribute.KeyValue{
				attribute.String(tracing.RequestPath, req.URL.Path),
				attribute.String(tracing.InterceptionID, intcID),
				attribute.String(tracing.Provider, config.ProviderAnthropic),
				attribute.String(tracing.Model, model),
				attribute.String(tracing.InitiatorID, userID),
				attribute.Bool(tracing.Streaming, tc.streaming),
				attribute.Bool(tracing.IsBedrock, tc.bedrock),
			}

			require.Len(t, sr.Ended(), totalCount)
			verifyTraces(t, sr, tc.expect, attrs)
		})
	}
}

func TestTraceAnthropicErr(t *testing.T) {
	expectNonStream := []expectTrace{
		{"Intercept", 1, codes.Error},
		{"Intercept.CreateInterceptor", 1, codes.Unset},
		{"Intercept.RecordInterception", 1, codes.Unset},
		{"Intercept.ProcessRequest", 1, codes.Error},
		{"Intercept.RecordInterceptionEnded", 1, codes.Unset},
		{"Intercept.ProcessRequest.Upstream", 1, codes.Error},
	}

	expectStreaming := []expectTrace{
		{"Intercept", 1, codes.Error},
		{"Intercept.CreateInterceptor", 1, codes.Unset},
		{"Intercept.RecordInterception", 1, codes.Unset},
		{"Intercept.ProcessRequest", 1, codes.Error},
		{"Intercept.RecordPromptUsage", 1, codes.Unset},
		{"Intercept.RecordTokenUsage", 1, codes.Unset},
		{"Intercept.RecordInterceptionEnded", 1, codes.Unset},
		{"Intercept.ProcessRequest.Upstream", 1, codes.Unset},
	}

	cases := []struct {
		name      string
		streaming bool
		bedrock   bool
		expect    []expectTrace
	}{
		{
			name:   "anthr_non_streaming_err",
			expect: expectNonStream,
		},
		{
			name:      "anthr_streaming_err",
			streaming: true,
			expect:    expectStreaming,
		},
		{
			name:    "bedrock_non_streaming_err",
			bedrock: true,
			expect:  expectNonStream,
		},
		{
			name:      "bedrock_streaming_err",
			streaming: true,
			bedrock:   true,
			expect:    expectStreaming,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			var arc *txtar.Archive
			if tc.streaming {
				arc = txtar.Parse(antMidStreamErr)
			} else {
				arc = txtar.Parse(antNonStreamErr)
			}

			files := filesMap(arc)
			require.Contains(t, files, fixtureRequest)
			if tc.streaming {
				require.Contains(t, files, fixtureStreamingResponse)
			} else {
				require.Contains(t, files, fixtureNonStreamingResponse)
			}

			fixtureReqBody := files[fixtureRequest]

			sr := tracetest.NewSpanRecorder()
			tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
			tracer := tp.Tracer(t.Name())
			defer func() { _ = tp.Shutdown(t.Context()) }()

			reqBody, err := setJSON(fixtureReqBody, "stream", tc.streaming)
			require.NoError(t, err)

			mockAPI := newMockServer(ctx, t, files, nil)
			t.Cleanup(mockAPI.Close)

			var bedrockCfg *config.AWSBedrock
			if tc.bedrock {
				bedrockCfg = testBedrockCfg(mockAPI.URL)
			}
			provider := provider.NewAnthropic(anthropicCfg(mockAPI.URL, apiKey), bedrockCfg)
			srv, recorder := newTestSrv(t, ctx, provider, nil, tracer)

			req := createAnthropicMessagesReq(t, srv.URL, reqBody)
			client := &http.Client{}
			resp, err := client.Do(req)
			require.NoError(t, err)
			if tc.streaming {
				require.Equal(t, http.StatusOK, resp.StatusCode)
			} else {
				require.Equal(t, http.StatusInternalServerError, resp.StatusCode)
			}
			defer resp.Body.Close()
			srv.Close()

			require.Equal(t, 1, len(recorder.interceptions))
			intcID := recorder.interceptions[0].ID

			totalCount := 0
			for _, e := range tc.expect {
				totalCount += e.count
			}
			for _, s := range sr.Ended() {
				t.Logf("SPAN: %v", s.Name())
			}
			require.Len(t, sr.Ended(), totalCount)

			model := gjson.Get(string(reqBody), "model").Str
			if tc.bedrock {
				model = "beddel"
			}

			attrs := []attribute.KeyValue{
				attribute.String(tracing.RequestPath, req.URL.Path),
				attribute.String(tracing.InterceptionID, intcID),
				attribute.String(tracing.Provider, config.ProviderAnthropic),
				attribute.String(tracing.Model, model),
				attribute.String(tracing.InitiatorID, userID),
				attribute.Bool(tracing.Streaming, tc.streaming),
				attribute.Bool(tracing.IsBedrock, tc.bedrock),
			}

			verifyTraces(t, sr, tc.expect, attrs)
		})
	}
}

func TestAnthropicInjectedToolsTrace(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		streaming bool
		bedrock   bool
	}{
		{
			name:      "anthr_blocking",
			streaming: false,
			bedrock:   false,
		},
		{
			name:      "anthr_streaming",
			streaming: true,
			bedrock:   false,
		},
		{
			name:      "bedrock_blocking",
			streaming: false,
			bedrock:   true,
		},
		{
			name:      "bedrock_streaming",
			streaming: true,
			bedrock:   true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			sr := tracetest.NewSpanRecorder()
			tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
			tracer := tp.Tracer(t.Name())
			defer func() { _ = tp.Shutdown(t.Context()) }()

			configureFn := func(addr string, client aibridge.Recorder, srvProxyMgr *mcp.ServerProxyManager) (*aibridge.RequestBridge, error) {
				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
				var bedrockCfg *config.AWSBedrock
				if tc.bedrock {
					bedrockCfg = testBedrockCfg(addr)
				}
				providers := []aibridge.Provider{provider.NewAnthropic(anthropicCfg(addr, apiKey), bedrockCfg)}
				return aibridge.NewRequestBridge(t.Context(), providers, client, srvProxyMgr, logger, nil, tracer)
			}

			var reqBody string
			var reqPath string
			reqFunc := func(t *testing.T, baseURL string, input []byte) *http.Request {
				reqBody = string(input)
				r := createAnthropicMessagesReq(t, baseURL, input)
				reqPath = r.URL.Path
				return r
			}

			// Build the requirements & make the assertions which are common to all providers.
			recorderClient, _, proxies, resp := setupInjectedToolTest(t, antSingleInjectedTool, tc.streaming, configureFn, reqFunc)

			defer resp.Body.Close()

			require.Len(t, recorderClient.interceptions, 1)
			intcID := recorderClient.interceptions[0].ID

			model := gjson.Get(string(reqBody), "model").Str
			if tc.bedrock {
				model = "beddel"
			}

			for _, proxy := range proxies {
				require.NotEmpty(t, proxy.ListTools())
				tool := proxy.ListTools()[0]

				attrs := []attribute.KeyValue{
					attribute.String(tracing.RequestPath, reqPath),
					attribute.String(tracing.InterceptionID, intcID),
					attribute.String(tracing.Provider, config.ProviderAnthropic),
					attribute.String(tracing.Model, model),
					attribute.String(tracing.InitiatorID, userID),
					attribute.String(tracing.MCPInput, "{\"owner\":\"admin\"}"),
					attribute.String(tracing.MCPToolName, "coder_list_workspaces"),
					attribute.String(tracing.MCPServerName, tool.ServerName),
					attribute.String(tracing.MCPServerURL, tool.ServerURL),
					attribute.Bool(tracing.Streaming, tc.streaming),
					attribute.Bool(tracing.IsBedrock, tc.bedrock),
				}
				verifyTraces(t, sr, []expectTrace{{"Intercept.ProcessRequest.ToolCall", 1, codes.Unset}}, attrs)
			}
		})
	}
}

func TestTraceOpenAI(t *testing.T) {
	cases := []struct {
		name      string
		fixture   []byte
		streaming bool
		expect    []expectTrace
	}{
		{
			name:      "trace_openai_streaming",
			fixture:   oaiSimple,
			streaming: true,
			expect: []expectTrace{
				{"Intercept", 1, codes.Unset},
				{"Intercept.CreateInterceptor", 1, codes.Unset},
				{"Intercept.RecordInterception", 1, codes.Unset},
				{"Intercept.ProcessRequest", 1, codes.Unset},
				{"Intercept.RecordInterceptionEnded", 1, codes.Unset},
				{"Intercept.RecordPromptUsage", 1, codes.Unset},
				{"Intercept.RecordTokenUsage", 1, codes.Unset},
				{"Intercept.ProcessRequest.Upstream", 1, codes.Unset},
			},
		},
		{
			name:      "trace_openai_non_streaming",
			fixture:   oaiSimple,
			streaming: false,
			expect: []expectTrace{
				{"Intercept", 1, codes.Unset},
				{"Intercept.CreateInterceptor", 1, codes.Unset},
				{"Intercept.RecordInterception", 1, codes.Unset},
				{"Intercept.ProcessRequest", 1, codes.Unset},
				{"Intercept.RecordInterceptionEnded", 1, codes.Unset},
				{"Intercept.RecordPromptUsage", 1, codes.Unset},
				{"Intercept.RecordTokenUsage", 1, codes.Unset},
				{"Intercept.ProcessRequest.Upstream", 1, codes.Unset},
			},
		},
	}

	for _, tc := range cases {
		t.Run(t.Name(), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			arc := txtar.Parse(tc.fixture)

			files := filesMap(arc)
			require.Contains(t, files, fixtureRequest)
			require.Contains(t, files, fixtureStreamingResponse)
			require.Contains(t, files, fixtureNonStreamingResponse)

			fixtureReqBody := files[fixtureRequest]

			sr := tracetest.NewSpanRecorder()
			tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
			tracer := tp.Tracer(t.Name())
			defer func() { _ = tp.Shutdown(t.Context()) }()

			reqBody, err := setJSON(fixtureReqBody, "stream", tc.streaming)
			require.NoError(t, err)

			mockAPI := newMockServer(ctx, t, files, nil)
			t.Cleanup(mockAPI.Close)
			provider := provider.NewOpenAI(openaiCfg(mockAPI.URL, apiKey))
			srv, recorder := newTestSrv(t, ctx, provider, nil, tracer)

			req := createOpenAIChatCompletionsReq(t, srv.URL, reqBody)
			client := &http.Client{}
			resp, err := client.Do(req)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, resp.StatusCode)
			defer resp.Body.Close()
			srv.Close()

			require.Equal(t, 1, len(recorder.interceptions))
			intcID := recorder.interceptions[0].ID

			totalCount := 0
			for _, e := range tc.expect {
				totalCount += e.count
			}
			require.Len(t, sr.Ended(), totalCount)

			attrs := []attribute.KeyValue{
				attribute.String(tracing.RequestPath, req.URL.Path),
				attribute.String(tracing.InterceptionID, intcID),
				attribute.String(tracing.Provider, config.ProviderOpenAI),
				attribute.String(tracing.Model, gjson.Get(string(reqBody), "model").Str),
				attribute.String(tracing.InitiatorID, userID),
				attribute.Bool(tracing.Streaming, tc.streaming),
			}
			verifyTraces(t, sr, tc.expect, attrs)
		})
	}
}

func TestTraceOpenAIErr(t *testing.T) {
	cases := []struct {
		name      string
		streaming bool
		expect    []expectTrace
	}{
		{
			name:      "trace_openai_streaming_err",
			streaming: true,
			expect: []expectTrace{
				{"Intercept", 1, codes.Error},
				{"Intercept.CreateInterceptor", 1, codes.Unset},
				{"Intercept.RecordInterception", 1, codes.Unset},
				{"Intercept.ProcessRequest", 1, codes.Error},
				{"Intercept.RecordInterceptionEnded", 1, codes.Unset},
				{"Intercept.RecordPromptUsage", 1, codes.Unset},
				{"Intercept.ProcessRequest.Upstream", 1, codes.Unset},
			},
		},
		{
			name:      "trace_openai_non_streaming_err",
			streaming: false,
			expect: []expectTrace{
				{"Intercept", 1, codes.Error},
				{"Intercept.CreateInterceptor", 1, codes.Unset},
				{"Intercept.RecordInterception", 1, codes.Unset},
				{"Intercept.ProcessRequest", 1, codes.Error},
				{"Intercept.RecordInterceptionEnded", 1, codes.Unset},
				{"Intercept.ProcessRequest.Upstream", 1, codes.Error},
			},
		},
	}

	for _, tc := range cases {
		t.Run(t.Name(), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			var arc *txtar.Archive
			if tc.streaming {
				arc = txtar.Parse(oaiMidStreamErr)
			} else {
				arc = txtar.Parse(oaiNonStreamErr)
			}

			files := filesMap(arc)
			require.Contains(t, files, fixtureRequest)
			if tc.streaming {
				require.Contains(t, files, fixtureStreamingResponse)
			} else {
				require.Contains(t, files, fixtureNonStreamingResponse)
			}

			fixtureReqBody := files[fixtureRequest]

			sr := tracetest.NewSpanRecorder()
			tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
			tracer := tp.Tracer(t.Name())
			defer func() { _ = tp.Shutdown(t.Context()) }()

			reqBody, err := setJSON(fixtureReqBody, "stream", tc.streaming)
			require.NoError(t, err)

			mockAPI := newMockServer(ctx, t, files, nil)
			t.Cleanup(mockAPI.Close)
			provider := provider.NewOpenAI(openaiCfg(mockAPI.URL, apiKey))
			srv, recorder := newTestSrv(t, ctx, provider, nil, tracer)

			req := createOpenAIChatCompletionsReq(t, srv.URL, reqBody)
			client := &http.Client{}
			resp, err := client.Do(req)
			require.NoError(t, err)
			if tc.streaming {
				require.Equal(t, http.StatusOK, resp.StatusCode)
			} else {
				require.Equal(t, http.StatusInternalServerError, resp.StatusCode)
			}
			defer resp.Body.Close()
			srv.Close()

			require.Equal(t, 1, len(recorder.interceptions))
			intcID := recorder.interceptions[0].ID

			totalCount := 0
			for _, e := range tc.expect {
				totalCount += e.count
			}
			require.Len(t, sr.Ended(), totalCount)

			attrs := []attribute.KeyValue{
				attribute.String(tracing.RequestPath, req.URL.Path),
				attribute.String(tracing.InterceptionID, intcID),
				attribute.String(tracing.Provider, config.ProviderOpenAI),
				attribute.String(tracing.Model, gjson.Get(string(reqBody), "model").Str),
				attribute.String(tracing.InitiatorID, userID),
				attribute.Bool(tracing.Streaming, tc.streaming),
			}
			verifyTraces(t, sr, tc.expect, attrs)
		})
	}
}

func TestOpenAIInjectedToolsTrace(t *testing.T) {
	t.Parallel()

	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming=%v", streaming), func(t *testing.T) {
			t.Parallel()

			sr := tracetest.NewSpanRecorder()
			tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
			tracer := tp.Tracer(t.Name())
			defer func() { _ = tp.Shutdown(t.Context()) }()

			configureFn := func(addr string, client aibridge.Recorder, srvProxyMgr *mcp.ServerProxyManager) (*aibridge.RequestBridge, error) {
				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
				providers := []aibridge.Provider{provider.NewOpenAI(openaiCfg(addr, apiKey))}
				return aibridge.NewRequestBridge(t.Context(), providers, client, srvProxyMgr, logger, nil, tracer)
			}

			var reqBody string
			var reqPath string
			reqFunc := func(t *testing.T, baseURL string, input []byte) *http.Request {
				reqBody = string(input)
				r := createOpenAIChatCompletionsReq(t, baseURL, input)
				reqPath = r.URL.Path
				return r
			}

			// Build the requirements & make the assertions which are common to all providers.
			recorderClient, _, proxies, resp := setupInjectedToolTest(t, oaiSingleInjectedTool, streaming, configureFn, reqFunc)

			defer resp.Body.Close()

			require.Len(t, recorderClient.interceptions, 1)
			intcID := recorderClient.interceptions[0].ID

			for _, proxy := range proxies {
				require.NotEmpty(t, proxy.ListTools())
				tool := proxy.ListTools()[0]

				attrs := []attribute.KeyValue{
					attribute.String(tracing.RequestPath, reqPath),
					attribute.String(tracing.InterceptionID, intcID),
					attribute.String(tracing.Provider, config.ProviderOpenAI),
					attribute.String(tracing.Model, gjson.Get(reqBody, "model").Str),
					attribute.String(tracing.InitiatorID, userID),
					attribute.String(tracing.MCPInput, "{\"owner\":\"admin\"}"),
					attribute.String(tracing.MCPToolName, "coder_list_workspaces"),
					attribute.String(tracing.MCPServerName, tool.ServerName),
					attribute.String(tracing.MCPServerURL, tool.ServerURL),
					attribute.Bool(tracing.Streaming, streaming),
				}
				verifyTraces(t, sr, []expectTrace{{"Intercept.ProcessRequest.ToolCall", 1, codes.Unset}}, attrs)
			}
		})
	}
}

func TestTracePassthrough(t *testing.T) {
	t.Parallel()

	arc := txtar.Parse(oaiFallthrough)
	files := filesMap(arc)

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(files[fixtureResponse])
	}))
	t.Cleanup(upstream.Close)

	sr := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
	tracer := tp.Tracer(t.Name())
	defer func() { _ = tp.Shutdown(t.Context()) }()

	provider := provider.NewOpenAI(openaiCfg(upstream.URL, apiKey))
	srv, _ := newTestSrv(t, t.Context(), provider, nil, tracer)

	req, err := http.NewRequestWithContext(t.Context(), "GET", srv.URL+"/openai/v1/models", nil)
	require.NoError(t, err)

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)
	srv.Close()

	spans := sr.Ended()
	require.Len(t, spans, 1)

	assert.Equal(t, spans[0].Name(), "Passthrough")
	want := []attribute.KeyValue{
		attribute.String(tracing.PassthroughMethod, "GET"),
		attribute.String(tracing.PassthroughURL, "/v1/models"),
	}
	got := slices.SortedFunc(slices.Values(spans[0].Attributes()), cmpAttrKeyVal)
	require.Equal(t, want, got)
}

func TestNewServerProxyManagerTraces(t *testing.T) {
	ctx := t.Context()

	sr := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
	tracer := tp.Tracer(t.Name())
	defer func() { _ = tp.Shutdown(t.Context()) }()

	serverName := "serverName"
	srv, _ := createMockMCPSrv(t)
	mcpSrv := httptest.NewServer(srv)
	t.Cleanup(mcpSrv.Close)

	logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
	proxy, err := mcp.NewStreamableHTTPServerProxy(serverName, mcpSrv.URL, nil, nil, nil, logger, tracer)
	require.NoError(t, err)
	tools := map[string]mcp.ServerProxier{"unusedValue": proxy}

	mcpMgr := mcp.NewServerProxyManager(tools, tracer)
	err = mcpMgr.Init(ctx)
	require.NoError(t, err)

	require.Len(t, sr.Ended(), 3)
	verifyTraces(t, sr, []expectTrace{{"ServerProxyManager.Init", 1, codes.Unset}}, []attribute.KeyValue{})

	attrs := []attribute.KeyValue{
		attribute.String(tracing.MCPProxyName, proxy.Name()),
		attribute.String(tracing.MCPServerURL, mcpSrv.URL),
		attribute.String(tracing.MCPServerName, serverName),
	}
	verifyTraces(t, sr, []expectTrace{{"StreamableHTTPServerProxy.Init", 1, codes.Unset}}, attrs)

	attrs = append(attrs, attribute.Int(tracing.MCPToolCount, len(proxy.ListTools())))
	verifyTraces(t, sr, []expectTrace{{"StreamableHTTPServerProxy.Init.fetchTools", 1, codes.Unset}}, attrs)
}

func cmpAttrKeyVal(a attribute.KeyValue, b attribute.KeyValue) int {
	return strings.Compare(string(a.Key), string(b.Key))
}

// checks counts of traces with given name, status and attributes
func verifyTraces(t *testing.T, spanRecorder *tracetest.SpanRecorder, expect []expectTrace, attrs []attribute.KeyValue) {
	spans := spanRecorder.Ended()

	for _, e := range expect {
		found := 0
		for _, s := range spans {
			if s.Name() != e.name || s.Status().Code != e.status {
				continue
			}
			found++
			want := slices.SortedFunc(slices.Values(attrs), cmpAttrKeyVal)
			got := slices.SortedFunc(slices.Values(s.Attributes()), cmpAttrKeyVal)
			require.Equal(t, want, got)
			assert.Equalf(t, e.status, s.Status().Code, "unexpected status for trace naned: %v got: %v want: %v", e.name, s.Status().Code, e.status)
		}
		if found != e.count {
			t.Errorf("found unexpected number of spans named: %v with status %v, got: %v want: %v", e.name, e.status, found, e.count)
		}
	}
}

func testBedrockCfg(url string) *config.AWSBedrock {
	return &config.AWSBedrock{
		Region:           "us-west-2",
		AccessKey:        "test-access-key",
		AccessKeySecret:  "test-secret-key",
		Model:            "beddel",  // This model should override the request's given one.
		SmallFastModel:   "modrock", // Unused but needed for validation.
		EndpointOverride: url,
	}
}
