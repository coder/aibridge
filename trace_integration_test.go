package aibridge_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"cdr.dev/slog"
	"cdr.dev/slog/sloggers/slogtest"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/testutil"
	"github.com/coder/aibridge/tracing"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
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

	fixture := testutil.MustParseTXTAR(t, antSingleBuiltinTool)
	fixture.RequireFiles(t, testutil.FixtureRequest, testutil.FixtureStreamingResponse, testutil.FixtureNonStreamingResponse)
	llm := testutil.MustLLMFixture(t, fixture)

	for _, tc := range cases {
		t.Run(fmt.Sprintf("%s/streaming=%v", t.Name(), tc.streaming), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			t.Cleanup(cancel)

			sr := tracetest.NewSpanRecorder()
			tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
			tracer := tp.Tracer(t.Name())
			defer func() { _ = tp.Shutdown(t.Context()) }()

			reqBody := llm.MustRequestBody(t, tc.streaming)

			upstream := testutil.NewUpstreamServer(t, ctx, llm)

			var bedrockCfg *aibridge.AWSBedrockConfig
			if tc.bedrock {
				bedrockCfg = testBedrockCfg(upstream.URL)
			}
			provider := aibridge.NewAnthropicProvider(anthropicCfg(upstream.URL, apiKey), bedrockCfg)
			srv, recorder := newTestSrv(t, ctx, provider, nil, tracer)

			req := srv.NewProviderRequest(t, provider.Name(), reqBody)
			resp, err := srv.Client.Do(req)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, resp.StatusCode)
			_, err = io.Copy(io.Discard, resp.Body)
			require.NoError(t, err)
			require.NoError(t, resp.Body.Close())

			interceptions := recorder.RecordedInterceptions()
			require.Len(t, interceptions, 1)
			intcID := interceptions[0].ID

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
				attribute.String(tracing.Provider, aibridge.ProviderAnthropic),
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
			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			t.Cleanup(cancel)

			fixtureBytes := antNonStreamErr
			if tc.streaming {
				fixtureBytes = antMidStreamErr
			}

			fixture := testutil.MustParseTXTAR(t, fixtureBytes)
			llm := testutil.MustLLMFixture(t, fixture)

			sr := tracetest.NewSpanRecorder()
			tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
			tracer := tp.Tracer(t.Name())
			defer func() { _ = tp.Shutdown(t.Context()) }()

			reqBody := llm.MustRequestBody(t, tc.streaming)

			upstream := testutil.NewUpstreamServer(t, ctx, llm)

			var bedrockCfg *aibridge.AWSBedrockConfig
			if tc.bedrock {
				bedrockCfg = testBedrockCfg(upstream.URL)
			}
			provider := aibridge.NewAnthropicProvider(anthropicCfg(upstream.URL, apiKey), bedrockCfg)
			srv, recorder := newTestSrv(t, ctx, provider, nil, tracer)

			req := srv.NewProviderRequest(t, provider.Name(), reqBody)
			resp, err := srv.Client.Do(req)
			require.NoError(t, err)
			if tc.streaming {
				require.Equal(t, http.StatusOK, resp.StatusCode)
			} else {
				require.Equal(t, http.StatusInternalServerError, resp.StatusCode)
			}
			_, err = io.Copy(io.Discard, resp.Body)
			require.NoError(t, err)
			require.NoError(t, resp.Body.Close())

			interceptions := recorder.RecordedInterceptions()
			require.Len(t, interceptions, 1)
			intcID := interceptions[0].ID

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
				attribute.String(tracing.Provider, aibridge.ProviderAnthropic),
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

			h := runInjectedToolTest(t, aibridge.ProviderAnthropic, antSingleInjectedTool, tc.streaming, tracer, func(upstreamURL string) []aibridge.Provider {
				var bedrockCfg *aibridge.AWSBedrockConfig
				if tc.bedrock {
					bedrockCfg = testBedrockCfg(upstreamURL)
				}
				return []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(upstreamURL, apiKey), bedrockCfg)}
			})

			defer h.Response.Body.Close()

			interceptions := h.Recorder.RecordedInterceptions()
			require.Len(t, interceptions, 1)
			intcID := interceptions[0].ID

			reqBody := string(h.RequestBody)
			model := gjson.Get(reqBody, "model").Str
			if tc.bedrock {
				model = "beddel"
			}

			for _, proxy := range h.MCPProxiers {
				require.NotEmpty(t, proxy.ListTools())
				tool := proxy.ListTools()[0]

				attrs := []attribute.KeyValue{
					attribute.String(tracing.RequestPath, h.RequestPath),
					attribute.String(tracing.InterceptionID, intcID),
					attribute.String(tracing.Provider, aibridge.ProviderAnthropic),
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
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			t.Cleanup(cancel)

			fixture := testutil.MustParseTXTAR(t, tc.fixture)
			fixture.RequireFiles(t, testutil.FixtureRequest, testutil.FixtureStreamingResponse, testutil.FixtureNonStreamingResponse)
			llm := testutil.MustLLMFixture(t, fixture)

			sr := tracetest.NewSpanRecorder()
			tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
			tracer := tp.Tracer(t.Name())
			defer func() { _ = tp.Shutdown(t.Context()) }()

			reqBody := llm.MustRequestBody(t, tc.streaming)

			upstream := testutil.NewUpstreamServer(t, ctx, llm)
			provider := aibridge.NewOpenAIProvider(openaiCfg(upstream.URL, apiKey))
			srv, recorder := newTestSrv(t, ctx, provider, nil, tracer)

			req := srv.NewProviderRequest(t, provider.Name(), reqBody)
			resp, err := srv.Client.Do(req)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, resp.StatusCode)
			_, err = io.Copy(io.Discard, resp.Body)
			require.NoError(t, err)
			require.NoError(t, resp.Body.Close())

			interceptions := recorder.RecordedInterceptions()
			require.Len(t, interceptions, 1)
			intcID := interceptions[0].ID

			totalCount := 0
			for _, e := range tc.expect {
				totalCount += e.count
			}
			require.Len(t, sr.Ended(), totalCount)

			attrs := []attribute.KeyValue{
				attribute.String(tracing.RequestPath, req.URL.Path),
				attribute.String(tracing.InterceptionID, intcID),
				attribute.String(tracing.Provider, aibridge.ProviderOpenAI),
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
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			t.Cleanup(cancel)

			fixtureBytes := oaiNonStreamErr
			if tc.streaming {
				fixtureBytes = oaiMidStreamErr
			}

			fixture := testutil.MustParseTXTAR(t, fixtureBytes)
			llm := testutil.MustLLMFixture(t, fixture)

			sr := tracetest.NewSpanRecorder()
			tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
			tracer := tp.Tracer(t.Name())
			defer func() { _ = tp.Shutdown(t.Context()) }()

			reqBody := llm.MustRequestBody(t, tc.streaming)

			upstream := testutil.NewUpstreamServer(t, ctx, llm)
			provider := aibridge.NewOpenAIProvider(openaiCfg(upstream.URL, apiKey))
			srv, recorder := newTestSrv(t, ctx, provider, nil, tracer)

			req := srv.NewProviderRequest(t, provider.Name(), reqBody)
			resp, err := srv.Client.Do(req)
			require.NoError(t, err)
			if tc.streaming {
				require.Equal(t, http.StatusOK, resp.StatusCode)
			} else {
				require.Equal(t, http.StatusInternalServerError, resp.StatusCode)
			}
			_, err = io.Copy(io.Discard, resp.Body)
			require.NoError(t, err)
			require.NoError(t, resp.Body.Close())

			interceptions := recorder.RecordedInterceptions()
			require.Len(t, interceptions, 1)
			intcID := interceptions[0].ID

			totalCount := 0
			for _, e := range tc.expect {
				totalCount += e.count
			}
			require.Len(t, sr.Ended(), totalCount)

			attrs := []attribute.KeyValue{
				attribute.String(tracing.RequestPath, req.URL.Path),
				attribute.String(tracing.InterceptionID, intcID),
				attribute.String(tracing.Provider, aibridge.ProviderOpenAI),
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

			h := runInjectedToolTest(t, aibridge.ProviderOpenAI, oaiSingleInjectedTool, streaming, tracer, func(upstreamURL string) []aibridge.Provider {
				return []aibridge.Provider{aibridge.NewOpenAIProvider(openaiCfg(upstreamURL, apiKey))}
			})

			defer h.Response.Body.Close()

			interceptions := h.Recorder.RecordedInterceptions()
			require.Len(t, interceptions, 1)
			intcID := interceptions[0].ID

			for _, proxy := range h.MCPProxiers {
				require.NotEmpty(t, proxy.ListTools())
				tool := proxy.ListTools()[0]

				attrs := []attribute.KeyValue{
					attribute.String(tracing.RequestPath, h.RequestPath),
					attribute.String(tracing.InterceptionID, intcID),
					attribute.String(tracing.Provider, aibridge.ProviderOpenAI),
					attribute.String(tracing.Model, gjson.Get(string(h.RequestBody), "model").Str),
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

	fixture := testutil.MustParseTXTAR(t, oaiFallthrough)
	respBody := fixture.MustFile(t, testutil.FixtureResponse)

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(respBody)
	}))
	t.Cleanup(upstream.Close)

	sr := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
	tracer := tp.Tracer(t.Name())
	defer func() { _ = tp.Shutdown(t.Context()) }()

	provider := aibridge.NewOpenAIProvider(openaiCfg(upstream.URL, apiKey))
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
	mcpSrv := testutil.NewMCPServer(t, testutil.DefaultCoderToolNames())

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

func testBedrockCfg(url string) *aibridge.AWSBedrockConfig {
	return &aibridge.AWSBedrockConfig{
		Region:           "us-west-2",
		AccessKey:        "test-access-key",
		AccessKeySecret:  "test-secret-key",
		Model:            "beddel",  // This model should override the request's given one.
		SmallFastModel:   "modrock", // Unused but needed for validation.
		EndpointOverride: url,
	}
}
