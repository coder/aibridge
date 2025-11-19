package aibridge_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/coder/aibridge"
	"github.com/coder/aibridge/aibtrace"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
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

	cases := []struct {
		name              string
		streaming         bool
		bedrock           bool
		expectTraceCounts []expectTrace
	}{
		{
			name:              "trace_anthr_non_streaming",
			expectTraceCounts: expectNonStreaming,
		},
		{
			name:              "trace_bedrock_non_streaming",
			bedrock:           true,
			expectTraceCounts: expectNonStreaming,
		},
		{
			name:      "trace_anthr_streaming",
			streaming: true,
			expectTraceCounts: []expectTrace{
				{"Intercept", 1, codes.Unset},
				{"Intercept.CreateInterceptor", 1, codes.Unset},
				{"Intercept.RecordInterception", 1, codes.Unset},
				{"Intercept.ProcessRequest", 1, codes.Unset},
				{"Intercept.RecordInterceptionEnded", 1, codes.Unset},
				{"Intercept.RecordPromptUsage", 1, codes.Unset},
				{"Intercept.RecordTokenUsage", 2, codes.Unset},
				{"Intercept.RecordToolUsage", 1, codes.Unset},
				{"Intercept.ProcessRequest.Upstream", 9, codes.Unset},
			},
		},
		{
			name:      "trace_bedrock_streaming",
			streaming: true,
			bedrock:   true,
			expectTraceCounts: []expectTrace{
				{"Intercept", 1, codes.Unset},
				{"Intercept.CreateInterceptor", 1, codes.Unset},
				{"Intercept.RecordInterception", 1, codes.Unset},
				{"Intercept.ProcessRequest", 1, codes.Unset},
				{"Intercept.RecordInterceptionEnded", 1, codes.Unset},
				{"Intercept.RecordPromptUsage", 1, codes.Unset},
				{"Intercept.ProcessRequest.Upstream", 1, codes.Unset},
			},
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

			var bedrockCfg *aibridge.AWSBedrockConfig
			if tc.bedrock {
				bedrockCfg = &aibridge.AWSBedrockConfig{
					Region:           "us-west-2",
					AccessKey:        "test-access-key",
					AccessKeySecret:  "test-secret-key",
					Model:            "beddel",  // This model should override the request's given one.
					SmallFastModel:   "modrock", // Unused but needed for validation.
					EndpointOverride: mockAPI.URL,
				}
			}
			provider := aibridge.NewAnthropicProvider(anthropicCfg(mockAPI.URL, apiKey), bedrockCfg)
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
			attrs := []attribute.KeyValue{
				attribute.String(aibtrace.InterceptionID, intcID),
				attribute.String(aibtrace.Provider, aibridge.ProviderAnthropic),
				attribute.String(aibtrace.Model, model),
				attribute.String(aibtrace.UserID, userID),
				attribute.Bool(aibtrace.Streaming, tc.streaming),
				attribute.Bool(aibtrace.IsBedrock, tc.bedrock),
			}

			verifyCommonTraceAttrs(t, sr, tc.expectTraceCounts, attrs)
		})
	}
}

func TestTraceAnthropicErr(t *testing.T) {
	cases := []struct {
		name      string
		streaming bool
		expect    []expectTrace
	}{
		{
			name: "trace_anthr_non_streaming_err",
			expect: []expectTrace{
				{"Intercept", 1, codes.Error},
				{"Intercept.CreateInterceptor", 1, codes.Unset},
				{"Intercept.RecordInterception", 1, codes.Unset},
				{"Intercept.ProcessRequest", 1, codes.Error},
				{"Intercept.RecordInterceptionEnded", 1, codes.Unset},
				{"Intercept.ProcessRequest.Upstream", 1, codes.Error},
			},
		},
		{
			name:      "trace_anthr_streaming_err",
			streaming: true,
			expect: []expectTrace{
				{"Intercept", 1, codes.Error},
				{"Intercept.CreateInterceptor", 1, codes.Unset},
				{"Intercept.RecordInterception", 1, codes.Unset},
				{"Intercept.ProcessRequest", 1, codes.Error},
				{"Intercept.RecordPromptUsage", 1, codes.Unset},
				{"Intercept.RecordTokenUsage", 1, codes.Unset},
				{"Intercept.RecordInterceptionEnded", 1, codes.Unset},
				{"Intercept.ProcessRequest.Upstream", 3, codes.Unset},
			},
		},
	}

	for _, tc := range cases {
		t.Run(t.Name(), func(t *testing.T) {
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

			provider := aibridge.NewAnthropicProvider(anthropicCfg(mockAPI.URL, apiKey), nil)
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

			attrs := []attribute.KeyValue{
				attribute.String(aibtrace.InterceptionID, intcID),
				attribute.String(aibtrace.Provider, aibridge.ProviderAnthropic),
				attribute.String(aibtrace.Model, gjson.Get(string(reqBody), "model").Str),
				attribute.String(aibtrace.UserID, userID),
				attribute.Bool(aibtrace.Streaming, tc.streaming),
				attribute.Bool(aibtrace.IsBedrock, false),
			}

			verifyCommonTraceAttrs(t, sr, tc.expect, attrs)
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
				{"Intercept.ProcessRequest.Upstream", 242, codes.Unset},
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
			provider := aibridge.NewOpenAIProvider(openaiCfg(mockAPI.URL, apiKey))
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

			attrs := []attribute.KeyValue{
				attribute.String(aibtrace.InterceptionID, intcID),
				attribute.String(aibtrace.Provider, aibridge.ProviderOpenAI),
				attribute.String(aibtrace.Model, gjson.Get(string(reqBody), "model").Str),
				attribute.String(aibtrace.UserID, userID),
				attribute.Bool(aibtrace.Streaming, tc.streaming),
			}

			verifyCommonTraceAttrs(t, sr, tc.expect, attrs)
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
				{"Intercept.ProcessRequest.Upstream", 5, codes.Unset},
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
			provider := aibridge.NewOpenAIProvider(openaiCfg(mockAPI.URL, apiKey))
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

			attrs := []attribute.KeyValue{
				attribute.String(aibtrace.InterceptionID, intcID),
				attribute.String(aibtrace.Provider, aibridge.ProviderOpenAI),
				attribute.String(aibtrace.Model, gjson.Get(string(reqBody), "model").Str),
				attribute.String(aibtrace.UserID, userID),
				attribute.Bool(aibtrace.Streaming, tc.streaming),
			}

			verifyCommonTraceAttrs(t, sr, tc.expect, attrs)
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
	require.Equal(t, len(spans), 1)
	assert.Equal(t, spans[0].Name(), "Passthrough")
	attrs := []attribute.KeyValue{
		attribute.String(aibtrace.PassthroughURL, "/v1/models"),
		attribute.String(aibtrace.PassthroughMethod, "GET"),
	}
	if attrDiff := cmp.Diff(spans[0].Attributes(), attrs, cmpopts.EquateComparable(attribute.KeyValue{}), cmpopts.SortSlices(cmpAttrKeyVal)); attrDiff != "" {
		t.Errorf("unexpectet attrs diff: %s", attrDiff)
	}
}

func cmpAttrKeyVal(a attribute.KeyValue, b attribute.KeyValue) bool {
	return a.Key < b.Key
}

func verifyCommonTraceAttrs(t *testing.T, spanRecorder *tracetest.SpanRecorder, expect []expectTrace, attrs []attribute.KeyValue) {
	spans := spanRecorder.Ended()

	totalCount := 0
	for _, e := range expect {
		totalCount += e.count
	}
	assert.Equal(t, totalCount, len(spans))

	for _, e := range expect {
		found := 0
		for _, s := range spans {
			if s.Name() != e.name || s.Status().Code != e.status {
				continue
			}
			found++
			if attrDiff := cmp.Diff(s.Attributes(), attrs, cmpopts.EquateComparable(attribute.KeyValue{}), cmpopts.SortSlices(cmpAttrKeyVal)); attrDiff != "" {
				t.Errorf("unexpectet attrs for span named: %v, diff: %s", e.name, attrDiff)
			}
			assert.Equalf(t, e.status, s.Status().Code, "unexpected status for trace naned: %v got: %v want: %v", e.name, s.Status().Code, e.status)
		}
		if found != e.count {
			t.Errorf("found unexpected number of spans named: %v with status %v, got: %v want: %v", e.name, e.status, found, e.count)
		}
	}
}
