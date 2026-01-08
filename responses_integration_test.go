package aibridge_test

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/coder/aibridge/provider"
	"github.com/stretchr/testify/require"
	"golang.org/x/tools/txtar"
)

var (
	//go:embed fixtures/openai/responses/blocking/simple.txtar
	fixtResponsesBlockingSimple []byte

	//go:embed fixtures/openai/responses/blocking/builtin_tool.txtar
	fixtResponsesBlockingBuiltinTool []byte

	//go:embed fixtures/openai/responses/blocking/conversation.txtar
	fixtResponsesBlockingConversation []byte

	//go:embed fixtures/openai/responses/blocking/prev_response_id.txtar
	fixtResponsesBlockingPrevResponseID []byte

	//go:embed fixtures/openai/responses/blocking/wrong_response_format.txtar
	fixtResponsesBlockingWrongResponseFormat []byte

	//go:embed fixtures/openai/responses/streaming/simple.txtar
	fixtResponsesStreamingSimple []byte

	//go:embed fixtures/openai/responses/streaming/builtin_tool.txtar
	fixtResponsesStreamingBuiltinTool []byte

	//go:embed fixtures/openai/responses/streaming/conversation.txtar
	fixtResponsesStreamingConversation []byte

	//go:embed fixtures/openai/responses/streaming/prev_response_id.txtar
	fixtResponsesStreamingPrevResponseID []byte

	//go:embed fixtures/openai/responses/streaming/stream_error.txtar
	fixtResponsesStreamingStreamError []byte

	//go:embed fixtures/openai/responses/streaming/stream_failure.txtar
	fixtResponsesStreamingStreamFailure []byte

	//go:embed fixtures/openai/responses/streaming/wrong_response_format.txtar
	fixtResponsesStreamingWrongResponseFormat []byte

	//go:embed fixtures/openai/responses/upstream_http_err.txtar
	fixtResponsesUpstreamHttpError []byte
)

func TestResponsesOutputMatchesUpstream(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name            string
		fixture         []byte
		streaming       bool
		expectErrStatus int
	}{
		{
			name:    "blocking_simple",
			fixture: fixtResponsesBlockingSimple,
		},
		{
			name:    "blocking_builtin_tool",
			fixture: fixtResponsesBlockingBuiltinTool,
		},
		{
			name:    "blocking_conversation",
			fixture: fixtResponsesBlockingConversation,
		},
		{
			name:    "blocking_prev_response_id",
			fixture: fixtResponsesBlockingPrevResponseID,
		},
		{
			name:            "blocking_wrong_format",
			fixture:         fixtResponsesBlockingWrongResponseFormat,
			expectErrStatus: 500,
		},
		{
			name:      "streaming_simple",
			fixture:   fixtResponsesStreamingSimple,
			streaming: true,
		},
		{
			name:      "streaming_builtin_tool",
			fixture:   fixtResponsesStreamingBuiltinTool,
			streaming: true,
		},
		{
			name:      "streaming_conversation",
			fixture:   fixtResponsesStreamingConversation,
			streaming: true,
		},
		{
			name:      "streaming_prev_response_id",
			fixture:   fixtResponsesStreamingPrevResponseID,
			streaming: true,
		},
		{
			name:      "stream_error",
			fixture:   fixtResponsesStreamingStreamError,
			streaming: true,
		},
		{
			name:      "stream_failure",
			fixture:   fixtResponsesStreamingStreamFailure,
			streaming: true,
		},

		{ // sdk seems to ignore not well formatted chunks when streaming
			name:      "streaming_wrong_format",
			fixture:   fixtResponsesStreamingWrongResponseFormat,
			streaming: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			files := filesMap(txtar.Parse(tc.fixture))
			require.Contains(t, files, fixtureRequest)
			fixtResp := fixtureNonStreamingResponse
			if tc.streaming {
				fixtResp = fixtureStreamingResponse
			}
			require.Contains(t, files, fixtResp)

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			mockAPI := newMockServer(ctx, t, files, nil)
			t.Cleanup(mockAPI.Close)

			provider := provider.NewOpenAI(openaiCfg(mockAPI.URL, apiKey))
			srv, _ := newTestSrv(t, ctx, provider, nil, testTracer)
			defer srv.Close()

			req := createOpenAIResponsesReq(t, srv.URL, files[fixtureRequest])
			client := &http.Client{}

			resp, err := client.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()
			if tc.expectErrStatus == 0 {
				require.Equal(t, http.StatusOK, resp.StatusCode)
				got, err := io.ReadAll(resp.Body)

				require.NoError(t, err)
				require.Equal(t, string(files[fixtResp]), string(got))
			} else {
				require.Equal(t, tc.expectErrStatus, resp.StatusCode)
			}
		})
	}
}

func TestResponsesBackgroundModeForbidden(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		streaming bool
	}{
		{
			name:      "blocking",
			streaming: false,
		},
		{
			name:      "streaming",
			streaming: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			// request with Background mode should be rejected before it reaches upstream
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				t.Errorf("unexpected request to upstream: %s %s", r.Method, r.URL.Path)
				w.WriteHeader(http.StatusInternalServerError)
			}))
			t.Cleanup(upstream.Close)

			prov := provider.NewOpenAI(openaiCfg(upstream.URL, apiKey))
			srv, _ := newTestSrv(t, ctx, prov, nil, testTracer)
			defer srv.Close()

			// Create a request with background mode enabled
			reqBody := map[string]any{
				"input":      "tell me a joke",
				"model":      "gpt-4o-mini",
				"stream":     tc.streaming,
				"background": true,
			}
			reqBytes, err := json.Marshal(reqBody)
			require.NoError(t, err)

			req := createOpenAIResponsesReq(t, srv.URL, reqBytes)
			client := &http.Client{}

			resp, err := client.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()

			require.Equal(t, http.StatusNotImplemented, resp.StatusCode)

			body, err := io.ReadAll(resp.Body)
			require.NoError(t, err)
			require.Contains(t, string(body), "background requests are currently not supported by aibridge")
		})
	}
}

func TestResponsesParallelToolsOverwritten(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                         string
		request                      string
		streaming                    bool
		expectParallelToolCalls      bool
		expectParallelToolCallsValue bool
	}{
		{
			name:                         "blocking_with_tools",
			request:                      `{"input": "hello", "model": "gpt-4o-mini", "stream": false, "parallel_tool_calls": true, "tools": [{"type": "function", "name": "test", "parameters": {}}]}`,
			streaming:                    false,
			expectParallelToolCalls:      true,
			expectParallelToolCallsValue: false,
		},
		{
			name:                         "streaming_with_tools",
			request:                      `{"input": "hello", "model": "gpt-4o-mini", "stream": true, "parallel_tool_calls": true, "tools": [{"type": "function", "name": "test", "parameters": {}}]}`,
			streaming:                    true,
			expectParallelToolCalls:      true,
			expectParallelToolCallsValue: false,
		},
		{
			name:                         "blocking_with_tools_no_parallel_param",
			request:                      `{"input": "hello", "model": "gpt-4o-mini", "stream": false, "tools": [{"type": "function", "name": "test", "parameters": {}}]}`,
			streaming:                    false,
			expectParallelToolCalls:      true,
			expectParallelToolCallsValue: false,
		},
		{
			name:                         "streaming_with_tools_no_parallel_param",
			request:                      `{"input": "hello", "model": "gpt-4o-mini", "stream": true, "tools": [{"type": "function", "name": "test", "parameters": {}}]}`,
			streaming:                    true,
			expectParallelToolCalls:      true,
			expectParallelToolCallsValue: false,
		},
		{
			name:      "blocking_without_tools",
			request:   `{"input": "hello", "model": "gpt-4o-mini", "stream": false}`,
			streaming: false,
		},
		{
			name:      "streaming_without_tools",
			request:   `{"input": "hello", "model": "gpt-4o-mini", "stream": true}`,
			streaming: true,
		},
		{
			name:                         "blocking_without_tools_parallel_true",
			request:                      `{"input": "hello", "model": "gpt-4o-mini", "stream": false, "parallel_tool_calls": true}`,
			streaming:                    false,
			expectParallelToolCalls:      true,
			expectParallelToolCallsValue: true,
		},
		{
			name:                         "streaming_without_tools_parallel_true",
			request:                      `{"input": "hello", "model": "gpt-4o-mini", "stream": true, "parallel_tool_calls": true}`,
			streaming:                    true,
			expectParallelToolCalls:      true,
			expectParallelToolCallsValue: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			var receivedRequest map[string]any
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, err := io.ReadAll(r.Body)
				require.NoError(t, err)
				defer r.Body.Close()

				require.NoError(t, json.Unmarshal(raw, &receivedRequest))

				w.WriteHeader(http.StatusOK)
			}))
			t.Cleanup(upstream.Close)

			prov := provider.NewOpenAI(openaiCfg(upstream.URL, apiKey))
			srv, _ := newTestSrv(t, ctx, prov, nil, testTracer)
			defer srv.Close()

			req := createOpenAIResponsesReq(t, srv.URL, []byte(tc.request))
			client := &http.Client{}

			resp, err := client.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()
			_, _ = io.ReadAll(resp.Body)

			require.NotNil(t, receivedRequest)
			if tc.expectParallelToolCalls {
				parallelToolCalls, ok := receivedRequest["parallel_tool_calls"].(bool)
				require.True(t, ok, "parallel_tool_calls should be present in upstream request")
				require.Equal(t, tc.expectParallelToolCallsValue, parallelToolCalls)
			} else {
				_, ok := receivedRequest["parallel_tool_calls"]
				require.False(t, ok, "parallel_tool_calls should not be present when not set")
			}
		})
	}
}

func createOpenAIResponsesReq(t *testing.T, baseURL string, input []byte) *http.Request {
	t.Helper()

	req, err := http.NewRequestWithContext(t.Context(), "POST", baseURL+"/openai/v1/responses", bytes.NewReader(input))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	return req
}
