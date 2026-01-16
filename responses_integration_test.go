package aibridge_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/coder/aibridge/config"
	aibcontext "github.com/coder/aibridge/context"
	"github.com/coder/aibridge/fixtures"
	"github.com/coder/aibridge/provider"
	"github.com/coder/aibridge/recorder"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/require"
	"golang.org/x/tools/txtar"
)

type keyVal struct {
	key string
	val any
}

func TestResponsesOutputMatchesUpstream(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                 string
		fixture              []byte
		streaming            bool
		expectModel          string
		expectPromptRecorded string
		expectToolRecorded   *recorder.ToolUsageRecord
	}{
		{
			name:                 "blocking_simple",
			fixture:              fixtures.OaiResponsesBlockingSimple,
			expectModel:          "gpt-4o-mini",
			expectPromptRecorded: "tell me a joke",
		},
		{
			name:                 "blocking_builtin_tool",
			fixture:              fixtures.OaiResponsesBlockingBuiltinTool,
			expectModel:          "gpt-4.1",
			expectPromptRecorded: "Is 3 + 5 a prime number? Use the add function to calculate the sum.",
			expectToolRecorded: &recorder.ToolUsageRecord{
				MsgID:    "resp_0da6045a8b68fa5200695fa23dcc2c81a19c849f627abf8a31",
				Tool:     "add",
				Args:     map[string]any{"a": float64(3), "b": float64(5)},
				Injected: false,
			},
		},
		{
			name:                 "blocking_custom_tool",
			fixture:              fixtures.OaiResponsesBlockingCustomTool,
			expectModel:          "gpt-5",
			expectPromptRecorded: "Use the code_exec tool to print hello world to the console.",
			expectToolRecorded: &recorder.ToolUsageRecord{
				MsgID:    "resp_09c614364030cdf000696942589da081a0af07f5859acb7308",
				Tool:     "code_exec",
				Args:     "print(\"hello world\")",
				Injected: false,
			},
		},
		{
			name:                 "blocking_conversation",
			fixture:              fixtures.OaiResponsesBlockingConversation,
			expectModel:          "gpt-4o-mini",
			expectPromptRecorded: "explain why this is funny.",
		},
		{
			name:                 "blocking_prev_response_id",
			fixture:              fixtures.OaiResponsesBlockingPrevResponseID,
			expectModel:          "gpt-4o-mini",
			expectPromptRecorded: "explain why this is funny.",
		},
		{
			name:                 "streaming_simple",
			fixture:              fixtures.OaiResponsesStreamingSimple,
			streaming:            true,
			expectModel:          "gpt-4o-mini",
			expectPromptRecorded: "tell me a joke",
		},
		{
			name:                 "streaming_codex",
			fixture:              fixtures.OaiResponsesStreamingCodex,
			streaming:            true,
			expectModel:          "gpt-5-codex",
			expectPromptRecorded: "hello",
		},
		{
			name:                 "streaming_builtin_tool",
			fixture:              fixtures.OaiResponsesStreamingBuiltinTool,
			streaming:            true,
			expectModel:          "gpt-4.1",
			expectPromptRecorded: "Is 3 + 5 a prime number? Use the add function to calculate the sum.",
		},
		{
			name:                 "streaming_conversation",
			fixture:              fixtures.OaiResponsesStreamingConversation,
			streaming:            true,
			expectModel:          "gpt-4o-mini",
			expectPromptRecorded: "explain why this is funny.",
		},
		{
			name:                 "streaming_prev_response_id",
			fixture:              fixtures.OaiResponsesStreamingPrevResponseID,
			streaming:            true,
			expectModel:          "gpt-4o-mini",
			expectPromptRecorded: "explain why this is funny.",
		},
		{
			name:                 "stream_error",
			fixture:              fixtures.OaiResponsesStreamingStreamError,
			streaming:            true,
			expectModel:          "gpt-6.7",
			expectPromptRecorded: "hello_stream_error",
		},
		{
			name:                 "stream_failure",
			fixture:              fixtures.OaiResponsesStreamingStreamFailure,
			streaming:            true,
			expectModel:          "gpt-6.7",
			expectPromptRecorded: "hello_stream_failure",
		},

		// Original status code and body is kept even with wrong json format
		{
			name:        "blocking_wrong_format",
			fixture:     fixtures.OaiResponsesBlockingWrongResponseFormat,
			expectModel: "gpt-6.7",
		},
		{
			name:                 "streaming_wrong_format",
			fixture:              fixtures.OaiResponsesStreamingWrongResponseFormat,
			streaming:            true,
			expectModel:          "gpt-6.7",
			expectPromptRecorded: "hello_wrong_format",
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
			ctx = aibcontext.AsActor(ctx, userID, nil)

			mockAPI := newMockServer(ctx, t, files, nil)
			t.Cleanup(mockAPI.Close)

			provider := provider.NewOpenAI(openaiCfg(mockAPI.URL, apiKey))
			srv, mockRecorder := newTestSrv(t, ctx, provider, nil, testTracer)
			defer srv.Close()

			req := createOpenAIResponsesReq(t, srv.URL, files[fixtureRequest])
			client := &http.Client{}

			resp, err := client.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()
			require.Equal(t, http.StatusOK, resp.StatusCode)
			got, err := io.ReadAll(resp.Body)
			srv.Close()

			require.NoError(t, err)
			require.Equal(t, string(files[fixtResp]), string(got))

			interceptions := mockRecorder.RecordedInterceptions()
			require.Len(t, interceptions, 1)
			intc := interceptions[0]
			require.Equal(t, intc.InitiatorID, userID)
			require.Equal(t, intc.Provider, config.ProviderOpenAI)
			require.Equal(t, intc.Model, tc.expectModel)

			recordedPrompts := mockRecorder.RecordedPromptUsages()
			if tc.expectPromptRecorded != "" {
				require.Len(t, recordedPrompts, 1)
				promptEq := func(pur *recorder.PromptUsageRecord) bool { return pur.Prompt == tc.expectPromptRecorded }
				require.Truef(t, slices.ContainsFunc(recordedPrompts, promptEq), "promnt not found, got: %v, want: %v", recordedPrompts, tc.expectPromptRecorded)
			} else {
				require.Empty(t, recordedPrompts)
			}

			recordedTools := mockRecorder.RecordedToolUsages()
			if tc.expectToolRecorded != nil {
				require.Len(t, recordedTools, 1)
				recordedTools[0].InterceptionID = tc.expectToolRecorded.InterceptionID // ignore interception id (interception id is not constant and response doesn't contain it)
				recordedTools[0].CreatedAt = tc.expectToolRecorded.CreatedAt           // ignore time
				require.Equal(t, tc.expectToolRecorded, recordedTools[0])
			} else {
				require.Empty(t, recordedTools)
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
			reqBytes := responsesRequestBytes(t, tc.streaming, keyVal{"background", true})
			req := createOpenAIResponsesReq(t, srv.URL, reqBytes)
			client := &http.Client{}

			resp, err := client.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()

			require.Equal(t, "application/json", resp.Header.Get("Content-Type"))
			require.Equal(t, http.StatusNotImplemented, resp.StatusCode)

			body, err := io.ReadAll(resp.Body)
			require.NoError(t, err)
			requireResponsesError(t, http.StatusNotImplemented, "background requests are currently not supported by AI Bridge", body)
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

			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, err := io.ReadAll(r.Body)
				require.NoError(t, err)
				defer r.Body.Close()

				var receivedRequest map[string]any
				require.NoError(t, json.Unmarshal(raw, &receivedRequest))
				if tc.expectParallelToolCalls {
					parallelToolCalls, ok := receivedRequest["parallel_tool_calls"].(bool)
					require.True(t, ok, "parallel_tool_calls should be present in upstream request")
					require.Equal(t, tc.expectParallelToolCallsValue, parallelToolCalls)
				} else {
					_, ok := receivedRequest["parallel_tool_calls"]
					require.False(t, ok, "parallel_tool_calls should not be present when not set")
				}

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
			_, err = io.ReadAll(resp.Body)
			require.NoError(t, err)
		})
	}
}

// TODO set MaxRetries to speed up this test
// option.WithMaxRetries(0), in base responses interceptor
// https://github.com/coder/aibridge/issues/115
func TestClientAndConnectionError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		addr        string
		streaming   bool
		errContains string
	}{
		{
			name:        "blocking_connection_refused",
			addr:        startRejectingListener(t),
			streaming:   false,
			errContains: "connection reset by peer",
		},
		{
			name:        "streaming_connection_refused",
			addr:        startRejectingListener(t),
			streaming:   true,
			errContains: "connection reset by peer",
		},
		{
			name:        "blocking_bad_url",
			addr:        "not_url",
			streaming:   false,
			errContains: "unsupported protocol scheme",
		},
		{
			name:        "streaming_bad_url",
			addr:        "not_url",
			streaming:   true,
			errContains: "unsupported protocol scheme",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			prov := provider.NewOpenAI(openaiCfg(tc.addr, apiKey))
			srv, mockRecorder := newTestSrv(t, ctx, prov, nil, testTracer)
			defer srv.Close()

			reqBytes := responsesRequestBytes(t, tc.streaming)
			req := createOpenAIResponsesReq(t, srv.URL, reqBytes)
			client := &http.Client{}

			resp, err := client.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()

			require.Equal(t, "application/json", resp.Header.Get("Content-Type"))
			require.Equal(t, http.StatusInternalServerError, resp.StatusCode)

			body, err := io.ReadAll(resp.Body)
			require.NoError(t, err)
			requireResponsesError(t, http.StatusInternalServerError, tc.errContains, body)
			require.Empty(t, mockRecorder.RecordedPromptUsages())
		})
	}
}

// TODO set MaxRetries to speed up this test
// option.WithMaxRetries(0), in base responses interceptor
// https://github.com/coder/aibridge/issues/115
func TestUpstreamError(t *testing.T) {
	t.Parallel()

	responsesError := `{"error":{"message":"Something went wrong","type":"invalid_request_error","param":null,"code":"invalid_request"}}`
	nonResponsesError := `plain text error`

	tests := []struct {
		name        string
		streaming   bool
		statusCode  int
		contentType string
		body        string
	}{
		{
			name:        "blocking_responses_error",
			streaming:   false,
			statusCode:  http.StatusBadRequest,
			contentType: "application/json",
			body:        responsesError,
		},
		{
			name:        "streaming_responses_error",
			streaming:   true,
			statusCode:  http.StatusBadRequest,
			contentType: "application/json",
			body:        responsesError,
		},
		{
			name:        "blocking_non_responses_error",
			streaming:   false,
			statusCode:  http.StatusBadGateway,
			contentType: "text/plain",
			body:        nonResponsesError,
		},
		{
			name:        "streaming_non_responses_error",
			streaming:   true,
			statusCode:  http.StatusBadGateway,
			contentType: "text/plain",
			body:        nonResponsesError,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", tc.contentType)
				w.WriteHeader(tc.statusCode)
				_, err := w.Write([]byte(tc.body))
				require.NoError(t, err)
			}))
			t.Cleanup(upstream.Close)

			prov := provider.NewOpenAI(openaiCfg(upstream.URL, apiKey))
			srv, _ := newTestSrv(t, ctx, prov, nil, testTracer)
			defer srv.Close()

			reqBytes := responsesRequestBytes(t, tc.streaming)
			req := createOpenAIResponsesReq(t, srv.URL, reqBytes)
			client := &http.Client{}

			resp, err := client.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()

			require.Equal(t, tc.statusCode, resp.StatusCode)
			require.Equal(t, tc.contentType, resp.Header.Get("Content-Type"))

			body, err := io.ReadAll(resp.Body)
			require.NoError(t, err)
			require.Equal(t, tc.body, string(body))
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

func requireResponsesError(t *testing.T, code int, message string, body []byte) {
	var respErr responses.Error
	err := json.Unmarshal(body, &respErr)
	require.NoError(t, err)

	require.Equal(t, strconv.Itoa(code), respErr.Code)
	require.Contains(t, respErr.Message, message)
}

func responsesRequestBytes(t *testing.T, streaming bool, additionalFields ...keyVal) []byte {
	reqBody := map[string]any{
		"input":  "tell me a joke",
		"model":  "gpt-4o-mini",
		"stream": streaming,
	}

	for _, kv := range additionalFields {
		reqBody[kv.key] = kv.val
	}

	reqBytes, err := json.Marshal(reqBody)
	require.NoError(t, err)
	return reqBytes
}

func startRejectingListener(t *testing.T) (addr string) {
	t.Helper()

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })

	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				// When ln.Close() is called, Accept returns an error -> exit.
				return
			}

			if tc, ok := c.(*net.TCPConn); ok {
				_ = tc.SetLinger(0)
			}
			_ = c.Close()
		}
	}()

	return "http://" + ln.Addr().String()
}
