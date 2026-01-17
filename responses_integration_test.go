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

	"cdr.dev/slog/v3"
	"cdr.dev/slog/v3/sloggers/slogtest"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/config"
	aibcontext "github.com/coder/aibridge/context"
	"github.com/coder/aibridge/fixtures"
	"github.com/coder/aibridge/internal/testutil"
	"github.com/coder/aibridge/mcp"
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
		expectTokenUsage     *recorder.TokenUsageRecord
	}{
		{
			name:                 "blocking_simple",
			fixture:              fixtures.OaiResponsesBlockingSimple,
			expectModel:          "gpt-4o-mini",
			expectPromptRecorded: "tell me a joke",
			expectTokenUsage: &recorder.TokenUsageRecord{
				MsgID:  "resp_0388c79043df3e3400695f9f83cd6481959062cec6830d8d51",
				Input:  11,
				Output: 18,
				ExtraTokenTypes: map[string]int64{
					"input_cached":     0,
					"output_reasoning": 0,
					"total_tokens":     29,
				},
			},
		},
		{
			name:                 "blocking_builtin_tool",
			fixture:              fixtures.OaiResponsesBlockingSingleBuiltinTool,
			expectModel:          "gpt-4.1",
			expectPromptRecorded: "Is 3 + 5 a prime number? Use the add function to calculate the sum.",
			expectToolRecorded: &recorder.ToolUsageRecord{
				MsgID:    "resp_0da6045a8b68fa5200695fa23dcc2c81a19c849f627abf8a31",
				Tool:     "add",
				Args:     map[string]any{"a": float64(3), "b": float64(5)},
				Injected: false,
			},
			expectTokenUsage: &recorder.TokenUsageRecord{
				MsgID:  "resp_0da6045a8b68fa5200695fa23dcc2c81a19c849f627abf8a31",
				Input:  58,
				Output: 18,
				ExtraTokenTypes: map[string]int64{
					"input_cached":     0,
					"output_reasoning": 0,
					"total_tokens":     76,
				},
			},
		},
		{
			name:                 "blocking_cached_input_tokens",
			fixture:              fixtures.OaiResponsesBlockingCachedInputTokens,
			expectModel:          "gpt-4.1",
			expectPromptRecorded: "This was a large input...",
			expectTokenUsage: &recorder.TokenUsageRecord{
				MsgID:  "resp_0cd5d6b8310055d600696a1776b42c81a199fbb02248a8bfa0",
				Input:  129, // 12033 input - 11904 cached
				Output: 44,
				ExtraTokenTypes: map[string]int64{
					"input_cached":     11904,
					"output_reasoning": 0,
					"total_tokens":     12077,
				},
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
			expectTokenUsage: &recorder.TokenUsageRecord{
				MsgID:  "resp_09c614364030cdf000696942589da081a0af07f5859acb7308",
				Input:  64,
				Output: 148,
				ExtraTokenTypes: map[string]int64{
					"input_cached":     0,
					"output_reasoning": 128,
					"total_tokens":     212,
				},
			},
		},
		{
			name:                 "blocking_conversation",
			fixture:              fixtures.OaiResponsesBlockingConversation,
			expectModel:          "gpt-4o-mini",
			expectPromptRecorded: "explain why this is funny.",
			expectTokenUsage: &recorder.TokenUsageRecord{
				MsgID:  "resp_0c9f1f0524a858fa00695fa15fc5a081958f4304aafd3bdec2",
				Input:  48,
				Output: 116,
				ExtraTokenTypes: map[string]int64{
					"input_cached":     0,
					"output_reasoning": 0,
					"total_tokens":     164,
				},
			},
		},
		{
			name:                 "blocking_prev_response_id",
			fixture:              fixtures.OaiResponsesBlockingPrevResponseID,
			expectModel:          "gpt-4o-mini",
			expectPromptRecorded: "explain why this is funny.",
			expectTokenUsage: &recorder.TokenUsageRecord{
				MsgID:  "resp_0388c79043df3e3400695f9f86cfa08195af1f015c60117a83",
				Input:  43,
				Output: 129,
				ExtraTokenTypes: map[string]int64{
					"input_cached":     0,
					"output_reasoning": 0,
					"total_tokens":     172,
				},
			},
		},
		{
			name:                 "streaming_simple",
			fixture:              fixtures.OaiResponsesStreamingSimple,
			streaming:            true,
			expectModel:          "gpt-4o-mini",
			expectPromptRecorded: "tell me a joke",
			expectTokenUsage: &recorder.TokenUsageRecord{
				MsgID:  "resp_0f9c4b2f224d858000695fa062bf048197a680f357bbb09000",
				Input:  11,
				Output: 18,
				ExtraTokenTypes: map[string]int64{
					"input_cached":     0,
					"output_reasoning": 0,
					"total_tokens":     29,
				},
			},
		},
		{
			name:                 "streaming_codex",
			fixture:              fixtures.OaiResponsesStreamingCodex,
			streaming:            true,
			expectModel:          "gpt-5-codex",
			expectPromptRecorded: "hello",
			expectTokenUsage: &recorder.TokenUsageRecord{
				MsgID:  "resp_0e172b76542a9100016964f7e63d888191a2a28cb2ba0ab6d3",
				Input:  4006,
				Output: 13,
				ExtraTokenTypes: map[string]int64{
					"input_cached":     0,
					"output_reasoning": 0,
					"total_tokens":     4019,
				},
			},
		},
		{
			name:                 "streaming_builtin_tool",
			fixture:              fixtures.OaiResponsesStreamingBuiltinTool,
			streaming:            true,
			expectModel:          "gpt-4.1",
			expectPromptRecorded: "Is 3 + 5 a prime number? Use the add function to calculate the sum.",
			expectToolRecorded: &recorder.ToolUsageRecord{
				MsgID:    "resp_0c3fb28cfcf463a500695fa2f0239481a095ec6ce3dfe4d458",
				Tool:     "add",
				Args:     map[string]any{"a": float64(3), "b": float64(5)},
				Injected: false,
			},
			expectTokenUsage: &recorder.TokenUsageRecord{
				MsgID:  "resp_0c3fb28cfcf463a500695fa2f0239481a095ec6ce3dfe4d458",
				Input:  58,
				Output: 18,
				ExtraTokenTypes: map[string]int64{
					"input_cached":     0,
					"output_reasoning": 0,
					"total_tokens":     76,
				},
			},
		},
		{
			name:                 "streaming_cached_tokens",
			fixture:              fixtures.OaiResponsesStreamingCachedInputTokens,
			streaming:            true,
			expectModel:          "gpt-5.2-codex",
			expectPromptRecorded: "Test cached input tokens.",
			expectTokenUsage: &recorder.TokenUsageRecord{
				MsgID:  "resp_05080461b406f3f501696a1409d34c8195a40ff4b092145c35",
				Input:  1165, // 16909 input - 15744 cached
				Output: 54,
				ExtraTokenTypes: map[string]int64{
					"input_cached":     15744,
					"output_reasoning": 0,
					"total_tokens":     16963,
				},
			},
		},
		{
			name:                 "streaming_custom_tool",
			fixture:              fixtures.OaiResponsesStreamingCustomTool,
			streaming:            true,
			expectModel:          "gpt-5",
			expectPromptRecorded: "Use the code_exec tool to print hello world to the console.",
			expectToolRecorded: &recorder.ToolUsageRecord{
				MsgID:    "resp_0c26996bc41c2a0500696942e83634819fb71b2b8ff8a4a76c",
				Tool:     "code_exec",
				Args:     "print(\"hello world\")",
				Injected: false,
			},
			expectTokenUsage: &recorder.TokenUsageRecord{
				MsgID:  "resp_0c26996bc41c2a0500696942e83634819fb71b2b8ff8a4a76c",
				Input:  64,
				Output: 340,
				ExtraTokenTypes: map[string]int64{
					"input_cached":     0,
					"output_reasoning": 320,
					"total_tokens":     404,
				},
			},
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
			expectTokenUsage: &recorder.TokenUsageRecord{
				MsgID:  "resp_0f9c4b2f224d858000695fa0649b8c8197b38914b15a7add0e",
				Input:  43,
				Output: 182,
				ExtraTokenTypes: map[string]int64{
					"input_cached":     0,
					"output_reasoning": 0,
					"total_tokens":     225,
				},
			},
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

			recordedTokens := mockRecorder.RecordedTokenUsages()
			if tc.expectTokenUsage != nil {
				require.Len(t, recordedTokens, 1)
				recordedTokens[0].InterceptionID = tc.expectTokenUsage.InterceptionID // ignore interception id
				recordedTokens[0].CreatedAt = tc.expectTokenUsage.CreatedAt           // ignore time
				require.Equal(t, tc.expectTokenUsage, recordedTokens[0])
			} else {
				require.Empty(t, recordedTokens)
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

// TestResponsesBlockingInjectedTool tests that injected MCP tool calls trigger the inner agentic loop,
// invoke the tool via MCP, and send the result back to the model.
func TestResponsesBlockingInjectedTool(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name             string
		fixture          []byte
		mcpToolName      string
		expectedToolArgs map[string]any
		expectedPrompt   string
		toolError        string // If non-empty, MCP tool returns this error.
	}{
		{
			name:        "success",
			fixture:     fixtures.OaiResponsesSingleInjectedTool,
			mcpToolName: "coder_template_version_parameters",
			expectedToolArgs: map[string]any{
				"template_version_id": "aa4e30e4-a086-4df6-a364-1343f1458104",
			},
			expectedPrompt: "list the template params for version aa4e30e4-a086-4df6-a364-1343f1458104",
		},
		{
			name:        "tool_error",
			fixture:     fixtures.OaiResponsesSingleInjectedToolError,
			mcpToolName: "coder_delete_template",
			expectedToolArgs: map[string]any{
				"template_id": "03cb4fdd-8109-4a22-8e22-bb4975171395",
			},
			expectedPrompt: "delete the template with ID 03cb4fdd-8109-4a22-8e22-bb4975171395, don't ask for confirmation",
			toolError:      "500 Internal error deleting template: unauthorized: rbac: forbidden",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			files := filesMap(txtar.Parse(tc.fixture))
			require.Contains(t, files, fixtureRequest)
			require.Contains(t, files, fixtureNonStreamingResponse)
			require.Contains(t, files, fixtureNonStreamingToolResponse)

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			// Setup mock server with response mutator for multi-turn interaction.
			mockAPI := newMockServer(ctx, t, files, func(reqCount uint32, resp []byte) []byte {
				if reqCount == 1 {
					return resp // First request gets the normal response (with tool call).
				}
				// Second request gets the tool response.
				return files[fixtureNonStreamingToolResponse]
			})
			t.Cleanup(mockAPI.Close)

			// Setup MCP server proxies (with mock tools).
			mcpProxiers, mcpCalls := setupMCPServerProxiesForTest(t, testTracer)
			if tc.toolError != "" {
				mcpCalls.setToolError(tc.mcpToolName, tc.toolError)
			}
			mcpMgr := mcp.NewServerProxyManager(mcpProxiers, testTracer)
			require.NoError(t, mcpMgr.Init(ctx))

			prov := provider.NewOpenAI(openaiCfg(mockAPI.URL, apiKey))
			mockRecorder := &testutil.MockRecorder{}
			logger := slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)

			bridge, err := aibridge.NewRequestBridge(ctx, []aibridge.Provider{prov}, mockRecorder, mcpMgr, logger, nil, testTracer)
			require.NoError(t, err)

			srv := httptest.NewUnstartedServer(bridge)
			srv.Config.BaseContext = func(_ net.Listener) context.Context {
				return aibcontext.AsActor(ctx, userID, nil)
			}
			srv.Start()
			t.Cleanup(srv.Close)

			req := createOpenAIResponsesReq(t, srv.URL, files[fixtureRequest])
			resp, err := http.DefaultClient.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()
			require.Equal(t, http.StatusOK, resp.StatusCode)

			body, err := io.ReadAll(resp.Body)
			require.NoError(t, err)

			// Wait for both requests to be made (inner agentic loop).
			require.Eventually(t, func() bool {
				return mockAPI.callCount.Load() == 2
			}, time.Second*10, time.Millisecond*50)

			// Verify the injected tool was invoked via MCP.
			invocations := mcpCalls.getCallsByTool(tc.mcpToolName)
			require.Len(t, invocations, 1, "expected MCP tool to be invoked once")

			// Verify the injected tool usage was recorded.
			toolUsages := mockRecorder.RecordedToolUsages()
			require.Len(t, toolUsages, 1)
			require.Equal(t, tc.mcpToolName, toolUsages[0].Tool)
			require.Equal(t, tc.expectedToolArgs, toolUsages[0].Args)
			require.True(t, toolUsages[0].Injected, "injected tool should be marked as injected")

			// Verify prompt was recorded.
			prompts := mockRecorder.RecordedPromptUsages()
			require.Len(t, prompts, 1)
			require.Equal(t, tc.expectedPrompt, prompts[0].Prompt)

			// Verify the response is the final tool response (after agentic loop).
			require.Equal(t, string(files[fixtureNonStreamingToolResponse]), string(body))
		})
	}
}
