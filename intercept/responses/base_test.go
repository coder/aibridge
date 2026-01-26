package responses

import (
	"net/http"
	"testing"
	"time"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/fixtures"
	"github.com/coder/aibridge/internal/testutil"
	"github.com/coder/aibridge/recorder"
	"github.com/google/uuid"
	oairesponses "github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/require"
)

func TestLastUserPrompt(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		reqPayload []byte
		expect     string
	}{
		{
			name:       "input_empty_string",
			reqPayload: []byte(`{"input": ""}`),
			expect:     "",
		},
		{
			name:       "input_array_content_empty_string",
			reqPayload: []byte(`{"model": "gpt-4o", "input": [{"role": "user", "content": ""}]}`),
			expect:     "",
		},
		{
			name:       "input_array_content_array_empty_string",
			reqPayload: []byte(`{"model": "gpt-4o", "input": [ { "role": "user", "content": [{"type": "input_text", "text": ""}] } ] }`),
		},
		{
			name:       "input_array_content_array_multiple_inputs",
			reqPayload: []byte(`{"model": "gpt-4o", "input": [ { "role": "user", "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}] } ] }`),
			expect:     "a\nb",
		},
		{
			name:       "simple_string_input",
			reqPayload: fixtures.Request(t, fixtures.OaiResponsesBlockingSimple),
			expect:     "tell me a joke",
		},
		{
			name:       "array_single_input_string",
			reqPayload: fixtures.Request(t, fixtures.OaiResponsesBlockingSingleBuiltinTool),
			expect:     "Is 3 + 5 a prime number? Use the add function to calculate the sum.",
		},
		{
			name:       "array_multiple_items_content_objects",
			reqPayload: fixtures.Request(t, fixtures.OaiResponsesStreamingCodex),
			expect:     "hello",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := &ResponsesNewParamsWrapper{}
			err := req.UnmarshalJSON(tc.reqPayload)
			require.NoError(t, err)

			base := &responsesInterceptionBase{
				req:        req,
				reqPayload: tc.reqPayload,
			}

			prompt, promptFound, err := base.lastUserPrompt(t.Context())
			require.NoError(t, err)
			require.Equal(t, tc.expect, prompt)
			require.True(t, promptFound)
		})
	}
}

func TestLastUserPromptNotFound(t *testing.T) {
	t.Parallel()

	t.Run("nil_struct", func(t *testing.T) {
		t.Parallel()

		var base *responsesInterceptionBase
		prompt, promptFound, err := base.lastUserPrompt(t.Context())
		require.Error(t, err)
		require.Empty(t, prompt)
		require.False(t, promptFound)
		require.Contains(t, "cannot get last user prompt: nil struct", err.Error())
	})

	t.Run("nil_request", func(t *testing.T) {
		t.Parallel()

		base := responsesInterceptionBase{}
		prompt, promptFound, err := base.lastUserPrompt(t.Context())
		require.Error(t, err)
		require.Empty(t, prompt)
		require.False(t, promptFound)
		require.Contains(t, "cannot get last user prompt: nil request struct", err.Error())
	})

	// Cases where the user prompt is not found / wrong format.
	tests := []struct {
		name       string
		reqPayload []byte
		expectErr  string
	}{
		{
			name:       "non_existing_input",
			reqPayload: []byte(`{"model": "gpt-4o"}`),
		},
		{
			name:       "input_empty_array",
			reqPayload: []byte(`{"model": "gpt-4o", "input": []}`),
		},
		{
			name:       "input_integer",
			reqPayload: []byte(`{"model": "gpt-4o", "input": 123}`),
			expectErr:  "unexpected input type",
		},
		{
			name:       "no_user_role",
			reqPayload: []byte(`{"model": "gpt-4o", "input": [{"role": "assistant", "content": "hello"}]}`),
		},
		{
			name:       "user_with_empty_content_array",
			reqPayload: []byte(`{"model": "gpt-4o", "input": [{"role": "user", "content": []}]}`),
		},
		{
			name:       "input_array_integer",
			reqPayload: []byte(`{"model": "gpt-4o", "input": [{"role": "user", "content": 123}]}`),
			expectErr:  "unexpected input content type",
		},
		{
			name:       "user_with_non_input_text_content",
			reqPayload: []byte(`{"model": "gpt-4o", "input": [{"role": "user", "content": [{"type": "input_image", "url": "http://example.com/img.png"}]}]}`),
		},
		{
			name:       "user_content_not_last",
			reqPayload: []byte(`{"model": "gpt-4o", "input": [ {"role": "user", "content":"input"}, {"role": "assistant", "content": "hello"} ]}`),
		},
		{
			name:       "input_array_content_array_integer",
			reqPayload: []byte(`{"model": "gpt-4o", "input": [ { "role": "user", "content": [{"type": "input_text", "text": 123}] } ] }`),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := &ResponsesNewParamsWrapper{}
			err := req.UnmarshalJSON(tc.reqPayload)
			require.NoError(t, err)

			base := &responsesInterceptionBase{
				req:        req,
				reqPayload: tc.reqPayload,
			}

			prompt, promptFound, err := base.lastUserPrompt(t.Context())
			if tc.expectErr != "" {
				require.Error(t, err)
				require.Contains(t, err.Error(), tc.expectErr)
			} else {
				require.NoError(t, err)
			}
			require.Empty(t, prompt)
			require.False(t, promptFound)
		})
	}
}

func TestRecordPrompt(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name              string
		promptWasRecorded bool
		prompt            string
		responseID        string
		wantRecorded      bool
		wantPrompt        string
	}{
		{
			name:         "records_prompt_successfully",
			prompt:       "tell me a joke",
			responseID:   "resp_123",
			wantRecorded: true,
			wantPrompt:   "tell me a joke",
		},
		{
			name:         "records_empty_prompt_successfully",
			prompt:       "",
			responseID:   "resp_123",
			wantRecorded: true,
			wantPrompt:   "",
		},
		{
			name:         "skips_recording_on_empty_response_id",
			prompt:       "tell me a joke",
			responseID:   "",
			wantRecorded: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			rec := &testutil.MockRecorder{}
			id := uuid.New()
			base := &responsesInterceptionBase{
				id:       id,
				recorder: rec,
				logger:   slog.Make(),
			}

			base.recordUserPrompt(t.Context(), tc.responseID, tc.prompt)

			prompts := rec.RecordedPromptUsages()
			if tc.wantRecorded {
				require.Len(t, prompts, 1)
				require.Equal(t, id.String(), prompts[0].InterceptionID)
				require.Equal(t, tc.responseID, prompts[0].MsgID)
				require.Equal(t, tc.wantPrompt, prompts[0].Prompt)
			} else {
				require.Empty(t, prompts)
			}
		})
	}
}

func TestRecordToolUsage(t *testing.T) {
	t.Parallel()

	id := uuid.MustParse("11111111-1111-1111-1111-111111111111")

	tests := []struct {
		name     string
		response *oairesponses.Response
		expected []*recorder.ToolUsageRecord
	}{
		{
			name:     "nil_response",
			response: nil,
			expected: nil,
		},
		{
			name: "empty_output",
			response: &oairesponses.Response{
				ID: "resp_123",
			},
			expected: nil,
		},
		{
			name: "empty_tool_args",
			response: &oairesponses.Response{
				ID: "resp_456",
				Output: []oairesponses.ResponseOutputItemUnion{
					{
						Type:      "function_call",
						Name:      "get_weather",
						Arguments: "",
					},
				},
			},
			expected: []*recorder.ToolUsageRecord{
				{
					InterceptionID: id.String(),
					MsgID:          "resp_456",
					Tool:           "get_weather",
					Args:           "",
					Injected:       false,
				},
			},
		},
		{
			name: "multiple_tool_calls",
			response: &oairesponses.Response{
				ID: "resp_789",
				Output: []oairesponses.ResponseOutputItemUnion{
					{
						Type:      "function_call",
						Name:      "get_weather",
						Arguments: `{"location": "NYC"}`,
					},
					{
						Type:      "function_call",
						Name:      "bad_json_args",
						Arguments: `{"bad": args`,
					},
					{
						Type: "message",
						ID:   "msg_1",
						Role: "assistant",
					},
					{
						Type:  "custom_tool_call",
						Name:  "search",
						Input: `{\"query\": \"test\"}`,
					},
					{
						Type:      "function_call",
						Name:      "calculate",
						Arguments: `{"a": 1, "b": 2}`,
					},
				},
			},
			expected: []*recorder.ToolUsageRecord{
				{
					InterceptionID: id.String(),
					MsgID:          "resp_789",
					Tool:           "get_weather",
					Args:           map[string]any{"location": "NYC"},
					Injected:       false,
				},
				{
					InterceptionID: id.String(),
					MsgID:          "resp_789",
					Tool:           "bad_json_args",
					Args:           `{"bad": args`,
					Injected:       false,
				},
				{
					InterceptionID: id.String(),
					MsgID:          "resp_789",
					Tool:           "search",
					Args:           `{\"query\": \"test\"}`,
					Injected:       false,
				},
				{
					InterceptionID: id.String(),
					MsgID:          "resp_789",
					Tool:           "calculate",
					Args:           map[string]any{"a": float64(1), "b": float64(2)},
					Injected:       false,
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			rec := &testutil.MockRecorder{}
			base := &responsesInterceptionBase{
				id:       id,
				recorder: rec,
				logger:   slog.Make(),
			}

			base.recordNonInjectedToolUsage(t.Context(), tc.response)

			tools := rec.RecordedToolUsages()
			require.Len(t, tools, len(tc.expected))
			for i, got := range tools {
				got.CreatedAt = time.Time{}
				require.Equal(t, tc.expected[i], got)
			}
		})
	}
}

func TestParseJSONArgs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		raw      string
		expected recorder.ToolArgs
	}{
		{
			name:     "empty_string",
			raw:      "",
			expected: "",
		},
		{
			name:     "whitespace_only",
			raw:      "   \t\n  ",
			expected: "",
		},
		{
			name:     "invalid_json",
			raw:      "{not valid json}",
			expected: "{not valid json}",
		},
		{
			name: "nested_object_with_trailing_spaces",
			raw:  ` {"user": {"name": "alice", "settings": {"theme": "dark", "notifications": true}}, "count": 42}   `,
			expected: map[string]any{
				"user": map[string]any{
					"name": "alice",
					"settings": map[string]any{
						"theme":         "dark",
						"notifications": true,
					},
				},
				"count": float64(42),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			base := &responsesInterceptionBase{}
			result := base.parseFunctionCallJSONArgs(t.Context(), tc.raw)
			require.Equal(t, tc.expected, result)
		})
	}
}

func TestRecordTokenUsage(t *testing.T) {
	t.Parallel()

	id := uuid.MustParse("22222222-2222-2222-2222-222222222222")

	tests := []struct {
		name     string
		response *oairesponses.Response
		expected *recorder.TokenUsageRecord
	}{
		{
			name:     "nil_response",
			response: nil,
			expected: nil,
		},
		{
			name: "with_all_token_details",
			response: &oairesponses.Response{
				ID: "resp_full",
				Usage: oairesponses.ResponseUsage{
					InputTokens:  10,
					OutputTokens: 20,
					TotalTokens:  30,
					InputTokensDetails: oairesponses.ResponseUsageInputTokensDetails{
						CachedTokens: 5,
					},
					OutputTokensDetails: oairesponses.ResponseUsageOutputTokensDetails{
						ReasoningTokens: 5,
					},
				},
			},
			expected: &recorder.TokenUsageRecord{
				InterceptionID: id.String(),
				MsgID:          "resp_full",
				Input:          5, // 10 input - 5 cached
				Output:         20,
				ExtraTokenTypes: map[string]int64{
					"input_cached":     5,
					"output_reasoning": 5,
					"total_tokens":     30,
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			rec := &testutil.MockRecorder{}
			base := &responsesInterceptionBase{
				id:       id,
				recorder: rec,
				logger:   slog.Make(),
			}

			base.recordTokenUsage(t.Context(), tc.response)

			tokens := rec.RecordedTokenUsages()
			if tc.expected == nil {
				require.Empty(t, tokens)
			} else {
				require.Len(t, tokens, 1)
				got := tokens[0]
				got.CreatedAt = time.Time{} // ignore time
				require.Equal(t, tc.expected, got)
			}
		})
	}
}

type mockResponseWriter struct {
	headerCalled      bool
	writeCalled       bool
	writeHeaderCalled bool
}

func (mrw *mockResponseWriter) Header() http.Header {
	mrw.headerCalled = true
	return http.Header{}
}

func (mrw *mockResponseWriter) Write([]byte) (int, error) {
	mrw.writeCalled = true
	return 0, nil
}

func (mrw *mockResponseWriter) WriteHeader(statusCode int) {
	mrw.writeHeaderCalled = true
}

func TestResponseCopierDoesntSendIfNoResponseReceived(t *testing.T) {
	mrw := mockResponseWriter{}

	respCopy := responseCopier{}
	body := "test_body"
	respCopy.buff.Write([]byte(body))

	respCopy.forwardResp(&mrw)
	require.False(t, mrw.headerCalled)
	require.False(t, mrw.writeCalled)
	require.False(t, mrw.writeHeaderCalled)

	// after response is received data is forwarded
	respCopy.responseReceived.Store(true)

	respCopy.forwardResp(&mrw)
	require.True(t, mrw.headerCalled)
	require.True(t, mrw.writeCalled)
	require.True(t, mrw.writeHeaderCalled)
}
