package responses

import (
	"testing"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/fixtures"
	"github.com/coder/aibridge/internal/testutil"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

func TestLastUserPrompt(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		reqPayload []byte
		expected   string
	}{
		{
			name:       "simple_string_input",
			reqPayload: fixtures.Request(t, fixtures.OaiResponsesBlockingSimple),
			expected:   "tell me a joke",
		},
		{
			name:       "array_single_input_string",
			reqPayload: fixtures.Request(t, fixtures.OaiResponsesBlockingBuiltinTool),
			expected:   "Is 3 + 5 a prime number? Use the add function to calculate the sum.",
		},
		{
			name:       "array_multiple_items_content_objects",
			reqPayload: fixtures.Request(t, fixtures.OaiResponsesStreamingCodex),
			expected:   "hello",
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

			prompt, err := base.lastUserPrompt()
			require.NoError(t, err)
			require.Equal(t, tc.expected, prompt)
		})
	}
}

func TestLastUserPromptErr(t *testing.T) {
	t.Parallel()

	t.Run("nil_struct", func(t *testing.T) {
		t.Parallel()

		var base *responsesInterceptionBase
		prompt, err := base.lastUserPrompt()
		require.Error(t, err)
		require.Empty(t, prompt)
		require.Contains(t, "cannot get last user prompt: nil struct", err.Error())
	})

	t.Run("nil_struct", func(t *testing.T) {
		t.Parallel()

		base := responsesInterceptionBase{}
		prompt, err := base.lastUserPrompt()
		require.Error(t, err)
		require.Empty(t, prompt)
		require.Contains(t, "cannot get last user prompt: nil req struct", err.Error())
	})

	tests := []struct {
		name       string
		reqPayload []byte
		wantErrMsg string
	}{
		{
			name:       "empty_input",
			reqPayload: []byte(`{"model": "gpt-4o", "input": []}`),
			wantErrMsg: "failed to find last user prompt",
		},
		{
			name:       "no_user_role",
			reqPayload: []byte(`{"model": "gpt-4o", "input": [{"role": "assistant", "content": "hello"}]}`),
			wantErrMsg: "failed to find last user prompt",
		},
		{
			name:       "user_with_empty_content",
			reqPayload: []byte(`{"model": "gpt-4o", "input": [{"role": "user", "content": ""}]}`),
			wantErrMsg: "failed to find last user prompt",
		},
		{
			name:       "user_with_empty_content_array",
			reqPayload: []byte(`{"model": "gpt-4o", "input": [{"role": "user", "content": []}]}`),
			wantErrMsg: "failed to find last user prompt",
		},
		{
			name:       "user_with_non_input_text_content",
			reqPayload: []byte(`{"model": "gpt-4o", "input": [{"role": "user", "content": [{"type": "input_image", "url": "http://example.com/img.png"}]}]}`),
			wantErrMsg: "failed to find last user prompt",
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

			prompt, err := base.lastUserPrompt()
			require.Error(t, err)
			require.Empty(t, prompt)
			require.Contains(t, tc.wantErrMsg, err.Error())
		})
	}
}

func TestRecordPrompt(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		reqPayload   []byte
		responseID   string
		wantRecorded bool
		wantPrompt   string
	}{
		{
			name:         "records_prompt_successfully",
			reqPayload:   fixtures.Request(t, fixtures.OaiResponsesBlockingSimple),
			responseID:   "resp_123",
			wantRecorded: true,
			wantPrompt:   "tell me a joke",
		},
		{
			name:         "skips_recording_on_empty_response_id",
			reqPayload:   fixtures.Request(t, fixtures.OaiResponsesBlockingSimple),
			responseID:   "",
			wantRecorded: false,
		},
		{
			name:         "skips_recording_on_lastUserPrompt_error",
			reqPayload:   []byte(`{"model": "gpt-4o", "input": []}`),
			responseID:   "resp_123",
			wantRecorded: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := &ResponsesNewParamsWrapper{}
			err := req.UnmarshalJSON(tc.reqPayload)
			require.NoError(t, err)

			rec := &testutil.MockRecorder{}
			id := uuid.New()
			base := &responsesInterceptionBase{
				id:         id,
				req:        req,
				reqPayload: tc.reqPayload,
				recorder:   rec,
				logger:     slog.Make(),
			}

			base.recordUserPrompt(t.Context(), tc.responseID)

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
