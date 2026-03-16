package messages

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/coder/aibridge/utils"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestNewMessagesRequestPayload(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name string

		requestBody []byte

		expectError bool
	}{
		{
			name:        "empty body",
			requestBody: []byte("   \n\t  "),
			expectError: true,
		},
		{
			name:        "invalid json",
			requestBody: []byte(`{"model":`),
			expectError: true,
		},
		{
			name:        "valid json",
			requestBody: []byte(`{"model":"claude-opus-4-5","max_tokens":1024}`),
			expectError: false,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			payload, err := NewMessagesRequestPayload(testCase.requestBody)
			if testCase.expectError {
				require.Error(t, err)
				require.Nil(t, payload)
				return
			}

			require.NoError(t, err)
			require.Equal(t, MessagesRequestPayload(testCase.requestBody), payload)
		})
	}
}

func TestMessagesRequestPayloadStream(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name string

		requestBody string

		expectedStream bool
	}{
		{
			name:           "stream true",
			requestBody:    `{"stream":true}`,
			expectedStream: true,
		},
		{
			name:           "stream false",
			requestBody:    `{"stream":false}`,
			expectedStream: false,
		},
		{
			name:           "stream missing",
			requestBody:    `{"model":"claude-opus-4-5"}`,
			expectedStream: false,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			payload := mustMessagesPayload(t, testCase.requestBody)
			require.Equal(t, testCase.expectedStream, payload.Stream())
		})
	}
}

func TestMessagesRequestPayloadLastUserPrompt(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name string

		requestBody string

		expectedPrompt string

		expectedFound bool

		expectError bool
	}{
		{
			name:           "last user message string content",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}]}`,
			expectedPrompt: "hello",
			expectedFound:  true,
			expectError:    false,
		},
		{
			name:           "last user message typed content returns last text block",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}},{"type":"text","text":"first"},{"type":"text","text":"last"}]}]}`,
			expectedPrompt: "last",
			expectedFound:  true,
			expectError:    false,
		},
		{
			name:           "last message not from user",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"assistant","content":"hello"}]}`,
			expectedPrompt: "",
			expectedFound:  false,
			expectError:    false,
		},
		{
			name:           "no messages key",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024}`,
			expectedPrompt: "",
			expectedFound:  false,
			expectError:    false,
		},
		{
			name:           "empty messages array",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[]}`,
			expectedPrompt: "",
			expectedFound:  false,
			expectError:    false,
		},
		{
			name:           "last user message with empty content array",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":[]}]}`,
			expectedPrompt: "",
			expectedFound:  false,
			expectError:    false,
		},
		{
			name:           "last user message with only non text content",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}},{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"def"}}]}]}`,
			expectedPrompt: "",
			expectedFound:  false,
			expectError:    false,
		},
		{
			name:           "multiple messages with last being user",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"first"},{"role":"assistant","content":[{"type":"text","text":"response"}]},{"role":"user","content":"second"}]}`,
			expectedPrompt: "second",
			expectedFound:  true,
			expectError:    false,
		},
		{
			name:           "messages wrong type returns error",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024,"messages":{}}`,
			expectedPrompt: "",
			expectedFound:  false,
			expectError:    true,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			payload := mustMessagesPayload(t, testCase.requestBody)
			prompt, found, err := payload.lastUserPrompt()
			if testCase.expectError {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.Equal(t, testCase.expectedFound, found)
			require.Equal(t, testCase.expectedPrompt, prompt)
		})
	}
}

func TestMessagesRequestPayloadCorrelatingToolCallID(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name string

		requestBody string

		expectedToolUseID *string
	}{
		{
			name:              "no tool result block",
			requestBody:       `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}]}`,
			expectedToolUseID: nil,
		},
		{
			name:              "returns last tool result from final message",
			requestBody:       `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_first","content":"first"},{"type":"tool_result","tool_use_id":"toolu_second","content":"second"}]}]}`,
			expectedToolUseID: utils.PtrTo("toolu_second"),
		},
		{
			name:              "ignores earlier message tool result",
			requestBody:       `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_first","content":"first"}]},{"role":"assistant","content":"done"}]}`,
			expectedToolUseID: nil,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			payload := mustMessagesPayload(t, testCase.requestBody)
			require.Equal(t, testCase.expectedToolUseID, payload.correlatingToolCallID())
		})
	}
}

func TestMessagesRequestPayloadInjectTools(t *testing.T) {
	t.Parallel()

	payload := mustMessagesPayload(t, `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}],"tools":[{"name":"existing_tool","type":"custom","input_schema":{"type":"object","properties":{}},"cache_control":{"type":"ephemeral"}}]}`)

	updatedPayload, err := payload.injectTools([]anthropic.ToolUnionParam{
		{
			OfTool: &anthropic.ToolParam{
				Name: "injected_tool",
				Type: anthropic.ToolTypeCustom,
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: map[string]interface{}{},
				},
			},
		},
	})
	require.NoError(t, err)

	toolItems := gjson.GetBytes(updatedPayload, "tools").Array()
	require.Len(t, toolItems, 2)
	require.Equal(t, "injected_tool", toolItems[0].Get("name").String())
	require.Equal(t, "existing_tool", toolItems[1].Get("name").String())
	require.Equal(t, "ephemeral", toolItems[1].Get("cache_control.type").String())
}

func TestMessagesRequestPayloadDisableParallelToolCalls(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name string

		requestBody string

		expectedType string

		expectedDisableParallel *bool
	}{
		{
			name:                    "defaults to auto when missing",
			requestBody:             `{"model":"claude-opus-4-5","max_tokens":1024}`,
			expectedType:            string(constant.ValueOf[constant.Auto]()),
			expectedDisableParallel: utils.PtrTo(true),
		},
		{
			name:                    "auto gets disabled",
			requestBody:             `{"tool_choice":{"type":"auto"}}`,
			expectedType:            string(constant.ValueOf[constant.Auto]()),
			expectedDisableParallel: utils.PtrTo(true),
		},
		{
			name:                    "any gets disabled",
			requestBody:             `{"tool_choice":{"type":"any"}}`,
			expectedType:            string(constant.ValueOf[constant.Any]()),
			expectedDisableParallel: utils.PtrTo(true),
		},
		{
			name:                    "tool gets disabled",
			requestBody:             `{"tool_choice":{"type":"tool","name":"abc"}}`,
			expectedType:            string(constant.ValueOf[constant.Tool]()),
			expectedDisableParallel: utils.PtrTo(true),
		},
		{
			name:                    "none remains unchanged",
			requestBody:             `{"tool_choice":{"type":"none"}}`,
			expectedType:            string(constant.ValueOf[constant.None]()),
			expectedDisableParallel: nil,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			payload := mustMessagesPayload(t, testCase.requestBody)
			updatedPayload, err := payload.disableParallelToolCalls()
			require.NoError(t, err)

			toolChoice := gjson.GetBytes(updatedPayload, "tool_choice")
			require.Equal(t, testCase.expectedType, toolChoice.Get("type").String())

			disableParallelResult := toolChoice.Get("disable_parallel_tool_use")
			if testCase.expectedDisableParallel == nil {
				require.False(t, disableParallelResult.Exists())
				return
			}

			require.True(t, disableParallelResult.Exists())
			require.Equal(t, *testCase.expectedDisableParallel, disableParallelResult.Bool())
		})
	}
}

func TestMessagesRequestPayloadAppendedMessages(t *testing.T) {
	t.Parallel()

	payload := mustMessagesPayload(t, `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}]}`)

	updatedPayload, err := payload.appendedMessages([]anthropic.MessageParam{
		{
			Role: anthropic.MessageParamRoleAssistant,
			Content: []anthropic.ContentBlockParamUnion{
				anthropic.NewTextBlock("assistant response"),
			},
		},
		anthropic.NewUserMessage(anthropic.NewToolResultBlock("toolu_123", "tool output", false)),
	})
	require.NoError(t, err)

	messageItems := gjson.GetBytes(updatedPayload, "messages").Array()
	require.Len(t, messageItems, 3)
	require.Equal(t, "hello", messageItems[0].Get("content").String())
	require.Equal(t, "assistant", messageItems[1].Get("role").String())
	require.Equal(t, "assistant response", messageItems[1].Get("content.0.text").String())
	require.Equal(t, "tool_result", messageItems[2].Get("content.0.type").String())
	require.Equal(t, "toolu_123", messageItems[2].Get("content.0.tool_use_id").String())
}
