package integrationtest

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/fixtures"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/provider"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func TestAnthropicMessages(t *testing.T) {
	t.Parallel()

	t.Run("single builtin tool", func(t *testing.T) {
		t.Parallel()

		cases := []struct {
			streaming            bool
			expectedInputTokens  int
			expectedOutputTokens int
			expectedToolCallID   string
		}{
			{
				streaming:            true,
				expectedInputTokens:  2,
				expectedOutputTokens: 66,
				expectedToolCallID:   "toolu_01RX68weRSquLx6HUTj65iBo",
			},
			{
				streaming:            false,
				expectedInputTokens:  5,
				expectedOutputTokens: 84,
				expectedToolCallID:   "toolu_01AusGgY5aKFhzWrFBv9JfHq",
			},
		}

		for _, tc := range cases {
			t.Run(fmt.Sprintf("%s/streaming=%v", t.Name(), tc.streaming), func(t *testing.T) {
				t.Parallel()

				ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
				t.Cleanup(cancel)

				fix := fixtures.Parse(t, fixtures.AntSingleBuiltinTool)
				upstream := newMockUpstream(t, ctx, newFixtureResponse(fix))

				ts := newBridgeTestServer(t, ctx,
					[]aibridge.Provider{provider.NewAnthropic(anthropicCfg(upstream.URL, apiKey), nil)},
				)

				// Make API call to aibridge for Anthropic /v1/messages
				reqBody, err := sjson.SetBytes(fix.Request(), "stream", tc.streaming)
				require.NoError(t, err)
				req := createAnthropicMessagesReq(t, ts.URL, reqBody)
				client := &http.Client{}
				resp, err := client.Do(req)
				require.NoError(t, err)
				require.Equal(t, http.StatusOK, resp.StatusCode)
				defer resp.Body.Close()

				// Response-specific checks.
				if tc.streaming {
					sp := aibridge.NewSSEParser()
					require.NoError(t, sp.Parse(resp.Body))

					// Ensure the message starts and completes, at a minimum.
					assert.Contains(t, sp.AllEvents(), "message_start")
					assert.Contains(t, sp.AllEvents(), "message_stop")
				}

				expectedTokenRecordings := 1
				if tc.streaming {
					// One for message_start, one for message_delta.
					expectedTokenRecordings = 2
				}
				tokenUsages := ts.Recorder.RecordedTokenUsages()
				require.Len(t, tokenUsages, expectedTokenRecordings)

				assert.EqualValues(t, tc.expectedInputTokens, calculateTotalInputTokens(tokenUsages), "input tokens miscalculated")
				assert.EqualValues(t, tc.expectedOutputTokens, calculateTotalOutputTokens(tokenUsages), "output tokens miscalculated")

				toolUsages := ts.Recorder.RecordedToolUsages()
				require.Len(t, toolUsages, 1)
				assert.Equal(t, "Read", toolUsages[0].Tool)
				assert.Equal(t, tc.expectedToolCallID, toolUsages[0].ToolCallID)
				require.IsType(t, json.RawMessage{}, toolUsages[0].Args)
				var args map[string]any
				require.NoError(t, json.Unmarshal(toolUsages[0].Args.(json.RawMessage), &args))
				require.Contains(t, args, "file_path")
				assert.Equal(t, "/tmp/blah/foo", args["file_path"])

				promptUsages := ts.Recorder.RecordedPromptUsages()
				require.Len(t, promptUsages, 1)
				assert.Equal(t, "read the foo file", promptUsages[0].Prompt)

				ts.Recorder.VerifyAllInterceptionsEnded(t)
			})
		}
	})
}

func TestAnthropicInjectedTools(t *testing.T) {
	t.Parallel()

	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming=%v", streaming), func(t *testing.T) {
			t.Parallel()

			// Build the requirements & make the assertions which are common to all providers.
			recorderClient, mockMCP, resp := setupInjectedToolTest(t, fixtures.AntSingleInjectedTool, streaming, newAnthropicProvider, defaultTracer, defaultActorID, createAnthropicMessagesReq, anthropicToolResultValidator(t))

			// Ensure expected tool was invoked with expected input.
			toolUsages := recorderClient.RecordedToolUsages()
			require.Len(t, toolUsages, 1)
			require.Equal(t, mockToolName, toolUsages[0].Tool)
			expected, err := json.Marshal(map[string]any{"owner": "admin"})
			require.NoError(t, err)
			actual, err := json.Marshal(toolUsages[0].Args)
			require.NoError(t, err)
			require.EqualValues(t, expected, actual)
			invocations := mockMCP.getCallsByTool(mockToolName)
			require.Len(t, invocations, 1)
			actual, err = json.Marshal(invocations[0])
			require.NoError(t, err)
			require.EqualValues(t, expected, actual)

			var (
				content *anthropic.ContentBlockUnion
				message anthropic.Message
			)
			if streaming {
				// Parse the response stream.
				decoder := ssestream.NewDecoder(resp)
				stream := ssestream.NewStream[anthropic.MessageStreamEventUnion](decoder, nil)
				for stream.Next() {
					event := stream.Current()
					require.NoError(t, message.Accumulate(event), "accumulate event")
				}

				require.NoError(t, stream.Err(), "stream error")
				require.Len(t, message.Content, 2)

				content = &message.Content[1]
			} else {
				// Parse & unmarshal the response.
				body, err := io.ReadAll(resp.Body)
				require.NoError(t, err, "read response body")

				require.NoError(t, json.Unmarshal(body, &message), "unmarshal response")
				require.GreaterOrEqual(t, len(message.Content), 1)

				content = &message.Content[0]
			}

			// Ensure tool returned expected value.
			require.NotNil(t, content)
			require.Contains(t, content.Text, "dd711d5c-83c6-4c08-a0af-b73055906e8c") // The ID of the workspace to be returned.

			// Check the token usage from the client's perspective.
			//
			// We overwrite the final message_delta which is relayed to the client to include the
			// accumulated tokens but currently the SDK only supports accumulating output tokens
			// for message_delta events.
			//
			// For non-streaming requests the token usage is also overwritten and should be faithfully
			// represented in the response.
			//
			// See https://github.com/anthropics/anthropic-sdk-go/blob/v1.12.0/message.go#L2619-L2622
			if !streaming {
				assert.EqualValues(t, 15308, message.Usage.InputTokens)
			}
			assert.EqualValues(t, 204, message.Usage.OutputTokens)

			// Ensure tokens used during injected tool invocation are accounted for.
			tokenUsages := recorderClient.RecordedTokenUsages()
			assert.EqualValues(t, 15308, calculateTotalInputTokens(tokenUsages))
			assert.EqualValues(t, 204, calculateTotalOutputTokens(tokenUsages))

			// Ensure we received exactly one prompt.
			promptUsages := recorderClient.RecordedPromptUsages()
			require.Len(t, promptUsages, 1)
		})
	}
}

// anthropicToolResultValidator returns a request validator that asserts the second
// upstream request contains the assistant's tool_use and user's tool_result messages
// appended by the inner agentic loop. If the raw payload is not kept in sync with
// the structured messages, the second request will be identical to the first.
func anthropicToolResultValidator(t *testing.T) func(*http.Request, []byte) {
	t.Helper()

	return func(_ *http.Request, raw []byte) {
		messages := gjson.GetBytes(raw, "messages").Array()

		// After the agentic loop the messages must contain at minimum:
		//   [0]   original user message
		//   [N-2] assistant message with tool_use content block
		//   [N-1] user message with tool_result content block
		require.GreaterOrEqual(t, len(messages), 3,
			"second upstream request must contain the original message, assistant tool_use, and user tool_result")

		assistantMsg := messages[len(messages)-2]
		require.Equal(t, "assistant", assistantMsg.Get("role").Str,
			"penultimate message must be from the assistant")
		var hasToolUse bool
		for _, block := range assistantMsg.Get("content").Array() {
			if block.Get("type").Str == "tool_use" {
				hasToolUse = true
				break
			}
		}
		require.True(t, hasToolUse, "assistant message must contain a tool_use content block")

		toolResultMsg := messages[len(messages)-1]
		require.Equal(t, "user", toolResultMsg.Get("role").Str,
			"last message must be a user message carrying the tool_result")
		var hasToolResult bool
		for _, block := range toolResultMsg.Get("content").Array() {
			if block.Get("type").Str == "tool_result" {
				hasToolResult = true
				break
			}
		}
		require.True(t, hasToolResult, "user message must contain a tool_result content block")
	}
}

// TestAnthropicToolChoiceParallelDisabled verifies that parallel tool use is
// correctly disabled based on the tool_choice parameter in the request.
// See https://github.com/coder/aibridge/issues/2
func TestAnthropicToolChoiceParallelDisabled(t *testing.T) {
	t.Parallel()

	var (
		toolChoiceAuto = string(constant.ValueOf[constant.Auto]())
		toolChoiceAny  = string(constant.ValueOf[constant.Any]())
		toolChoiceNone = string(constant.ValueOf[constant.None]())
		toolChoiceTool = string(constant.ValueOf[constant.Tool]())
	)

	cases := []struct {
		name                          string
		toolChoice                    any // nil, or map with "type" key.
		withInjectedTools             bool
		expectDisableParallel         bool
		expectToolChoiceTypeInRequest string
	}{
		// With injected tools - disable_parallel_tool_use should be set.
		{
			name:                          "with injected tools: no tool_choice defined defaults to auto",
			toolChoice:                    nil,
			withInjectedTools:             true,
			expectDisableParallel:         true,
			expectToolChoiceTypeInRequest: toolChoiceAuto,
		},
		{
			name:                          "with injected tools: tool_choice auto",
			toolChoice:                    map[string]any{"type": toolChoiceAuto},
			withInjectedTools:             true,
			expectDisableParallel:         true,
			expectToolChoiceTypeInRequest: toolChoiceAuto,
		},
		{
			name:                          "with injected tools: tool_choice any",
			toolChoice:                    map[string]any{"type": toolChoiceAny},
			withInjectedTools:             true,
			expectDisableParallel:         true,
			expectToolChoiceTypeInRequest: toolChoiceAny,
		},
		{
			name:                          "with injected tools: tool_choice tool",
			toolChoice:                    map[string]any{"type": toolChoiceTool, "name": "some_tool"},
			withInjectedTools:             true,
			expectDisableParallel:         true,
			expectToolChoiceTypeInRequest: toolChoiceTool,
		},
		{
			name:                          "with injected tools: tool_choice none",
			toolChoice:                    map[string]any{"type": toolChoiceNone},
			withInjectedTools:             true,
			expectDisableParallel:         false,
			expectToolChoiceTypeInRequest: toolChoiceNone,
		},
		// Without injected tools - disable_parallel_tool_use should NOT be set.
		{
			name:                          "without injected tools: tool_choice auto",
			toolChoice:                    map[string]any{"type": toolChoiceAuto},
			withInjectedTools:             false,
			expectDisableParallel:         false,
			expectToolChoiceTypeInRequest: toolChoiceAuto,
		},
		{
			name:                          "without injected tools: tool_choice any",
			toolChoice:                    map[string]any{"type": toolChoiceAny},
			withInjectedTools:             false,
			expectDisableParallel:         false,
			expectToolChoiceTypeInRequest: toolChoiceAny,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			// Setup MCP tools conditionally.
			var mcpMgr mcp.ServerProxier
			if tc.withInjectedTools {
				mcpMgr = setupMCPForTest(t, defaultTracer)
			} else {
				mcpMgr = newNoopMCPManager()
			}

			fix := fixtures.Parse(t, fixtures.AntSimple)
			upstream := newMockUpstream(t, ctx, newFixtureResponse(fix))

			ts := newBridgeTestServer(t, ctx,
				[]aibridge.Provider{provider.NewAnthropic(anthropicCfg(upstream.URL, apiKey), nil)},
				withMCP(mcpMgr),
			)

			// Prepare request body with tool_choice set.
			reqBody, err := sjson.SetBytes(fix.Request(), "tool_choice", tc.toolChoice)
			require.NoError(t, err)

			req := createAnthropicMessagesReq(t, ts.URL, reqBody)
			client := &http.Client{}
			resp, err := client.Do(req)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, resp.StatusCode)
			_ = resp.Body.Close()

			// Verify tool_choice in the upstream request.
			received := upstream.receivedRequests()
			require.Len(t, received, 1)
			var receivedRequest map[string]any
			require.NoError(t, json.Unmarshal(received[0].Body, &receivedRequest))
			toolChoice, ok := receivedRequest["tool_choice"].(map[string]any)
			require.True(t, ok, "expected tool_choice in upstream request")

			// Verify the type matches expectation.
			assert.Equal(t, tc.expectToolChoiceTypeInRequest, toolChoice["type"])

			// Verify name is preserved for tool_choice=tool.
			if tc.expectToolChoiceTypeInRequest == toolChoiceTool {
				assert.Equal(t, "some_tool", toolChoice["name"])
			}

			// Verify disable_parallel_tool_use based on expectations.
			// See https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#parallel-tool-use
			disableParallel, hasDisableParallel := toolChoice["disable_parallel_tool_use"].(bool)

			if tc.expectDisableParallel {
				require.True(t, hasDisableParallel, "expected disable_parallel_tool_use in tool_choice")
				assert.True(t, disableParallel, "expected disable_parallel_tool_use to be true")
			} else {
				assert.False(t, hasDisableParallel, "expected disable_parallel_tool_use to not be set")
			}
		})
	}
}

func TestThinkingAdaptiveIsPreserved(t *testing.T) {
	t.Parallel()

	fix := fixtures.Parse(t, fixtures.AntSimple)

	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming=%v", streaming), func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			// Create a mock server that captures the request body sent upstream.
			upstream := newMockUpstream(t, ctx, newFixtureResponse(fix))

			ts := newBridgeTestServer(t, ctx,
				[]aibridge.Provider{provider.NewAnthropic(anthropicCfg(upstream.URL, apiKey), nil)},
			)

			// Inject adaptive thinking into the fixture request.
			reqBody, err := sjson.SetBytes(fix.Request(), "thinking", map[string]string{"type": "adaptive"})
			require.NoError(t, err)
			reqBody, err = sjson.SetBytes(reqBody, "stream", streaming)
			require.NoError(t, err)

			req := createAnthropicMessagesReq(t, ts.URL, reqBody)
			resp, err := http.DefaultClient.Do(req)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, resp.StatusCode)
			_, _ = io.ReadAll(resp.Body)
			_ = resp.Body.Close()

			// Verify the thinking field was preserved in the upstream request.
			received := upstream.receivedRequests()
			require.Len(t, received, 1)
			assert.Equal(t, "adaptive", gjson.GetBytes(received[0].Body, "thinking.type").Str)
		})
	}
}
