package integrationtest

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/coder/aibridge"
	"github.com/coder/aibridge/fixtures"
	"github.com/openai/openai-go/v3"
	oaissestream "github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func TestOpenAIChatCompletions(t *testing.T) {
	t.Parallel()

	t.Run("single builtin tool", func(t *testing.T) {
		t.Parallel()

		cases := []struct {
			streaming                                 bool
			expectedInputTokens, expectedOutputTokens int
			expectedToolCallID                        string
		}{
			{
				streaming:            true,
				expectedInputTokens:  60,
				expectedOutputTokens: 15,
				expectedToolCallID:   "call_HjeqP7YeRkoNj0de9e3U4X4B",
			},
			{
				streaming:            false,
				expectedInputTokens:  60,
				expectedOutputTokens: 15,
				expectedToolCallID:   "call_KjzAbhiZC6nk81tQzL7pwlpc",
			},
		}

		for _, tc := range cases {
			t.Run(fmt.Sprintf("%s/streaming=%v", t.Name(), tc.streaming), func(t *testing.T) {
				t.Parallel()

				ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
				t.Cleanup(cancel)

				fix := fixtures.Parse(t, fixtures.OaiChatSingleBuiltinTool)
				upstream := newMockUpstream(t, ctx, newFixtureResponse(fix))

				ts := newBridgeTestServer(t, ctx, upstream.URL)

				// Make API call to aibridge for OpenAI /v1/chat/completions
				reqBody, err := sjson.SetBytes(fix.Request(), "stream", tc.streaming)
				require.NoError(t, err)
				req := ts.newRequest(t, pathOpenAIChatCompletions, reqBody)

				client := &http.Client{}
				resp, err := client.Do(req)
				require.NoError(t, err)
				require.Equal(t, http.StatusOK, resp.StatusCode)
				defer resp.Body.Close()

				// Response-specific checks.
				if tc.streaming {
					sp := aibridge.NewSSEParser()
					require.NoError(t, sp.Parse(resp.Body))

					// OpenAI sends all events under the same type.
					messageEvents := sp.MessageEvents()
					assert.NotEmpty(t, messageEvents)

					// OpenAI streaming ends with [DONE]
					lastEvent := messageEvents[len(messageEvents)-1]
					assert.Equal(t, "[DONE]", lastEvent.Data)
				}

				tokenUsages := ts.Recorder.RecordedTokenUsages()
				require.Len(t, tokenUsages, 1)
				assert.EqualValues(t, tc.expectedInputTokens, calculateTotalInputTokens(tokenUsages), "input tokens miscalculated")
				assert.EqualValues(t, tc.expectedOutputTokens, calculateTotalOutputTokens(tokenUsages), "output tokens miscalculated")

				toolUsages := ts.Recorder.RecordedToolUsages()
				require.Len(t, toolUsages, 1)
				assert.Equal(t, "read_file", toolUsages[0].Tool)
				assert.Equal(t, tc.expectedToolCallID, toolUsages[0].ToolCallID)
				require.IsType(t, map[string]any{}, toolUsages[0].Args)
				require.Contains(t, toolUsages[0].Args, "path")
				assert.Equal(t, "README.md", toolUsages[0].Args.(map[string]any)["path"])

				promptUsages := ts.Recorder.RecordedPromptUsages()
				require.Len(t, promptUsages, 1)
				assert.Equal(t, "how large is the README.md file in my current path", promptUsages[0].Prompt)

				ts.Recorder.VerifyAllInterceptionsEnded(t)
			})
		}
	})

	t.Run("streaming injected tool call edge cases", func(t *testing.T) {
		t.Parallel()

		cases := []struct {
			name         string
			fixture      []byte
			expectedArgs map[string]any
		}{
			{
				name:         "tool call no preamble",
				fixture:      fixtures.OaiChatStreamingInjectedToolNoPreamble,
				expectedArgs: map[string]any{"owner": "me"},
			},
			{
				name:         "tool call with non-zero index",
				fixture:      fixtures.OaiChatStreamingInjectedToolNonzeroIndex,
				expectedArgs: nil, // No arguments in this fixture
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
				t.Cleanup(cancel)

				// Setup mock server for multi-turn interaction.
				// First request → tool call response, second → tool response.
				fix := fixtures.Parse(t, tc.fixture)
				upstream := newMockUpstream(t, ctx, newFixtureResponse(fix), newFixtureToolResponse(fix))

				// Setup MCP proxies with the tool from the fixture
				mockMCP := setupMCPForTest(t, defaultTracer)

				ts := newBridgeTestServer(t, ctx, upstream.URL,
					withMCP(mockMCP),
				)

				// Add the stream param to the request.
				reqBody, err := sjson.SetBytes(fix.Request(), "stream", true)
				require.NoError(t, err)
				req := ts.newRequest(t, pathOpenAIChatCompletions, reqBody)

				client := &http.Client{}
				resp, err := client.Do(req)
				require.NoError(t, err)
				require.Equal(t, http.StatusOK, resp.StatusCode)

				// Verify SSE headers are sent correctly
				require.Equal(t, "text/event-stream", resp.Header.Get("Content-Type"))
				require.Equal(t, "no-cache", resp.Header.Get("Cache-Control"))
				require.Equal(t, "keep-alive", resp.Header.Get("Connection"))

				// Consume the full response body to ensure the interception completes
				_, err = io.ReadAll(resp.Body)
				require.NoError(t, err)
				resp.Body.Close()

				// Verify the MCP tool was actually invoked
				invocations := mockMCP.getCallsByTool(mockToolName)
				require.Len(t, invocations, 1, "expected MCP tool to be invoked")

				// Verify tool was invoked with the expected args (if specified)
				if tc.expectedArgs != nil {
					expected, err := json.Marshal(tc.expectedArgs)
					require.NoError(t, err)
					actual, err := json.Marshal(invocations[0])
					require.NoError(t, err)
					require.EqualValues(t, expected, actual)
				}

				// Verify tool usage was recorded
				toolUsages := ts.Recorder.RecordedToolUsages()
				require.Len(t, toolUsages, 1)
				assert.Equal(t, mockToolName, toolUsages[0].Tool)

				ts.Recorder.VerifyAllInterceptionsEnded(t)
			})
		}
	})
}

func TestOpenAIInjectedTools(t *testing.T) {
	t.Parallel()

	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming=%v", streaming), func(t *testing.T) {
			t.Parallel()

			// Build the requirements & make the assertions which are common to all providers.
			recorderClient, mockMCP, resp := setupInjectedToolTest(t, fixtures.OaiChatSingleInjectedTool, streaming, defaultTracer, defaultActorID, pathOpenAIChatCompletions, openaiChatToolResultValidator(t))

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
				content *openai.ChatCompletionChoice
				message openai.ChatCompletion
			)
			if streaming {
				// Parse the response stream.
				decoder := oaissestream.NewDecoder(resp)
				stream := oaissestream.NewStream[openai.ChatCompletionChunk](decoder, nil)
				var acc openai.ChatCompletionAccumulator
				detectedToolCalls := make(map[string]struct{})
				for stream.Next() {
					chunk := stream.Current()
					acc.AddChunk(chunk)

					if len(chunk.Choices) == 0 {
						continue
					}

					for _, c := range chunk.Choices {
						if len(c.Delta.ToolCalls) == 0 {
							continue
						}

						for _, t := range c.Delta.ToolCalls {
							if t.Function.Name == "" {
								continue
							}

							detectedToolCalls[t.Function.Name] = struct{}{}
						}
					}
				}

				// Verify that no injected tool call events (or partials thereof) were sent to the client.
				require.Len(t, detectedToolCalls, 0)

				message = acc.ChatCompletion
				require.NoError(t, stream.Err(), "stream error")
			} else {
				// Parse & unmarshal the response.
				body, err := io.ReadAll(resp.Body)
				require.NoError(t, err, "read response body")
				require.NoError(t, json.Unmarshal(body, &message), "unmarshal response")

				// Verify that no injected tools were sent to the client.
				require.GreaterOrEqual(t, len(message.Choices), 1)
				require.Len(t, message.Choices[0].Message.ToolCalls, 0)
			}

			require.GreaterOrEqual(t, len(message.Choices), 1)
			content = &message.Choices[0]

			// Ensure tool returned expected value.
			require.NotNil(t, content)
			require.Contains(t, content.Message.Content, "dd711d5c-83c6-4c08-a0af-b73055906e8c") // The ID of the workspace to be returned.

			// Check the token usage from the client's perspective.
			// This *should* work but the openai SDK doesn't accumulate the prompt token details :(.
			// See https://github.com/openai/openai-go/blob/v2.7.0/streamaccumulator.go#L145-L147.
			// assert.EqualValues(t, 5047, message.Usage.PromptTokens-message.Usage.PromptTokensDetails.CachedTokens)
			assert.EqualValues(t, 105, message.Usage.CompletionTokens)

			// Ensure tokens used during injected tool invocation are accounted for.
			tokenUsages := recorderClient.RecordedTokenUsages()
			require.EqualValues(t, 5047, calculateTotalInputTokens(tokenUsages))
			require.EqualValues(t, 105, calculateTotalOutputTokens(tokenUsages))

			// Ensure we received exactly one prompt.
			promptUsages := recorderClient.RecordedPromptUsages()
			require.Len(t, promptUsages, 1)
		})
	}
}

// openaiChatToolResultValidator returns a request validator that asserts the second
// upstream request contains the assistant's tool_calls and a role=tool result message
// appended by the inner agentic loop.
func openaiChatToolResultValidator(t *testing.T) func(*http.Request, []byte) {
	t.Helper()

	return func(_ *http.Request, raw []byte) {
		messages := gjson.GetBytes(raw, "messages").Array()

		// After the agentic loop the messages must contain at minimum:
		//   [0]   original user message
		//   [N-2] assistant message with tool_calls array
		//   [N-1] message with role=tool
		require.GreaterOrEqual(t, len(messages), 3,
			"second upstream request must contain the original message, assistant tool_calls, and tool result")

		assistantMsg := messages[len(messages)-2]
		require.Equal(t, "assistant", assistantMsg.Get("role").Str,
			"penultimate message must be from the assistant")
		require.NotEmpty(t, len(assistantMsg.Get("tool_calls").Array()),
			"assistant message must contain a tool_calls array")

		toolResultMsg := messages[len(messages)-1]
		require.Equal(t, "tool", toolResultMsg.Get("role").Str,
			"last message must have role=tool")
		require.NotEmpty(t, toolResultMsg.Get("tool_call_id").Str,
			"tool result message must have a tool_call_id")
	}
}
