package aibridge_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"
	"time"

	"cdr.dev/slog"
	"cdr.dev/slog/sloggers/slogtest"
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/testutil"
	"github.com/openai/openai-go/v2"
	oaissestream "github.com/openai/openai-go/v2/packages/ssestream"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace"
)

type injectedToolHarness struct {
	Recorder    *testutil.RecorderSpy
	MCP         *testutil.MCPServer
	MCPProxiers map[string]mcp.ServerProxier
	Upstream    *testutil.UpstreamServer
	Bridge      *testutil.BridgeServer
	Inspector   *testutil.Inspector
	Response    *http.Response

	RequestBody []byte
	RequestPath string
}

func runInjectedToolTest(t *testing.T, providerName string, fixture []byte, streaming bool, tracer trace.Tracer, makeProviders func(upstreamURL string) []aibridge.Provider) injectedToolHarness {
	t.Helper()

	if tracer == nil {
		tracer = testTracer
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	t.Cleanup(cancel)

	logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)

	// Fixture-driven upstream.
	arc := testutil.MustParseTXTAR(t, fixture)
	llm := testutil.MustLLMFixture(t, arc)
	upstream := testutil.NewUpstreamServer(t, ctx, llm)

	// MCP server + proxies.
	mcpSrv := testutil.NewMCPServer(t, testutil.DefaultCoderToolNames())
	mcpProxiers := mcpSrv.Proxiers(t, "coder", logger, tracer)

	recorder := &testutil.RecorderSpy{}
	bridge := testutil.NewBridgeServer(t, testutil.BridgeConfig{
		Ctx:         ctx,
		ActorID:     userID,
		Providers:   makeProviders(upstream.URL),
		Recorder:    recorder,
		MCPProxiers: mcpProxiers,
		Logger:      logger,
		Tracer:      tracer,
	})

	reqBody := llm.MustRequestBody(t, streaming)
	req := bridge.NewProviderRequest(t, providerName, reqBody)

	resp, err := bridge.Client.Do(req)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode)
	t.Cleanup(func() { _ = resp.Body.Close() })

	// Injected tool tests must always produce exactly 2 upstream calls.
	upstream.RequireCallCountEventually(t, 2)

	inspector := testutil.NewInspector(recorder, mcpSrv, upstream)

	return injectedToolHarness{
		Recorder:    recorder,
		MCP:         mcpSrv,
		MCPProxiers: mcpProxiers,
		Upstream:    upstream,
		Bridge:      bridge,
		Inspector:   inspector,
		Response:    resp,

		RequestBody: reqBody,
		RequestPath: req.URL.Path,
	}
}

func TestAnthropicInjectedTools(t *testing.T) {
	t.Parallel()

	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming=%v", streaming), func(t *testing.T) {
			t.Parallel()

			h := runInjectedToolTest(t, aibridge.ProviderAnthropic, antSingleInjectedTool, streaming, testTracer, func(upstreamURL string) []aibridge.Provider {
				return []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(upstreamURL, apiKey), nil)}
			})
			resp := h.Response

			// Ensure expected tool was invoked with expected input.
			h.Inspector.RequireToolCalledOnceWithArgs(t, testutil.ToolCoderListWorkspaces, map[string]any{"owner": "admin"})

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
			tokenUsages := h.Recorder.RecordedTokenUsages()
			assert.EqualValues(t, 15308, testutil.TotalInputTokens(tokenUsages))
			assert.EqualValues(t, 204, testutil.TotalOutputTokens(tokenUsages))

			// Ensure we received exactly one prompt.
			promptUsages := h.Recorder.RecordedPromptUsages()
			require.Len(t, promptUsages, 1)
		})
	}
}

func TestOpenAIInjectedTools(t *testing.T) {
	t.Parallel()

	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming=%v", streaming), func(t *testing.T) {
			t.Parallel()

			h := runInjectedToolTest(t, aibridge.ProviderOpenAI, oaiSingleInjectedTool, streaming, testTracer, func(upstreamURL string) []aibridge.Provider {
				return []aibridge.Provider{aibridge.NewOpenAIProvider(openaiCfg(upstreamURL, apiKey))}
			})
			resp := h.Response

			// Ensure expected tool was invoked with expected input.
			h.Inspector.RequireToolCalledOnceWithArgs(t, testutil.ToolCoderListWorkspaces, map[string]any{"owner": "admin"})

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
			tokenUsages := h.Recorder.RecordedTokenUsages()
			require.EqualValues(t, 5047, testutil.TotalInputTokens(tokenUsages))
			require.EqualValues(t, 105, testutil.TotalOutputTokens(tokenUsages))

			// Ensure we received exactly one prompt.
			promptUsages := h.Recorder.RecordedPromptUsages()
			require.Len(t, promptUsages, 1)
		})
	}
}
