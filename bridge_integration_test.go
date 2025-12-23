package aibridge_test

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"cdr.dev/slog"
	"cdr.dev/slog/sloggers/slogtest"
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/testutil"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v2"
	oaissestream "github.com/openai/openai-go/v2/packages/ssestream"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"go.opentelemetry.io/otel"
	"go.uber.org/goleak"
)

var (
	//go:embed fixtures/anthropic/simple.txtar
	antSimple []byte
	//go:embed fixtures/anthropic/single_builtin_tool.txtar
	antSingleBuiltinTool []byte
	//go:embed fixtures/anthropic/single_injected_tool.txtar
	antSingleInjectedTool []byte
	//go:embed fixtures/anthropic/fallthrough.txtar
	antFallthrough []byte
	//go:embed fixtures/anthropic/stream_error.txtar
	antMidStreamErr []byte
	//go:embed fixtures/anthropic/non_stream_error.txtar
	antNonStreamErr []byte

	//go:embed fixtures/openai/simple.txtar
	oaiSimple []byte
	//go:embed fixtures/openai/single_builtin_tool.txtar
	oaiSingleBuiltinTool []byte
	//go:embed fixtures/openai/single_injected_tool.txtar
	oaiSingleInjectedTool []byte
	//go:embed fixtures/openai/fallthrough.txtar
	oaiFallthrough []byte
	//go:embed fixtures/openai/stream_error.txtar
	oaiMidStreamErr []byte
	//go:embed fixtures/openai/non_stream_error.txtar
	oaiNonStreamErr []byte

	testTracer = otel.Tracer("forTesting")
)

const (
	apiKey = "api-key"
	userID = "ae235cc1-9f8f-417d-a636-a7b170bac62e"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

func TestAnthropicMessages(t *testing.T) {
	t.Parallel()

	t.Run("single builtin tool", func(t *testing.T) {
		t.Parallel()

		cases := []struct {
			streaming            bool
			expectedInputTokens  int
			expectedOutputTokens int
		}{
			{
				streaming:            true,
				expectedInputTokens:  2,
				expectedOutputTokens: 66,
			},
			{
				streaming:            false,
				expectedInputTokens:  5,
				expectedOutputTokens: 84,
			},
		}

		for _, tc := range cases {
			t.Run(fmt.Sprintf("%s/streaming=%v", t.Name(), tc.streaming), func(t *testing.T) {
				t.Parallel()

				arc := testutil.MustParseTXTAR(t, antSingleBuiltinTool)
				t.Logf("%s: %s", t.Name(), arc.Comment)

				llm := testutil.MustLLMFixture(t, arc)

				ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
				t.Cleanup(cancel)

				upstream := testutil.NewUpstreamServer(t, ctx, llm)

				recorderClient := &testutil.RecorderSpy{}

				logger := slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)
				bridgeSrv := testutil.NewBridgeServer(t, testutil.BridgeConfig{
					Ctx:       ctx,
					ActorID:   userID,
					Providers: []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(upstream.URL, apiKey), nil)},
					Recorder:  recorderClient,
					Logger:    logger,
					Tracer:    testTracer,
				})

				reqBody := llm.MustRequestBody(t, tc.streaming)
				req := bridgeSrv.NewProviderRequest(t, aibridge.ProviderAnthropic, reqBody)
				resp, err := bridgeSrv.Client.Do(req)
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
				tokenUsages := recorderClient.RecordedTokenUsages()
				require.Len(t, tokenUsages, expectedTokenRecordings)

				assert.EqualValues(t, tc.expectedInputTokens, testutil.TotalInputTokens(tokenUsages), "input tokens miscalculated")
				assert.EqualValues(t, tc.expectedOutputTokens, testutil.TotalOutputTokens(tokenUsages), "output tokens miscalculated")

				toolUsages := recorderClient.RecordedToolUsages()
				require.Len(t, toolUsages, 1)
				assert.Equal(t, "Read", toolUsages[0].Tool)
				require.IsType(t, json.RawMessage{}, toolUsages[0].Args)
				var args map[string]any
				require.NoError(t, json.Unmarshal(toolUsages[0].Args.(json.RawMessage), &args))
				require.Contains(t, args, "file_path")
				assert.Equal(t, "/tmp/blah/foo", args["file_path"])

				promptUsages := recorderClient.RecordedPromptUsages()
				require.Len(t, promptUsages, 1)
				assert.Equal(t, "read the foo file", promptUsages[0].Prompt)

				recorderClient.RequireAllInterceptionsEnded(t)
			})
		}
	})
}

func TestAWSBedrockIntegration(t *testing.T) {
	t.Parallel()

	t.Run("invalid config", func(t *testing.T) {
		t.Parallel()

		arc := testutil.MustParseTXTAR(t, antSingleBuiltinTool)
		reqBody := arc.MustFile(t, testutil.FixtureRequest)

		ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
		t.Cleanup(cancel)

		// Invalid bedrock config - missing region
		bedrockCfg := &aibridge.AWSBedrockConfig{
			Region:          "",
			AccessKey:       "test-key",
			AccessKeySecret: "test-secret",
			Model:           "test-model",
			SmallFastModel:  "test-haiku",
		}

		recorderClient := &testutil.RecorderSpy{}
		logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: true}).Leveled(slog.LevelDebug)
		bridgeSrv := testutil.NewBridgeServer(t, testutil.BridgeConfig{
			Ctx:     ctx,
			ActorID: userID,
			Providers: []aibridge.Provider{
				aibridge.NewAnthropicProvider(anthropicCfg("http://unused", apiKey), bedrockCfg),
			},
			Recorder: recorderClient,
			Logger:   logger,
			Tracer:   testTracer,
		})

		req := bridgeSrv.NewProviderRequest(t, aibridge.ProviderAnthropic, reqBody)
		resp, err := bridgeSrv.Client.Do(req)
		require.NoError(t, err)
		defer resp.Body.Close()

		require.Equal(t, http.StatusInternalServerError, resp.StatusCode)
		body, err := io.ReadAll(resp.Body)
		require.NoError(t, err)
		require.Contains(t, string(body), "create anthropic client")
		require.Contains(t, string(body), "region required")
	})

	t.Run("/v1/messages", func(t *testing.T) {
		for _, streaming := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/streaming=%v", t.Name(), streaming), func(t *testing.T) {
				t.Parallel()

				arc := testutil.MustParseTXTAR(t, antSingleBuiltinTool)
				t.Logf("%s: %s", t.Name(), arc.Comment)

				llm := testutil.MustLLMFixture(t, arc)

				ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
				t.Cleanup(cancel)

				upstream := testutil.NewUpstreamServer(t, ctx, llm)

				// Configure Bedrock with test credentials and model names.
				// The EndpointOverride will make requests go to the mock server instead of real AWS endpoints.
				bedrockCfg := &aibridge.AWSBedrockConfig{
					Region:           "us-west-2",
					AccessKey:        "test-access-key",
					AccessKeySecret:  "test-secret-key",
					Model:            "danthropic",      // This model should override the request's given one.
					SmallFastModel:   "danthropic-mini", // Unused but needed for validation.
					EndpointOverride: upstream.URL,
				}

				recorderClient := &testutil.RecorderSpy{}

				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: true}).Leveled(slog.LevelDebug)
				bridgeSrv := testutil.NewBridgeServer(t, testutil.BridgeConfig{
					Ctx:       ctx,
					ActorID:   userID,
					Providers: []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(upstream.URL, apiKey), bedrockCfg)},
					Recorder:  recorderClient,
					Logger:    logger,
					Tracer:    testTracer,
				})

				reqBody := llm.MustRequestBody(t, streaming)
				req := bridgeSrv.NewProviderRequest(t, aibridge.ProviderAnthropic, reqBody)
				resp, err := bridgeSrv.Client.Do(req)
				require.NoError(t, err)
				require.Equal(t, http.StatusOK, resp.StatusCode)
				defer resp.Body.Close()

				// For streaming responses, consume the body to allow the stream to complete.
				if streaming {
					_, err = io.ReadAll(resp.Body)
					require.NoError(t, err)
				}

				reqs := upstream.Requests()
				require.Len(t, reqs, 1)

				// AWS Bedrock encodes the model name in the URL path: /model/{model-id}/invoke or /model/{model-id}/invoke-with-response-stream.
				pathParts := strings.Split(reqs[0].Path, "/")
				receivedModelName := ""
				if len(pathParts) >= 3 && pathParts[1] == "model" {
					receivedModelName = pathParts[2]
				}

				// Verify that Bedrock-specific model name was used in the request to the mock server
				// and the interception data.
				require.Equal(t, bedrockCfg.Model, receivedModelName)
				interceptions := recorderClient.RecordedInterceptions()
				require.Len(t, interceptions, 1)
				require.Equal(t, interceptions[0].Model, bedrockCfg.Model)
				recorderClient.RequireAllInterceptionsEnded(t)
			})
		}
	})
}

func TestOpenAIChatCompletions(t *testing.T) {
	t.Parallel()

	t.Run("single builtin tool", func(t *testing.T) {
		t.Parallel()

		cases := []struct {
			streaming                                 bool
			expectedInputTokens, expectedOutputTokens int
		}{
			{
				streaming:            true,
				expectedInputTokens:  60,
				expectedOutputTokens: 15,
			},
			{
				streaming:            false,
				expectedInputTokens:  60,
				expectedOutputTokens: 15,
			},
		}

		for _, tc := range cases {
			t.Run(fmt.Sprintf("%s/streaming=%v", t.Name(), tc.streaming), func(t *testing.T) {
				t.Parallel()

				arc := testutil.MustParseTXTAR(t, oaiSingleBuiltinTool)
				t.Logf("%s: %s", t.Name(), arc.Comment)

				llm := testutil.MustLLMFixture(t, arc)

				ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
				t.Cleanup(cancel)

				upstream := testutil.NewUpstreamServer(t, ctx, llm)

				recorderClient := &testutil.RecorderSpy{}

				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
				bridgeSrv := testutil.NewBridgeServer(t, testutil.BridgeConfig{
					Ctx:       ctx,
					ActorID:   userID,
					Providers: []aibridge.Provider{aibridge.NewOpenAIProvider(openaiCfg(upstream.URL, apiKey))},
					Recorder:  recorderClient,
					Logger:    logger,
					Tracer:    testTracer,
				})

				reqBody := llm.MustRequestBody(t, tc.streaming)
				req := bridgeSrv.NewProviderRequest(t, aibridge.ProviderOpenAI, reqBody)

				resp, err := bridgeSrv.Client.Do(req)
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

				tokenUsages := recorderClient.RecordedTokenUsages()
				require.Len(t, tokenUsages, 1)
				assert.EqualValues(t, tc.expectedInputTokens, testutil.TotalInputTokens(tokenUsages), "input tokens miscalculated")
				assert.EqualValues(t, tc.expectedOutputTokens, testutil.TotalOutputTokens(tokenUsages), "output tokens miscalculated")

				toolUsages := recorderClient.RecordedToolUsages()
				require.Len(t, toolUsages, 1)
				assert.Equal(t, "read_file", toolUsages[0].Tool)
				require.IsType(t, map[string]any{}, toolUsages[0].Args)
				require.Contains(t, toolUsages[0].Args, "path")
				assert.Equal(t, "README.md", toolUsages[0].Args.(map[string]any)["path"])

				promptUsages := recorderClient.RecordedPromptUsages()
				require.Len(t, promptUsages, 1)
				assert.Equal(t, "how large is the README.md file in my current path", promptUsages[0].Prompt)

				recorderClient.RequireAllInterceptionsEnded(t)
			})
		}
	})
}

func TestSimple(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name              string
		fixture           []byte
		configureFunc     func(string, aibridge.Recorder) (*aibridge.RequestBridge, error)
		getResponseIDFunc func(bool, *http.Response) (string, error)
		expectedMsgID     string
	}{
		{
			name:    aibridge.ProviderAnthropic,
			fixture: antSimple,
			configureFunc: func(addr string, client aibridge.Recorder) (*aibridge.RequestBridge, error) {
				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
				provider := []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(addr, apiKey), nil)}
				return aibridge.NewRequestBridge(t.Context(), provider, client, mcp.NewServerProxyManager(nil, testTracer), logger, nil, testTracer)
			},
			getResponseIDFunc: func(streaming bool, resp *http.Response) (string, error) {
				if streaming {
					decoder := ssestream.NewDecoder(resp)
					stream := ssestream.NewStream[anthropic.MessageStreamEventUnion](decoder, nil)
					var message anthropic.Message
					for stream.Next() {
						event := stream.Current()
						if err := message.Accumulate(event); err != nil {
							return "", fmt.Errorf("accumulate event: %w", err)
						}
					}
					if stream.Err() != nil {
						return "", fmt.Errorf("stream error: %w", stream.Err())
					}
					return message.ID, nil
				}

				body, err := io.ReadAll(resp.Body)
				if err != nil {
					return "", fmt.Errorf("read body: %w", err)
				}

				var message anthropic.Message
				if err := json.Unmarshal(body, &message); err != nil {
					return "", fmt.Errorf("unmarshal response: %w", err)
				}
				return message.ID, nil
			},
			expectedMsgID: "msg_01Pvyf26bY17RcjmWfJsXGBn",
		},
		{
			name:    aibridge.ProviderOpenAI,
			fixture: oaiSimple,
			configureFunc: func(addr string, client aibridge.Recorder) (*aibridge.RequestBridge, error) {
				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
				providers := []aibridge.Provider{aibridge.NewOpenAIProvider(openaiCfg(addr, apiKey))}
				return aibridge.NewRequestBridge(t.Context(), providers, client, mcp.NewServerProxyManager(nil, testTracer), logger, nil, testTracer)
			},
			getResponseIDFunc: func(streaming bool, resp *http.Response) (string, error) {
				if streaming {
					// Parse the response stream.
					decoder := oaissestream.NewDecoder(resp)
					stream := oaissestream.NewStream[openai.ChatCompletionChunk](decoder, nil)
					var message openai.ChatCompletionAccumulator
					for stream.Next() {
						chunk := stream.Current()
						message.AddChunk(chunk)
					}
					if stream.Err() != nil {
						return "", fmt.Errorf("stream error: %w", stream.Err())
					}
					return message.ID, nil
				}

				// Parse & unmarshal the response.
				body, err := io.ReadAll(resp.Body)
				if err != nil {
					return "", fmt.Errorf("read body: %w", err)
				}

				var message openai.ChatCompletion
				if err := json.Unmarshal(body, &message); err != nil {
					return "", fmt.Errorf("unmarshal response: %w", err)
				}
				return message.ID, nil
			},
			expectedMsgID: "chatcmpl-BwoiPTGRbKkY5rncfaM0s9KtWrq5N",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			for _, streaming := range []bool{true, false} {
				t.Run(fmt.Sprintf("streaming=%v", streaming), func(t *testing.T) {
					t.Parallel()

					arc := testutil.MustParseTXTAR(t, tc.fixture)
					t.Logf("%s: %s", t.Name(), arc.Comment)

					llm := testutil.MustLLMFixture(t, arc)

					ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
					t.Cleanup(cancel)

					upstream := testutil.NewUpstreamServer(t, ctx, llm)

					recorderClient := &testutil.RecorderSpy{}

					bridge, err := tc.configureFunc(upstream.URL, recorderClient)
					require.NoError(t, err)

					bridgeSrv := testutil.NewBridgeServer(t, testutil.BridgeConfig{
						Ctx:     ctx,
						ActorID: userID,
						Handler: bridge,
					})

					reqBody := llm.MustRequestBody(t, streaming)
					req := bridgeSrv.NewProviderRequest(t, tc.name, reqBody)
					resp, err := bridgeSrv.Client.Do(req)
					require.NoError(t, err)
					require.Equal(t, http.StatusOK, resp.StatusCode)
					defer resp.Body.Close()

					// Then: I expect a non-empty response.
					bodyBytes, err := io.ReadAll(resp.Body)
					require.NoError(t, err)
					assert.NotEmpty(t, bodyBytes, "should have received response body")

					// Reset the body after being read.
					resp.Body = io.NopCloser(bytes.NewReader(bodyBytes))

					// Then: I expect the prompt to have been tracked.
					promptUsages := recorderClient.RecordedPromptUsages()
					require.NotEmpty(t, promptUsages, "no prompts tracked")
					assert.Contains(t, promptUsages[0].Prompt, "how many angels can dance on the head of a pin")

					// Validate that responses have their IDs overridden with a interception ID rather than the original ID from the upstream provider.
					// The reason for this is that Bridge may make multiple upstream requests (i.e. to invoke injected tools), and clients will not be expecting
					// multiple messages in response to a single request.
					id, err := tc.getResponseIDFunc(streaming, resp)
					require.NoError(t, err, "failed to retrieve response ID")
					require.Nilf(t, uuid.Validate(id), "%s is not a valid UUID", id)

					tokenUsages := recorderClient.RecordedTokenUsages()
					require.GreaterOrEqual(t, len(tokenUsages), 1)
					require.Equal(t, tokenUsages[0].MsgID, tc.expectedMsgID)

					recorderClient.RequireAllInterceptionsEnded(t)
				})
			}
		})
	}
}

func TestFallthrough(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name          string
		fixture       []byte
		configureFunc func(string, aibridge.Recorder) (aibridge.Provider, *aibridge.RequestBridge)
	}{
		{
			name:    aibridge.ProviderAnthropic,
			fixture: antFallthrough,
			configureFunc: func(addr string, client aibridge.Recorder) (aibridge.Provider, *aibridge.RequestBridge) {
				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
				provider := aibridge.NewAnthropicProvider(anthropicCfg(addr, apiKey), nil)
				bridge, err := aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{provider}, client, mcp.NewServerProxyManager(nil, testTracer), logger, nil, testTracer)
				require.NoError(t, err)
				return provider, bridge
			},
		},
		{
			name:    aibridge.ProviderOpenAI,
			fixture: oaiFallthrough,
			configureFunc: func(addr string, client aibridge.Recorder) (aibridge.Provider, *aibridge.RequestBridge) {
				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
				provider := aibridge.NewOpenAIProvider(openaiCfg(addr, apiKey))
				bridge, err := aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{provider}, client, mcp.NewServerProxyManager(nil, testTracer), logger, nil, testTracer)
				require.NoError(t, err)
				return provider, bridge
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			arc := testutil.MustParseTXTAR(t, tc.fixture)
			t.Logf("%s: %s", t.Name(), arc.Comment)

			var receivedHeaders *http.Header
			respBody := arc.MustFile(t, testutil.FixtureResponse)
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/models" {
					t.Errorf("unexpected request path: %q", r.URL.Path)
					t.FailNow()
				}

				receivedHeaders = &r.Header

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(respBody)
			}))
			t.Cleanup(upstream.Close)

			recorderClient := &testutil.RecorderSpy{}

			provider, bridge := tc.configureFunc(upstream.URL, recorderClient)

			bridgeSrv := testutil.NewBridgeServer(t, testutil.BridgeConfig{
				Ctx:     t.Context(),
				ActorID: userID,
				Handler: bridge,
			})

			req, err := http.NewRequestWithContext(t.Context(), "GET", fmt.Sprintf("%s/%s/v1/models", bridgeSrv.URL, tc.name), nil)
			require.NoError(t, err)

			resp, err := bridgeSrv.Client.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()

			require.Equal(t, http.StatusOK, resp.StatusCode)

			// Ensure that the API key was sent.
			require.NotNil(t, receivedHeaders)
			require.Contains(t, receivedHeaders.Get(provider.AuthHeader()), apiKey)

			gotBytes, err := io.ReadAll(resp.Body)
			require.NoError(t, err)

			// Compare JSON bodies for semantic equality.
			var got any
			var exp any
			require.NoError(t, json.Unmarshal(gotBytes, &got))
			require.NoError(t, json.Unmarshal(respBody, &exp))
			require.EqualValues(t, exp, got)
		})
	}
}

type (
	configureFunc func(string, aibridge.Recorder, *mcp.ServerProxyManager) (*aibridge.RequestBridge, error)
)

func TestErrorHandling(t *testing.T) {
	t.Parallel()

	// Tests that errors which occur *before* a streaming response begins, or in non-streaming requests, are handled as expected.
	t.Run("non-stream error", func(t *testing.T) {
		cases := []struct {
			name              string
			fixture           []byte
			configureFunc     configureFunc
			responseHandlerFn func(resp *http.Response)
		}{
			{
				name:    aibridge.ProviderAnthropic,
				fixture: antNonStreamErr,
				configureFunc: func(addr string, client aibridge.Recorder, srvProxyMgr *mcp.ServerProxyManager) (*aibridge.RequestBridge, error) {
					logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
					providers := []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(addr, apiKey), nil)}
					return aibridge.NewRequestBridge(t.Context(), providers, client, srvProxyMgr, logger, nil, testTracer)
				},
				responseHandlerFn: func(resp *http.Response) {
					require.Equal(t, http.StatusBadRequest, resp.StatusCode)
					body, err := io.ReadAll(resp.Body)
					require.NoError(t, err)
					require.Equal(t, "error", gjson.GetBytes(body, "type").Str)
					require.Equal(t, "invalid_request_error", gjson.GetBytes(body, "error.type").Str)
					require.Contains(t, gjson.GetBytes(body, "error.message").Str, "prompt is too long")
				},
			},
			{
				name:    aibridge.ProviderOpenAI,
				fixture: oaiNonStreamErr,
				configureFunc: func(addr string, client aibridge.Recorder, srvProxyMgr *mcp.ServerProxyManager) (*aibridge.RequestBridge, error) {
					logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
					providers := []aibridge.Provider{aibridge.NewOpenAIProvider(openaiCfg(addr, apiKey))}
					return aibridge.NewRequestBridge(t.Context(), providers, client, srvProxyMgr, logger, nil, testTracer)
				},
				responseHandlerFn: func(resp *http.Response) {
					require.Equal(t, http.StatusBadRequest, resp.StatusCode)
					body, err := io.ReadAll(resp.Body)
					require.NoError(t, err)
					require.Equal(t, "context_length_exceeded", gjson.GetBytes(body, "error.code").Str)
					require.Equal(t, "invalid_request_error", gjson.GetBytes(body, "error.type").Str)
					require.Contains(t, gjson.GetBytes(body, "error.message").Str, "Input tokens exceed the configured limit")
				},
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				for _, streaming := range []bool{true, false} {
					t.Run(fmt.Sprintf("streaming=%v", streaming), func(t *testing.T) {
						t.Parallel()

						ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
						t.Cleanup(cancel)

						arc := testutil.MustParseTXTAR(t, tc.fixture)
						t.Logf("%s: %s", t.Name(), arc.Comment)

						llm := testutil.MustLLMFixture(t, arc)
						reqBody := llm.MustRequestBody(t, streaming)

						mockResp, err := llm.Response(1, streaming)
						require.NoError(t, err)
						mockSrv := testutil.NewHTTPReflectorServer(t, ctx, mockResp)

						recorderClient := &testutil.RecorderSpy{}

						bridge, err := tc.configureFunc(mockSrv.URL, recorderClient, mcp.NewServerProxyManager(nil, testTracer))
						require.NoError(t, err)

						bridgeSrv := testutil.NewBridgeServer(t, testutil.BridgeConfig{
							Ctx:     ctx,
							ActorID: userID,
							Handler: bridge,
						})

						req := bridgeSrv.NewProviderRequest(t, tc.name, reqBody)
						resp, err := bridgeSrv.Client.Do(req)
						require.NoError(t, err)
						defer resp.Body.Close()

						tc.responseHandlerFn(resp)
						recorderClient.RequireAllInterceptionsEnded(t)
					})
				}
			})
		}
	})

	// Tests that errors which occur *during* a streaming response are handled as expected.
	t.Run("mid-stream error", func(t *testing.T) {
		cases := []struct {
			name              string
			fixture           []byte
			configureFunc     configureFunc
			responseHandlerFn func(resp *http.Response)
		}{
			{
				name:    aibridge.ProviderAnthropic,
				fixture: antMidStreamErr,
				configureFunc: func(addr string, client aibridge.Recorder, srvProxyMgr *mcp.ServerProxyManager) (*aibridge.RequestBridge, error) {
					logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
					providers := []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(addr, apiKey), nil)}
					return aibridge.NewRequestBridge(t.Context(), providers, client, srvProxyMgr, logger, nil, testTracer)
				},
				responseHandlerFn: func(resp *http.Response) {
					// Server responds first with 200 OK then starts streaming.
					require.Equal(t, http.StatusOK, resp.StatusCode)

					sp := aibridge.NewSSEParser()
					require.NoError(t, sp.Parse(resp.Body))
					require.Len(t, sp.EventsByType("error"), 1)
					require.Contains(t, sp.EventsByType("error")[0].Data, "Overloaded")
				},
			},
			{
				name:    aibridge.ProviderOpenAI,
				fixture: oaiMidStreamErr,
				configureFunc: func(addr string, client aibridge.Recorder, srvProxyMgr *mcp.ServerProxyManager) (*aibridge.RequestBridge, error) {
					logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
					providers := []aibridge.Provider{aibridge.NewOpenAIProvider(openaiCfg(addr, apiKey))}
					return aibridge.NewRequestBridge(t.Context(), providers, client, srvProxyMgr, logger, nil, testTracer)
				},
				responseHandlerFn: func(resp *http.Response) {
					// Server responds first with 200 OK then starts streaming.
					require.Equal(t, http.StatusOK, resp.StatusCode)

					sp := aibridge.NewSSEParser()
					require.NoError(t, sp.Parse(resp.Body))
					// OpenAI sends all events under the same type.
					messageEvents := sp.MessageEvents()
					require.NotEmpty(t, messageEvents)

					errEvent := sp.MessageEvents()[len(sp.MessageEvents())-2] // Last event is termination marker ("[DONE]").
					require.NotEmpty(t, errEvent)
					require.Contains(t, errEvent.Data, "The server had an error while processing your request. Sorry about that!")
				},
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
				t.Cleanup(cancel)

				arc := testutil.MustParseTXTAR(t, tc.fixture)
				t.Logf("%s: %s", t.Name(), arc.Comment)

				llm := testutil.MustLLMFixture(t, arc)

				upstream := testutil.NewUpstreamServer(t, ctx, llm)

				reqBody := llm.MustRequestBody(t, true)

				recorderClient := &testutil.RecorderSpy{}

				bridge, err := tc.configureFunc(upstream.URL, recorderClient, mcp.NewServerProxyManager(nil, testTracer))
				require.NoError(t, err)

				bridgeSrv := testutil.NewBridgeServer(t, testutil.BridgeConfig{
					Ctx:     ctx,
					ActorID: userID,
					Handler: bridge,
				})

				req := bridgeSrv.NewProviderRequest(t, tc.name, reqBody)
				resp, err := bridgeSrv.Client.Do(req)
				require.NoError(t, err)
				defer resp.Body.Close()

				tc.responseHandlerFn(resp)
				recorderClient.RequireAllInterceptionsEnded(t)
			})
		}
	})
}

// TestStableRequestEncoding validates that a given intercepted request and a
// given set of injected tools should result identical payloads.
//
// Should the payload vary, it may subvert any caching mechanisms the provider may have.
func TestStableRequestEncoding(t *testing.T) {
	t.Parallel()

	logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)

	cases := []struct {
		name        string
		fixture     []byte
		newProvider func(upstreamURL string) aibridge.Provider
	}{
		{
			name:    aibridge.ProviderAnthropic,
			fixture: antSimple,
			newProvider: func(upstreamURL string) aibridge.Provider {
				return aibridge.NewAnthropicProvider(anthropicCfg(upstreamURL, apiKey), nil)
			},
		},
		{
			name:    aibridge.ProviderOpenAI,
			fixture: oaiSimple,
			newProvider: func(upstreamURL string) aibridge.Provider {
				return aibridge.NewOpenAIProvider(openaiCfg(upstreamURL, apiKey))
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			t.Cleanup(cancel)

			arc := testutil.MustParseTXTAR(t, tc.fixture)
			t.Logf("%s: %s", t.Name(), arc.Comment)

			llm := testutil.MustLLMFixture(t, arc)
			reqBody := arc.MustFile(t, testutil.FixtureRequest)

			upstream := testutil.NewUpstreamServer(t, ctx, llm)

			// Setup mocked MCP server & tools.
			mcpSrv := testutil.NewMCPServer(t, testutil.DefaultCoderToolNames())
			mcpProxiers := mcpSrv.Proxiers(t, "coder", logger, testTracer)

			recorder := &testutil.RecorderSpy{}
			bridgeSrv := testutil.NewBridgeServer(t, testutil.BridgeConfig{
				Ctx:         ctx,
				ActorID:     userID,
				Providers:   []aibridge.Provider{tc.newProvider(upstream.URL)},
				Recorder:    recorder,
				MCPProxiers: mcpProxiers,
				Logger:      logger,
				Tracer:      testTracer,
			})

			// Make multiple requests and verify they all have identical payloads.
			count := 10
			for i := 0; i < count; i++ {
				req := bridgeSrv.NewProviderRequest(t, tc.name, reqBody)
				resp, err := bridgeSrv.Client.Do(req)
				require.NoError(t, err)
				require.Equal(t, http.StatusOK, resp.StatusCode)
				_ = resp.Body.Close()
			}

			upstream.RequireCallCountEventually(t, count)
			reqs := upstream.Requests()
			require.Len(t, reqs, count)

			reference := string(reqs[0].Body)
			for i := 1; i < len(reqs); i++ {
				assert.JSONEq(t, reference, string(reqs[i].Body))
			}

			recorder.RequireAllInterceptionsEnded(t)
		})
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
		expectDisableParallel         bool
		expectToolChoiceTypeInRequest string
	}{
		{
			name:                          "no tool_choice defined defaults to auto",
			toolChoice:                    nil,
			expectDisableParallel:         true,
			expectToolChoiceTypeInRequest: toolChoiceAuto,
		},
		{
			name:                          "tool_choice auto",
			toolChoice:                    map[string]any{"type": toolChoiceAuto},
			expectDisableParallel:         true,
			expectToolChoiceTypeInRequest: toolChoiceAuto,
		},
		{
			name:                          "tool_choice any",
			toolChoice:                    map[string]any{"type": toolChoiceAny},
			expectDisableParallel:         true,
			expectToolChoiceTypeInRequest: toolChoiceAny,
		},
		{
			name:                          "tool_choice tool",
			toolChoice:                    map[string]any{"type": toolChoiceTool, "name": "some_tool"},
			expectDisableParallel:         true,
			expectToolChoiceTypeInRequest: toolChoiceTool,
		},
		{
			name:                          "tool_choice none",
			toolChoice:                    map[string]any{"type": toolChoiceNone},
			expectDisableParallel:         false,
			expectToolChoiceTypeInRequest: toolChoiceNone,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			t.Cleanup(cancel)

			arc := testutil.MustParseTXTAR(t, antSimple)
			llm := testutil.MustLLMFixture(t, arc)

			// Prepare request body with tool_choice set.
			var reqJSON map[string]any
			require.NoError(t, json.Unmarshal(arc.MustFile(t, testutil.FixtureRequest), &reqJSON))
			if tc.toolChoice != nil {
				reqJSON["tool_choice"] = tc.toolChoice
			}
			reqBody, err := json.Marshal(reqJSON)
			require.NoError(t, err)

			upstream := testutil.NewUpstreamServer(t, ctx, llm)

			recorder := &testutil.RecorderSpy{}
			logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
			providers := []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(upstream.URL, apiKey), nil)}
			bridgeSrv := testutil.NewBridgeServer(t, testutil.BridgeConfig{
				Ctx:       ctx,
				ActorID:   userID,
				Providers: providers,
				Recorder:  recorder,
				Logger:    logger,
				Tracer:    testTracer,
			})

			req := bridgeSrv.NewProviderRequest(t, aibridge.ProviderAnthropic, reqBody)
			resp, err := bridgeSrv.Client.Do(req)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, resp.StatusCode)
			_ = resp.Body.Close()

			upstream.RequireCallCountEventually(t, 1)
			reqs := upstream.Requests()
			require.Len(t, reqs, 1)

			var receivedRequest map[string]any
			require.NoError(t, json.Unmarshal(reqs[0].Body, &receivedRequest))

			// Verify tool_choice in the upstream request.
			require.NotNil(t, receivedRequest)
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

func TestEnvironmentDoNotLeak(t *testing.T) {
	// NOTE: Cannot use t.Parallel() here because subtests use t.Setenv which requires sequential execution.

	// Test that environment variables containing API keys/tokens are not leaked to upstream requests.
	// See https://github.com/coder/aibridge/issues/60.
	testCases := []struct {
		name          string
		fixture       []byte
		configureFunc func(string, aibridge.Recorder) (*aibridge.RequestBridge, error)
		envVars       map[string]string
		headerName    string
	}{
		{
			name:    aibridge.ProviderAnthropic,
			fixture: antSimple,
			configureFunc: func(addr string, client aibridge.Recorder) (*aibridge.RequestBridge, error) {
				logger := slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)
				providers := []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(addr, apiKey), nil)}
				return aibridge.NewRequestBridge(t.Context(), providers, client, mcp.NewServerProxyManager(nil, testTracer), logger, nil, testTracer)
			},
			envVars: map[string]string{
				"ANTHROPIC_AUTH_TOKEN": "should-not-leak",
			},
			headerName: "Authorization", // We only send through the X-Api-Key, so this one should not be present.
		},
		{
			name:    aibridge.ProviderOpenAI,
			fixture: oaiSimple,
			configureFunc: func(addr string, client aibridge.Recorder) (*aibridge.RequestBridge, error) {
				logger := slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)
				providers := []aibridge.Provider{aibridge.NewOpenAIProvider(openaiCfg(addr, apiKey))}
				return aibridge.NewRequestBridge(t.Context(), providers, client, mcp.NewServerProxyManager(nil, testTracer), logger, nil, testTracer)
			},
			envVars: map[string]string{
				"OPENAI_ORG_ID": "should-not-leak",
			},
			headerName: "OpenAI-Organization",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// NOTE: Cannot use t.Parallel() here because t.Setenv requires sequential execution.

			arc := testutil.MustParseTXTAR(t, tc.fixture)
			llm := testutil.MustLLMFixture(t, arc)
			reqBody := arc.MustFile(t, testutil.FixtureRequest)

			ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
			t.Cleanup(cancel)

			upstream := testutil.NewUpstreamServer(t, ctx, llm)

			// Set environment variables that the SDK would automatically read.
			// These should NOT leak into upstream requests.
			for key, val := range tc.envVars {
				t.Setenv(key, val)
			}

			recorderClient := &testutil.RecorderSpy{}
			bridge, err := tc.configureFunc(upstream.URL, recorderClient)
			require.NoError(t, err)

			bridgeSrv := testutil.NewBridgeServer(t, testutil.BridgeConfig{
				Ctx:     ctx,
				ActorID: userID,
				Handler: bridge,
			})

			req := bridgeSrv.NewProviderRequest(t, tc.name, reqBody)
			resp, err := bridgeSrv.Client.Do(req)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, resp.StatusCode)
			defer resp.Body.Close()

			upstream.RequireCallCountEventually(t, 1)
			reqs := upstream.Requests()
			require.Len(t, reqs, 1)

			// Verify that environment values did not leak.
			receivedHeaders := reqs[0].Header
			require.NotNil(t, receivedHeaders)
			require.Empty(t, receivedHeaders.Get(tc.headerName))
		})
	}
}

func openaiCfg(url, key string) aibridge.OpenAIConfig {
	return aibridge.OpenAIConfig{
		BaseURL: url,
		Key:     key,
	}
}

func anthropicCfg(url, key string) aibridge.AnthropicConfig {
	return aibridge.AnthropicConfig{
		BaseURL: url,
		Key:     key,
	}
}
