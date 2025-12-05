package aibridge_test

import (
	"bufio"
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"cdr.dev/slog"
	"cdr.dev/slog/sloggers/slogtest"
	"go.uber.org/goleak"
	"golang.org/x/tools/txtar"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/mcp"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"

	"github.com/openai/openai-go/v2"
	oaissestream "github.com/openai/openai-go/v2/packages/ssestream"

	mcplib "github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
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
)

const (
	fixtureRequest                  = "request"
	fixtureStreamingResponse        = "streaming"
	fixtureNonStreamingResponse     = "non-streaming"
	fixtureStreamingToolResponse    = "streaming/tool-call"
	fixtureNonStreamingToolResponse = "non-streaming/tool-call"
	fixtureResponse                 = "response"

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
			streaming                                 bool
			expectedInputTokens, expectedOutputTokens int
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

				arc := txtar.Parse(antSingleBuiltinTool)
				t.Logf("%s: %s", t.Name(), arc.Comment)

				files := filesMap(arc)
				require.Len(t, files, 3)
				require.Contains(t, files, fixtureRequest)
				require.Contains(t, files, fixtureStreamingResponse)
				require.Contains(t, files, fixtureNonStreamingResponse)

				reqBody := files[fixtureRequest]

				// Add the stream param to the request.
				newBody, err := setJSON(reqBody, "stream", tc.streaming)
				require.NoError(t, err)
				reqBody = newBody

				ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
				t.Cleanup(cancel)
				srv := newMockServer(ctx, t, files, nil)
				t.Cleanup(srv.Close)

				recorderClient := &mockRecorderClient{}

				logger := slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)
				b, err := aibridge.NewRequestBridge(ctx, []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(srv.URL, apiKey), nil)}, recorderClient, mcp.NewServerProxyManager(nil), nil, logger)
				require.NoError(t, err)

				mockSrv := httptest.NewUnstartedServer(b)
				t.Cleanup(mockSrv.Close)
				mockSrv.Config.BaseContext = func(_ net.Listener) context.Context {
					return aibridge.AsActor(ctx, userID, nil)
				}
				mockSrv.Start()

				// Make API call to aibridge for Anthropic /v1/messages
				req := createAnthropicMessagesReq(t, mockSrv.URL, reqBody)
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
				require.Len(t, recorderClient.tokenUsages, expectedTokenRecordings)

				assert.EqualValues(t, tc.expectedInputTokens, calculateTotalInputTokens(recorderClient.tokenUsages), "input tokens miscalculated")
				assert.EqualValues(t, tc.expectedOutputTokens, calculateTotalOutputTokens(recorderClient.tokenUsages), "output tokens miscalculated")

				require.Len(t, recorderClient.toolUsages, 1)
				assert.Equal(t, "Read", recorderClient.toolUsages[0].Tool)
				require.IsType(t, json.RawMessage{}, recorderClient.toolUsages[0].Args)
				var args map[string]any
				require.NoError(t, json.Unmarshal(recorderClient.toolUsages[0].Args.(json.RawMessage), &args))
				require.Contains(t, args, "file_path")
				assert.Equal(t, "/tmp/blah/foo", args["file_path"])

				require.Len(t, recorderClient.userPrompts, 1)
				assert.Equal(t, "read the foo file", recorderClient.userPrompts[0].Prompt)

				recorderClient.verifyAllInterceptionsEnded(t)
			})
		}
	})
}

func TestAWSBedrockIntegration(t *testing.T) {
	t.Parallel()

	t.Run("invalid config", func(t *testing.T) {
		t.Parallel()

		arc := txtar.Parse(antSingleBuiltinTool)
		files := filesMap(arc)
		reqBody := files[fixtureRequest]

		ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
		t.Cleanup(cancel)

		// Invalid bedrock config - missing region
		bedrockCfg := &aibridge.AWSBedrockConfig{
			Region:          "",
			AccessKey:       "test-key",
			AccessKeySecret: "test-secret",
			Model:           "test-model",
			SmallFastModel:  "test-haiku",
		}

		recorderClient := &mockRecorderClient{}
		logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: true}).Leveled(slog.LevelDebug)
		b, err := aibridge.NewRequestBridge(ctx, []aibridge.Provider{
			aibridge.NewAnthropicProvider(anthropicCfg("http://unused", apiKey), bedrockCfg),
		}, recorderClient, mcp.NewServerProxyManager(nil), nil, logger)
		require.NoError(t, err)

		mockSrv := httptest.NewUnstartedServer(b)
		t.Cleanup(mockSrv.Close)
		mockSrv.Config.BaseContext = func(_ net.Listener) context.Context {
			return aibridge.AsActor(ctx, userID, nil)
		}
		mockSrv.Start()

		req := createAnthropicMessagesReq(t, mockSrv.URL, reqBody)
		resp, err := http.DefaultClient.Do(req)
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

				arc := txtar.Parse(antSingleBuiltinTool)
				t.Logf("%s: %s", t.Name(), arc.Comment)

				files := filesMap(arc)
				require.Len(t, files, 3)
				require.Contains(t, files, fixtureRequest)

				reqBody := files[fixtureRequest]

				newBody, err := setJSON(reqBody, "stream", streaming)
				require.NoError(t, err)
				reqBody = newBody

				ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
				t.Cleanup(cancel)

				var receivedModelName string
				var requestCount int

				// Create a mock server that intercepts requests to capture model name and return fixtures.
				srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					requestCount++
					t.Logf("Mock server received request #%d: %s %s (streaming=%v)", requestCount, r.Method, r.URL.Path, streaming)
					t.Logf("Request headers: %v", r.Header)

					// AWS Bedrock encodes the model name in the URL path: /model/{model-id}/invoke or /model/{model-id}/invoke-with-response-stream.
					// Extract the model name from the path.
					pathParts := strings.Split(r.URL.Path, "/")
					if len(pathParts) >= 3 && pathParts[1] == "model" {
						receivedModelName = pathParts[2]
						t.Logf("Extracted model name from path: %s", receivedModelName)
					}

					// Return appropriate fixture response.
					var respBody []byte
					if streaming {
						respBody = files[fixtureStreamingResponse]
						w.Header().Set("Content-Type", "text/event-stream")
						w.Header().Set("Cache-Control", "no-cache")
						w.Header().Set("Connection", "keep-alive")
					} else {
						respBody = files[fixtureNonStreamingResponse]
						w.Header().Set("Content-Type", "application/json")
					}

					w.WriteHeader(http.StatusOK)
					_, _ = w.Write(respBody)
				}))

				srv.Config.BaseContext = func(_ net.Listener) context.Context {
					return ctx
				}
				srv.Start()
				t.Cleanup(srv.Close)

				// Configure Bedrock with test credentials and model names.
				// The EndpointOverride will make requests go to the mock server instead of real AWS endpoints.
				bedrockCfg := &aibridge.AWSBedrockConfig{
					Region:           "us-west-2",
					AccessKey:        "test-access-key",
					AccessKeySecret:  "test-secret-key",
					Model:            "danthropic",      // This model should override the request's given one.
					SmallFastModel:   "danthropic-mini", // Unused but needed for validation.
					EndpointOverride: srv.URL,
				}

				recorderClient := &mockRecorderClient{}

				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: true}).Leveled(slog.LevelDebug)
				b, err := aibridge.NewRequestBridge(
					ctx, []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(srv.URL, apiKey), bedrockCfg)},
					recorderClient, mcp.NewServerProxyManager(nil), nil, logger)
				require.NoError(t, err)

				mockBridgeSrv := httptest.NewUnstartedServer(b)
				t.Cleanup(mockBridgeSrv.Close)
				mockBridgeSrv.Config.BaseContext = func(_ net.Listener) context.Context {
					return aibridge.AsActor(ctx, userID, nil)
				}
				mockBridgeSrv.Start()

				// Make API call to aibridge for Anthropic /v1/messages, which will be routed via AWS Bedrock.
				// We override the AWS Bedrock client to route requests through our mock server.
				req := createAnthropicMessagesReq(t, mockBridgeSrv.URL, reqBody)
				client := &http.Client{}
				resp, err := client.Do(req)
				require.NoError(t, err)
				defer resp.Body.Close()

				// For streaming responses, consume the body to allow the stream to complete.
				if streaming {
					// Read the streaming response.
					_, err = io.ReadAll(resp.Body)
					require.NoError(t, err)
				}

				// Verify that Bedrock-specific model name was used in the request to the mock server
				// and the interception data.
				require.Equal(t, requestCount, 1)
				require.Equal(t, bedrockCfg.Model, receivedModelName)
				require.Len(t, recorderClient.interceptions, 1)
				require.Equal(t, recorderClient.interceptions[0].Model, bedrockCfg.Model)
				recorderClient.verifyAllInterceptionsEnded(t)
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

				arc := txtar.Parse(oaiSingleBuiltinTool)
				t.Logf("%s: %s", t.Name(), arc.Comment)

				files := filesMap(arc)
				require.Len(t, files, 3)
				require.Contains(t, files, fixtureRequest)
				require.Contains(t, files, fixtureStreamingResponse)
				require.Contains(t, files, fixtureNonStreamingResponse)

				reqBody := files[fixtureRequest]

				// Add the stream param to the request.
				newBody, err := setJSON(reqBody, "stream", tc.streaming)
				require.NoError(t, err)
				reqBody = newBody

				ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
				t.Cleanup(cancel)
				srv := newMockServer(ctx, t, files, nil)
				t.Cleanup(srv.Close)

				recorderClient := &mockRecorderClient{}

				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
				b, err := aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{aibridge.NewOpenAIProvider(openaiCfg(srv.URL, apiKey))}, recorderClient, mcp.NewServerProxyManager(nil), nil, logger)
				require.NoError(t, err)

				mockSrv := httptest.NewUnstartedServer(b)
				t.Cleanup(mockSrv.Close)
				mockSrv.Config.BaseContext = func(_ net.Listener) context.Context {
					return aibridge.AsActor(ctx, userID, nil)
				}
				mockSrv.Start()
				// Make API call to aibridge for OpenAI /v1/chat/completions
				req := createOpenAIChatCompletionsReq(t, mockSrv.URL, reqBody)

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

				require.Len(t, recorderClient.tokenUsages, 1)
				assert.EqualValues(t, tc.expectedInputTokens, calculateTotalInputTokens(recorderClient.tokenUsages), "input tokens miscalculated")
				assert.EqualValues(t, tc.expectedOutputTokens, calculateTotalOutputTokens(recorderClient.tokenUsages), "output tokens miscalculated")

				require.Len(t, recorderClient.toolUsages, 1)
				assert.Equal(t, "read_file", recorderClient.toolUsages[0].Tool)
				require.IsType(t, map[string]any{}, recorderClient.toolUsages[0].Args)
				require.Contains(t, recorderClient.toolUsages[0].Args, "path")
				assert.Equal(t, "README.md", recorderClient.toolUsages[0].Args.(map[string]any)["path"])

				require.Len(t, recorderClient.userPrompts, 1)
				assert.Equal(t, "how large is the README.md file in my current path", recorderClient.userPrompts[0].Prompt)

				recorderClient.verifyAllInterceptionsEnded(t)
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
		createRequest     func(*testing.T, string, []byte) *http.Request
		expectedMsgID     string
	}{
		{
			name:    aibridge.ProviderAnthropic,
			fixture: antSimple,
			configureFunc: func(addr string, client aibridge.Recorder) (*aibridge.RequestBridge, error) {
				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
				return aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(addr, apiKey), nil)}, client, mcp.NewServerProxyManager(nil), nil, logger)
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
			createRequest: createAnthropicMessagesReq,
			expectedMsgID: "msg_01Pvyf26bY17RcjmWfJsXGBn",
		},
		{
			name:    aibridge.ProviderOpenAI,
			fixture: oaiSimple,
			configureFunc: func(addr string, client aibridge.Recorder) (*aibridge.RequestBridge, error) {
				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
				return aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{aibridge.NewOpenAIProvider(openaiCfg(addr, apiKey))}, client, mcp.NewServerProxyManager(nil), nil, logger)
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
			createRequest: createOpenAIChatCompletionsReq,
			expectedMsgID: "chatcmpl-BwoiPTGRbKkY5rncfaM0s9KtWrq5N",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			for _, streaming := range []bool{true, false} {
				t.Run(fmt.Sprintf("streaming=%v", streaming), func(t *testing.T) {
					t.Parallel()

					arc := txtar.Parse(tc.fixture)
					t.Logf("%s: %s", t.Name(), arc.Comment)

					files := filesMap(arc)
					require.Len(t, files, 3)
					require.Contains(t, files, fixtureRequest)
					require.Contains(t, files, fixtureStreamingResponse)
					require.Contains(t, files, fixtureNonStreamingResponse)

					reqBody := files[fixtureRequest]

					// Add the stream param to the request.
					newBody, err := setJSON(reqBody, "stream", streaming)
					require.NoError(t, err)
					reqBody = newBody

					// Given: a mock API server and a Bridge through which the requests will flow.
					ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
					t.Cleanup(cancel)
					srv := newMockServer(ctx, t, files, nil)
					t.Cleanup(srv.Close)

					recorderClient := &mockRecorderClient{}

					b, err := tc.configureFunc(srv.URL, recorderClient)
					require.NoError(t, err)

					mockSrv := httptest.NewUnstartedServer(b)
					t.Cleanup(mockSrv.Close)
					mockSrv.Config.BaseContext = func(_ net.Listener) context.Context {
						return aibridge.AsActor(ctx, userID, nil)
					}
					mockSrv.Start()
					// When: calling the "API server" with the fixture's request body.
					req := tc.createRequest(t, mockSrv.URL, reqBody)
					client := &http.Client{}
					resp, err := client.Do(req)
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
					require.NotEmpty(t, recorderClient.userPrompts, "no prompts tracked")
					assert.Contains(t, recorderClient.userPrompts[0].Prompt, "how many angels can dance on the head of a pin")

					// Validate that responses have their IDs overridden with a interception ID rather than the original ID from the upstream provider.
					// The reason for this is that Bridge may make multiple upstream requests (i.e. to invoke injected tools), and clients will not be expecting
					// multiple messages in response to a single request.
					id, err := tc.getResponseIDFunc(streaming, resp)
					require.NoError(t, err, "failed to retrieve response ID")
					require.Nilf(t, uuid.Validate(id), "%s is not a valid UUID", id)

					require.GreaterOrEqual(t, len(recorderClient.tokenUsages), 1)
					require.Equal(t, recorderClient.tokenUsages[0].MsgID, tc.expectedMsgID)

					recorderClient.verifyAllInterceptionsEnded(t)
				})
			}
		})
	}
}

func setJSON(in []byte, key string, val bool) ([]byte, error) {
	var body map[string]any
	err := json.Unmarshal(in, &body)
	if err != nil {
		return nil, err
	}
	body[key] = val
	out, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	return out, nil
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
				bridge, err := aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{provider}, client, mcp.NewServerProxyManager(nil), nil, logger)
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
				bridge, err := aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{provider}, client, mcp.NewServerProxyManager(nil), nil, logger)
				require.NoError(t, err)
				return provider, bridge
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			arc := txtar.Parse(tc.fixture)
			t.Logf("%s: %s", t.Name(), arc.Comment)

			files := filesMap(arc)
			require.Contains(t, files, fixtureResponse)

			var receivedHeaders *http.Header
			respBody := files[fixtureResponse]
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

			recorderClient := &mockRecorderClient{}

			provider, bridge := tc.configureFunc(upstream.URL, recorderClient)

			bridgeSrv := httptest.NewUnstartedServer(bridge)
			bridgeSrv.Config.BaseContext = func(_ net.Listener) context.Context {
				return aibridge.AsActor(t.Context(), userID, nil)
			}
			bridgeSrv.Start()
			t.Cleanup(bridgeSrv.Close)

			req, err := http.NewRequestWithContext(t.Context(), "GET", fmt.Sprintf("%s/%s/v1/models", bridgeSrv.URL, tc.name), nil)
			require.NoError(t, err)

			resp, err := http.DefaultClient.Do(req)
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

// setupMCPServerProxiesForTest creates a mock MCP server, initializes the MCP bridge, and returns the tools
func setupMCPServerProxiesForTest(t *testing.T) (map[string]mcp.ServerProxier, *callAccumulator) {
	t.Helper()

	// Setup Coder MCP integration
	srv, acc := createMockMCPSrv(t)
	mcpSrv := httptest.NewServer(srv)
	t.Cleanup(mcpSrv.Close)

	logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
	proxy, err := mcp.NewStreamableHTTPServerProxy(logger, "coder", mcpSrv.URL, nil, nil, nil)
	require.NoError(t, err)

	// Initialize MCP client, fetch tools, and inject into bridge
	ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
	t.Cleanup(cancel)
	require.NoError(t, proxy.Init(ctx))
	tools := proxy.ListTools()
	require.NotEmpty(t, tools)

	return map[string]mcp.ServerProxier{proxy.Name(): proxy}, acc
}

type (
	configureFunc     func(string, aibridge.Recorder, *mcp.ServerProxyManager) (*aibridge.RequestBridge, error)
	createRequestFunc func(*testing.T, string, []byte) *http.Request
)

func TestAnthropicInjectedTools(t *testing.T) {
	t.Parallel()

	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming=%v", streaming), func(t *testing.T) {
			t.Parallel()

			configureFn := func(addr string, client aibridge.Recorder, srvProxyMgr *mcp.ServerProxyManager) (*aibridge.RequestBridge, error) {
				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
				return aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(addr, apiKey), nil)}, client, srvProxyMgr, nil, logger)
			}

			// Build the requirements & make the assertions which are common to all providers.
			recorderClient, mcpCalls, resp := setupInjectedToolTest(t, antSingleInjectedTool, streaming, configureFn, createAnthropicMessagesReq)

			// Ensure expected tool was invoked with expected input.
			require.Len(t, recorderClient.toolUsages, 1)
			require.Equal(t, mockToolName, recorderClient.toolUsages[0].Tool)
			expected, err := json.Marshal(map[string]any{"owner": "admin"})
			require.NoError(t, err)
			actual, err := json.Marshal(recorderClient.toolUsages[0].Args)
			require.NoError(t, err)
			require.EqualValues(t, expected, actual)
			invocations := mcpCalls.getCallsByTool(mockToolName)
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
			assert.EqualValues(t, 15308, calculateTotalInputTokens(recorderClient.tokenUsages))
			assert.EqualValues(t, 204, calculateTotalOutputTokens(recorderClient.tokenUsages))

			// Ensure we received exactly one prompt.
			require.Len(t, recorderClient.userPrompts, 1)
		})
	}
}

func TestOpenAIInjectedTools(t *testing.T) {
	t.Parallel()

	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming=%v", streaming), func(t *testing.T) {
			t.Parallel()

			configureFn := func(addr string, client aibridge.Recorder, srvProxyMgr *mcp.ServerProxyManager) (*aibridge.RequestBridge, error) {
				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
				return aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{aibridge.NewOpenAIProvider(openaiCfg(addr, apiKey))}, client, srvProxyMgr, nil, logger)
			}

			// Build the requirements & make the assertions which are common to all providers.
			recorderClient, mcpCalls, resp := setupInjectedToolTest(t, oaiSingleInjectedTool, streaming, configureFn, createOpenAIChatCompletionsReq)

			// Ensure expected tool was invoked with expected input.
			require.Len(t, recorderClient.toolUsages, 1)
			require.Equal(t, mockToolName, recorderClient.toolUsages[0].Tool)
			expected, err := json.Marshal(map[string]any{"owner": "admin"})
			require.NoError(t, err)
			actual, err := json.Marshal(recorderClient.toolUsages[0].Args)
			require.NoError(t, err)
			require.EqualValues(t, expected, actual)
			invocations := mcpCalls.getCallsByTool(mockToolName)
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
			require.EqualValues(t, 5047, calculateTotalInputTokens(recorderClient.tokenUsages))
			require.EqualValues(t, 105, calculateTotalOutputTokens(recorderClient.tokenUsages))

			// Ensure we received exactly one prompt.
			require.Len(t, recorderClient.userPrompts, 1)
		})
	}
}

// setupInjectedToolTest abstracts the common aspects required for the Test*InjectedTools tests.
// Kinda fugly right now, we can refactor this later.
func setupInjectedToolTest(t *testing.T, fixture []byte, streaming bool, configureFn configureFunc, createRequestFn func(*testing.T, string, []byte) *http.Request) (*mockRecorderClient, *callAccumulator, *http.Response) {
	t.Helper()

	arc := txtar.Parse(fixture)
	t.Logf("%s: %s", t.Name(), arc.Comment)

	files := filesMap(arc)
	require.Len(t, files, 5)
	require.Contains(t, files, fixtureRequest)
	require.Contains(t, files, fixtureStreamingResponse)
	require.Contains(t, files, fixtureNonStreamingResponse)
	require.Contains(t, files, fixtureStreamingToolResponse)
	require.Contains(t, files, fixtureNonStreamingToolResponse)

	reqBody := files[fixtureRequest]

	// Add the stream param to the request.
	newBody, err := setJSON(reqBody, "stream", streaming)
	require.NoError(t, err)
	reqBody = newBody

	ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
	t.Cleanup(cancel)

	// Setup mock server with response mutator for multi-turn interaction.
	mockSrv := newMockServer(ctx, t, files, func(reqCount uint32, resp []byte) []byte {
		if reqCount == 1 {
			return resp // First request gets the normal response (with tool call).
		}

		if reqCount > 2 {
			// This should not happen in single injected tool tests.
			return resp
		}

		// Second request gets the tool response.
		if streaming {
			return files[fixtureStreamingToolResponse]
		}
		return files[fixtureNonStreamingToolResponse]
	})
	t.Cleanup(mockSrv.Close)

	recorderClient := &mockRecorderClient{}

	// Setup MCP mcpProxiers.
	mcpProxiers, acc := setupMCPServerProxiesForTest(t)

	// Configure the bridge with injected tools.
	mcpMgr := mcp.NewServerProxyManager(mcpProxiers)
	require.NoError(t, mcpMgr.Init(ctx))
	b, err := configureFn(mockSrv.URL, recorderClient, mcpMgr)
	require.NoError(t, err)

	// Invoke request to mocked API via aibridge.
	bridgeSrv := httptest.NewUnstartedServer(b)
	bridgeSrv.Config.BaseContext = func(_ net.Listener) context.Context {
		return aibridge.AsActor(ctx, userID, nil)
	}
	bridgeSrv.Start()
	t.Cleanup(bridgeSrv.Close)

	req := createRequestFn(t, bridgeSrv.URL, reqBody)
	client := &http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode)
	t.Cleanup(func() {
		_ = resp.Body.Close()
	})

	// We must ALWAYS have 2 calls to the bridge for injected tool tests.
	require.Eventually(t, func() bool {
		return mockSrv.callCount.Load() == 2
	}, time.Second*10, time.Millisecond*50)

	return recorderClient, acc, resp
}

func TestErrorHandling(t *testing.T) {
	t.Parallel()

	// Tests that errors which occur *before* a streaming response begins, or in non-streaming requests, are handled as expected.
	t.Run("non-stream error", func(t *testing.T) {
		cases := []struct {
			name              string
			fixture           []byte
			createRequestFunc createRequestFunc
			configureFunc     configureFunc
			responseHandlerFn func(resp *http.Response)
		}{
			{
				name:              aibridge.ProviderAnthropic,
				fixture:           antNonStreamErr,
				createRequestFunc: createAnthropicMessagesReq,
				configureFunc: func(addr string, client aibridge.Recorder, srvProxyMgr *mcp.ServerProxyManager) (*aibridge.RequestBridge, error) {
					logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
					return aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(addr, apiKey), nil)}, client, srvProxyMgr, nil, logger)
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
				name:              aibridge.ProviderOpenAI,
				fixture:           oaiNonStreamErr,
				createRequestFunc: createOpenAIChatCompletionsReq,
				configureFunc: func(addr string, client aibridge.Recorder, srvProxyMgr *mcp.ServerProxyManager) (*aibridge.RequestBridge, error) {
					logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
					return aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{aibridge.NewOpenAIProvider(openaiCfg(addr, apiKey))}, client, srvProxyMgr, nil, logger)
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

						ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
						t.Cleanup(cancel)

						arc := txtar.Parse(tc.fixture)
						t.Logf("%s: %s", t.Name(), arc.Comment)

						files := filesMap(arc)
						require.Len(t, files, 3)
						require.Contains(t, files, fixtureRequest)
						require.Contains(t, files, fixtureStreamingResponse)
						require.Contains(t, files, fixtureNonStreamingResponse)

						reqBody := files[fixtureRequest]
						// Add the stream param to the request.
						newBody, err := setJSON(reqBody, "stream", streaming)
						require.NoError(t, err)
						reqBody = newBody

						// Setup mock server.
						mockResp := files[fixtureStreamingResponse]
						if !streaming {
							mockResp = files[fixtureNonStreamingResponse]
						}
						mockSrv := newMockHTTPReflector(ctx, t, mockResp)
						t.Cleanup(mockSrv.Close)

						recorderClient := &mockRecorderClient{}

						b, err := tc.configureFunc(mockSrv.URL, recorderClient, mcp.NewServerProxyManager(nil))
						require.NoError(t, err)

						// Invoke request to mocked API via aibridge.
						bridgeSrv := httptest.NewUnstartedServer(b)
						bridgeSrv.Config.BaseContext = func(_ net.Listener) context.Context {
							return aibridge.AsActor(ctx, userID, nil)
						}
						bridgeSrv.Start()
						t.Cleanup(bridgeSrv.Close)

						req := tc.createRequestFunc(t, bridgeSrv.URL, reqBody)
						resp, err := http.DefaultClient.Do(req)
						t.Cleanup(func() { _ = resp.Body.Close() })
						require.NoError(t, err)

						tc.responseHandlerFn(resp)
						recorderClient.verifyAllInterceptionsEnded(t)
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
			createRequestFunc createRequestFunc
			configureFunc     configureFunc
			responseHandlerFn func(resp *http.Response)
		}{
			{
				name:              aibridge.ProviderAnthropic,
				fixture:           antMidStreamErr,
				createRequestFunc: createAnthropicMessagesReq,
				configureFunc: func(addr string, client aibridge.Recorder, srvProxyMgr *mcp.ServerProxyManager) (*aibridge.RequestBridge, error) {
					logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
					return aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(addr, apiKey), nil)}, client, srvProxyMgr, nil, logger)
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
				name:              aibridge.ProviderOpenAI,
				fixture:           oaiMidStreamErr,
				createRequestFunc: createOpenAIChatCompletionsReq,
				configureFunc: func(addr string, client aibridge.Recorder, srvProxyMgr *mcp.ServerProxyManager) (*aibridge.RequestBridge, error) {
					logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
					return aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{aibridge.NewOpenAIProvider(openaiCfg(addr, apiKey))}, client, srvProxyMgr, nil, logger)
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

				ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
				t.Cleanup(cancel)

				arc := txtar.Parse(tc.fixture)
				t.Logf("%s: %s", t.Name(), arc.Comment)

				files := filesMap(arc)
				require.Len(t, files, 2)
				require.Contains(t, files, fixtureRequest)
				require.Contains(t, files, fixtureStreamingResponse)

				reqBody := files[fixtureRequest]

				// Setup mock server.
				mockSrv := newMockServer(ctx, t, files, nil)
				mockSrv.statusCode = http.StatusInternalServerError
				t.Cleanup(mockSrv.Close)

				recorderClient := &mockRecorderClient{}

				b, err := tc.configureFunc(mockSrv.URL, recorderClient, mcp.NewServerProxyManager(nil))
				require.NoError(t, err)

				// Invoke request to mocked API via aibridge.
				bridgeSrv := httptest.NewUnstartedServer(b)
				bridgeSrv.Config.BaseContext = func(_ net.Listener) context.Context {
					return aibridge.AsActor(ctx, userID, nil)
				}
				bridgeSrv.Start()
				t.Cleanup(bridgeSrv.Close)

				req := tc.createRequestFunc(t, bridgeSrv.URL, reqBody)
				resp, err := http.DefaultClient.Do(req)
				t.Cleanup(func() { _ = resp.Body.Close() })
				require.NoError(t, err)

				tc.responseHandlerFn(resp)
				recorderClient.verifyAllInterceptionsEnded(t)
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
		name              string
		fixture           []byte
		createRequestFunc createRequestFunc
		configureFunc     configureFunc
	}{
		{
			name:              aibridge.ProviderAnthropic,
			fixture:           antSimple,
			createRequestFunc: createAnthropicMessagesReq,
			configureFunc: func(addr string, client aibridge.Recorder, srvProxyMgr *mcp.ServerProxyManager) (*aibridge.RequestBridge, error) {
				return aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(addr, apiKey), nil)}, client, srvProxyMgr, nil, logger)
			},
		},
		{
			name:              aibridge.ProviderOpenAI,
			fixture:           oaiSimple,
			createRequestFunc: createOpenAIChatCompletionsReq,
			configureFunc: func(addr string, client aibridge.Recorder, srvProxyMgr *mcp.ServerProxyManager) (*aibridge.RequestBridge, error) {
				return aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{aibridge.NewOpenAIProvider(openaiCfg(addr, apiKey))}, client, srvProxyMgr, nil, logger)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			// Setup MCP tools.
			mcpProxiers, _ := setupMCPServerProxiesForTest(t)

			// Configure the bridge with injected tools.
			mcpMgr := mcp.NewServerProxyManager(mcpProxiers)
			require.NoError(t, mcpMgr.Init(ctx))

			arc := txtar.Parse(tc.fixture)
			t.Logf("%s: %s", t.Name(), arc.Comment)

			files := filesMap(arc)
			require.Contains(t, files, fixtureRequest)
			require.Contains(t, files, fixtureNonStreamingResponse)

			var (
				reference []byte
				reqCount  atomic.Int32
			)

			// Create a mock server that captures and compares request bodies.
			mockSrv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				reqCount.Add(1)

				// Capture the raw request body.
				raw, err := io.ReadAll(r.Body)
				defer r.Body.Close()
				require.NoError(t, err)
				require.NotEmpty(t, raw)

				// Store the first instance as the reference value.
				if reference == nil {
					reference = raw
				} else {
					// Compare all subsequent requests to the reference.
					assert.JSONEq(t, string(reference), string(raw))
				}

				// Return a valid API response.
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(files[fixtureNonStreamingResponse])
			}))
			mockSrv.Config.BaseContext = func(_ net.Listener) context.Context {
				return ctx
			}
			mockSrv.Start()
			t.Cleanup(mockSrv.Close)

			recorder := &mockRecorderClient{}
			bridge, err := tc.configureFunc(mockSrv.URL, recorder, mcpMgr)
			require.NoError(t, err)

			// Invoke request to mocked API via aibridge.
			bridgeSrv := httptest.NewUnstartedServer(bridge)
			bridgeSrv.Config.BaseContext = func(_ net.Listener) context.Context {
				return aibridge.AsActor(ctx, userID, nil)
			}
			bridgeSrv.Start()
			t.Cleanup(bridgeSrv.Close)

			// Make multiple requests and verify they all have identical payloads.
			count := 10
			for range count {
				req := tc.createRequestFunc(t, bridgeSrv.URL, files[fixtureRequest])
				client := &http.Client{}
				resp, err := client.Do(req)
				require.NoError(t, err)
				require.Equal(t, http.StatusOK, resp.StatusCode)
				_ = resp.Body.Close()
			}

			require.EqualValues(t, count, reqCount.Load())
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
		createRequest func(*testing.T, string, []byte) *http.Request
		envVars       map[string]string
		headerName    string
	}{
		{
			name:    aibridge.ProviderAnthropic,
			fixture: antSimple,
			configureFunc: func(addr string, client aibridge.Recorder) (*aibridge.RequestBridge, error) {
				logger := slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)
				return aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{aibridge.NewAnthropicProvider(anthropicCfg(addr, apiKey), nil)}, client, mcp.NewServerProxyManager(nil), nil, logger)
			},
			createRequest: createAnthropicMessagesReq,
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
				return aibridge.NewRequestBridge(t.Context(), []aibridge.Provider{aibridge.NewOpenAIProvider(openaiCfg(addr, apiKey))}, client, mcp.NewServerProxyManager(nil), nil, logger)
			},
			createRequest: createOpenAIChatCompletionsReq,
			envVars: map[string]string{
				"OPENAI_ORG_ID": "should-not-leak",
			},
			headerName: "OpenAI-Organization",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// NOTE: Cannot use t.Parallel() here because t.Setenv requires sequential execution.

			arc := txtar.Parse(tc.fixture)
			files := filesMap(arc)
			reqBody := files[fixtureRequest]

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			// Track headers received by the upstream server.
			var receivedHeaders http.Header
			srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedHeaders = r.Header.Clone()
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(files[fixtureNonStreamingResponse])
			}))
			srv.Config.BaseContext = func(_ net.Listener) context.Context {
				return ctx
			}
			srv.Start()
			t.Cleanup(srv.Close)

			// Set environment variables that the SDK would automatically read.
			// These should NOT leak into upstream requests.
			for key, val := range tc.envVars {
				t.Setenv(key, val)
			}

			recorderClient := &mockRecorderClient{}
			b, err := tc.configureFunc(srv.URL, recorderClient)
			require.NoError(t, err)

			mockSrv := httptest.NewUnstartedServer(b)
			t.Cleanup(mockSrv.Close)
			mockSrv.Config.BaseContext = func(_ net.Listener) context.Context {
				return aibridge.AsActor(ctx, userID, nil)
			}
			mockSrv.Start()

			req := tc.createRequest(t, mockSrv.URL, reqBody)
			client := &http.Client{}
			resp, err := client.Do(req)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, resp.StatusCode)
			defer resp.Body.Close()

			// Verify that environment values did not leak.
			require.NotNil(t, receivedHeaders)
			require.Empty(t, receivedHeaders.Get(tc.headerName))
		})
	}
}

func calculateTotalInputTokens(in []*aibridge.TokenUsageRecord) int64 {
	var total int64
	for _, el := range in {
		total += el.Input
	}
	return total
}

func calculateTotalOutputTokens(in []*aibridge.TokenUsageRecord) int64 {
	var total int64
	for _, el := range in {
		total += el.Output
	}
	return total
}

type archiveFileMap map[string][]byte

func filesMap(archive *txtar.Archive) archiveFileMap {
	if len(archive.Files) == 0 {
		return nil
	}

	out := make(archiveFileMap, len(archive.Files))
	for _, f := range archive.Files {
		out[f.Name] = f.Data
	}
	return out
}

func createAnthropicMessagesReq(t *testing.T, baseURL string, input []byte) *http.Request {
	t.Helper()

	req, err := http.NewRequestWithContext(t.Context(), "POST", baseURL+"/anthropic/v1/messages", bytes.NewReader(input))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	return req
}

func createOpenAIChatCompletionsReq(t *testing.T, baseURL string, input []byte) *http.Request {
	t.Helper()

	req, err := http.NewRequestWithContext(t.Context(), "POST", baseURL+"/openai/v1/chat/completions", bytes.NewReader(input))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	return req
}

type mockHTTPReflector struct {
	*httptest.Server
}

func newMockHTTPReflector(ctx context.Context, t *testing.T, resp []byte) *mockHTTPReflector {
	ref := &mockHTTPReflector{}

	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mock, err := http.ReadResponse(bufio.NewReader(bytes.NewBuffer(resp)), r)
		require.NoError(t, err)
		defer mock.Body.Close()

		// Copy headers from the mocked response.
		for key, values := range mock.Header {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}

		// Write the status code.
		w.WriteHeader(mock.StatusCode)

		// Copy the body.
		_, err = io.Copy(w, mock.Body)
		require.NoError(t, err)
	}))
	srv.Config.BaseContext = func(_ net.Listener) context.Context {
		return ctx
	}

	srv.Start()
	t.Cleanup(srv.Close)

	ref.Server = srv
	return ref
}

// TODO: replace this with mockHTTPReflector.
type mockServer struct {
	*httptest.Server

	callCount atomic.Uint32

	statusCode int
}

func newMockServer(ctx context.Context, t *testing.T, files archiveFileMap, responseMutatorFn func(reqCount uint32, resp []byte) []byte) *mockServer {
	t.Helper()

	ms := &mockServer{}
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		statusCode := http.StatusOK
		if ms.statusCode != 0 {
			statusCode = ms.statusCode
		}

		ms.callCount.Add(1)

		body, err := io.ReadAll(r.Body)
		defer r.Body.Close()
		require.NoError(t, err)

		type msg struct {
			Stream bool `json:"stream"`
		}
		var reqMsg msg
		require.NoError(t, json.Unmarshal(body, &reqMsg))

		if !reqMsg.Stream {
			resp := files[fixtureNonStreamingResponse]
			if responseMutatorFn != nil {
				resp = responseMutatorFn(ms.callCount.Load(), resp)
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(statusCode)
			w.Write(resp)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("Access-Control-Allow-Origin", "*")

		resp := files[fixtureStreamingResponse]
		if responseMutatorFn != nil {
			resp = responseMutatorFn(ms.callCount.Load(), resp)
		}

		scanner := bufio.NewScanner(bytes.NewReader(resp))
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
			return
		}

		for scanner.Scan() {
			line := scanner.Text()

			fmt.Fprintf(w, "%s\n", line)
			flusher.Flush()
		}

		if err := scanner.Err(); err != nil {
			http.Error(w, fmt.Sprintf("Error reading fixture: %v", err), http.StatusInternalServerError)
			return
		}
	}))

	srv.Config.BaseContext = func(_ net.Listener) context.Context {
		return ctx
	}

	srv.Start()
	t.Cleanup(srv.Close)

	ms.Server = srv
	return ms
}

var _ aibridge.Recorder = &mockRecorderClient{}

type mockRecorderClient struct {
	mu sync.Mutex

	interceptions    []*aibridge.InterceptionRecord
	tokenUsages      []*aibridge.TokenUsageRecord
	userPrompts      []*aibridge.PromptUsageRecord
	toolUsages       []*aibridge.ToolUsageRecord
	interceptionsEnd map[string]time.Time
}

func (m *mockRecorderClient) RecordInterception(ctx context.Context, req *aibridge.InterceptionRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.interceptions = append(m.interceptions, req)
	return nil
}

func (m *mockRecorderClient) RecordInterceptionEnded(ctx context.Context, req *aibridge.InterceptionRecordEnded) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.interceptionsEnd == nil {
		m.interceptionsEnd = make(map[string]time.Time)
	}
	if !slices.ContainsFunc(m.interceptions, func(intc *aibridge.InterceptionRecord) bool { return intc.ID == req.ID }) {
		return fmt.Errorf("id not found")
	}
	m.interceptionsEnd[req.ID] = req.EndedAt
	return nil
}

func (m *mockRecorderClient) RecordPromptUsage(ctx context.Context, req *aibridge.PromptUsageRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.userPrompts = append(m.userPrompts, req)
	return nil
}

func (m *mockRecorderClient) RecordTokenUsage(ctx context.Context, req *aibridge.TokenUsageRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tokenUsages = append(m.tokenUsages, req)
	return nil
}

func (m *mockRecorderClient) RecordToolUsage(ctx context.Context, req *aibridge.ToolUsageRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.toolUsages = append(m.toolUsages, req)
	return nil
}

// verify all recorded interceptions has been marked as completed
func (m *mockRecorderClient) verifyAllInterceptionsEnded(t *testing.T) {
	t.Helper()

	m.mu.Lock()
	defer m.mu.Unlock()
	require.Equalf(t, len(m.interceptions), len(m.interceptionsEnd), "got %v interception ended calls, want: %v", len(m.interceptionsEnd), len(m.interceptions))
	for _, intc := range m.interceptions {
		require.Containsf(t, m.interceptionsEnd, intc.ID, "interception with id: %v has not been ended", intc.ID)
	}
}

const mockToolName = "coder_list_workspaces"

// callAccumulator tracks all tool invocations by name and each instance's arguments.
type callAccumulator struct {
	calls   map[string][]any
	callsMu sync.Mutex
}

func newCallAccumulator() *callAccumulator {
	return &callAccumulator{
		calls: make(map[string][]any),
	}
}

func (a *callAccumulator) addCall(tool string, args any) {
	a.callsMu.Lock()
	defer a.callsMu.Unlock()

	a.calls[tool] = append(a.calls[tool], args)
}

func (a *callAccumulator) getCallsByTool(name string) []any {
	a.callsMu.Lock()
	defer a.callsMu.Unlock()

	// Protect against concurrent access of the slice.
	result := make([]any, len(a.calls[name]))
	copy(result, a.calls[name])
	return result
}

func createMockMCPSrv(t *testing.T) (http.Handler, *callAccumulator) {
	t.Helper()

	s := server.NewMCPServer(
		"Mock coder MCP server",
		"1.0.0",
		server.WithToolCapabilities(true),
	)

	// Accumulate tool calls & their arguments.
	acc := newCallAccumulator()

	for _, name := range []string{mockToolName, "coder_list_templates", "coder_template_version_parameters", "coder_get_authenticated_user", "coder_create_workspace_build"} {
		tool := mcplib.NewTool(name,
			mcplib.WithDescription(fmt.Sprintf("Mock of the %s tool", name)),
		)
		s.AddTool(tool, func(ctx context.Context, request mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
			acc.addCall(request.Params.Name, request.Params.Arguments)
			return mcplib.NewToolResultText("mock"), nil
		})
	}

	return server.NewStreamableHTTPServer(s), acc
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
