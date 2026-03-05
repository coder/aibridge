package integrationtest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/fixtures"
	"github.com/coder/aibridge/intercept"
	"github.com/coder/aibridge/internal/testutil"
	"github.com/coder/aibridge/provider"
	"github.com/coder/aibridge/recorder"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v3"
	oaissestream "github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/goleak"
)

type (
	providerFunc      func(addr string) aibridge.Provider
	createRequestFunc func(*testing.T, string, []byte) *http.Request
)

func newAnthropicProvider(addr string) aibridge.Provider {
	return provider.NewAnthropic(anthropicCfg(addr, apiKey), nil)
}

func newOpenAIProvider(addr string) aibridge.Provider {
	return provider.NewOpenAI(openAICfg(addr, apiKey))
}

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

func TestSimple(t *testing.T) {
	t.Parallel()

	getAnthropicResponseID := func(streaming bool, resp *http.Response) (string, error) {
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
	}

	getOpenAIResponseID := func(streaming bool, resp *http.Response) (string, error) {
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
	}

	testCases := []struct {
		name              string
		fixture           []byte
		basePath          string
		expectedPath      string
		providerFn        providerFunc
		getResponseIDFunc func(streaming bool, resp *http.Response) (string, error)
		createRequest     createRequestFunc
		expectedMsgID     string
		userAgent         string
		expectedClient    aibridge.Client
	}{
		{
			name:              config.ProviderAnthropic,
			fixture:           fixtures.AntSimple,
			basePath:          "",
			expectedPath:      "/v1/messages",
			providerFn:        newAnthropicProvider,
			getResponseIDFunc: getAnthropicResponseID,
			createRequest:     createAnthropicMessagesReq,
			expectedMsgID:     "msg_01Pvyf26bY17RcjmWfJsXGBn",
			userAgent:         "claude-cli/2.0.67 (external, cli)",
			expectedClient:    aibridge.ClientClaudeCode,
		},
		{
			name:              config.ProviderOpenAI,
			fixture:           fixtures.OaiChatSimple,
			basePath:          "",
			expectedPath:      "/chat/completions",
			providerFn:        newOpenAIProvider,
			getResponseIDFunc: getOpenAIResponseID,
			createRequest:     createOpenAIChatCompletionsReq,
			expectedMsgID:     "chatcmpl-BwoiPTGRbKkY5rncfaM0s9KtWrq5N",
			userAgent:         "codex_cli_rs/0.87.0 (Mac OS 26.2.0; arm64)",
			expectedClient:    aibridge.ClientCodex,
		},
		{
			name:              config.ProviderAnthropic + "_baseURL_path",
			fixture:           fixtures.AntSimple,
			basePath:          "/api",
			expectedPath:      "/api/v1/messages",
			providerFn:        newAnthropicProvider,
			getResponseIDFunc: getAnthropicResponseID,
			createRequest:     createAnthropicMessagesReq,
			expectedMsgID:     "msg_01Pvyf26bY17RcjmWfJsXGBn",
			userAgent:         "GitHubCopilotChat/0.37.2026011603",
			expectedClient:    aibridge.ClientCopilotVSC,
		},
		{
			name:              config.ProviderOpenAI + "_baseURL_path",
			fixture:           fixtures.OaiChatSimple,
			basePath:          "/api",
			expectedPath:      "/api/chat/completions",
			providerFn:        newOpenAIProvider,
			getResponseIDFunc: getOpenAIResponseID,
			createRequest:     createOpenAIChatCompletionsReq,
			expectedMsgID:     "chatcmpl-BwoiPTGRbKkY5rncfaM0s9KtWrq5N",
			userAgent:         "Zed/0.219.4+stable.119.abc123 (macos; aarch64)",
			expectedClient:    aibridge.ClientZed,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			for _, streaming := range []bool{true, false} {
				t.Run(fmt.Sprintf("streaming=%v", streaming), func(t *testing.T) {
					t.Parallel()

					ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
					t.Cleanup(cancel)

					fix := fixtures.Parse(t, tc.fixture)
					upstream := newMockUpstream(t, ctx, newFixtureResponse(fix))

					ts := newBridgeTestServer(t, ctx,
						[]aibridge.Provider{tc.providerFn(upstream.URL + tc.basePath)},
					)

					// When: calling the "API server" with the fixture's request body.
					reqBody, err := sjson.SetBytes(fix.Request(), "stream", streaming)
					require.NoError(t, err)
					req := tc.createRequest(t, ts.URL, reqBody)
					req.Header.Set("User-Agent", tc.userAgent)
					client := &http.Client{}
					resp, err := client.Do(req)
					require.NoError(t, err)
					require.Equal(t, http.StatusOK, resp.StatusCode)
					defer resp.Body.Close()

					// Then: I expect the upstream request to have the correct path.
					received := upstream.receivedRequests()
					require.Len(t, received, 1)
					require.Equal(t, tc.expectedPath, received[0].Path)

					// Then: I expect a non-empty response.
					bodyBytes, err := io.ReadAll(resp.Body)
					require.NoError(t, err)
					assert.NotEmpty(t, bodyBytes, "should have received response body")

					// Reset the body after being read.
					resp.Body = io.NopCloser(bytes.NewReader(bodyBytes))

					// Then: I expect the prompt to have been tracked.
					promptUsages := ts.Recorder.RecordedPromptUsages()
					require.NotEmpty(t, promptUsages, "no prompts tracked")
					assert.Contains(t, promptUsages[0].Prompt, "how many angels can dance on the head of a pin")

					// Validate that responses have their IDs overridden with a interception ID rather than the original ID from the upstream provider.
					// The reason for this is that Bridge may make multiple upstream requests (i.e. to invoke injected tools), and clients will not be expecting
					// multiple messages in response to a single request.
					id, err := tc.getResponseIDFunc(streaming, resp)
					require.NoError(t, err, "failed to retrieve response ID")
					require.Nilf(t, uuid.Validate(id), "%s is not a valid UUID", id)

					tokenUsages := ts.Recorder.RecordedTokenUsages()
					require.GreaterOrEqual(t, len(tokenUsages), 1)
					require.Equal(t, tokenUsages[0].MsgID, tc.expectedMsgID)

					// Validate user agent and client have been recorded.
					interceptions := ts.Recorder.RecordedInterceptions()
					require.Len(t, interceptions, 1, "expected exactly one interception, got: %v", interceptions)
					assert.Equal(t, tc.userAgent, interceptions[0].UserAgent)
					assert.Equal(t, string(tc.expectedClient), interceptions[0].Client)

					ts.Recorder.VerifyAllInterceptionsEnded(t)
				})
			}
		})
	}
}

func TestSessionIDTracking(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name           string
		fixture        []byte
		expectedClient aibridge.Client
		sessionID      string
		configureFunc  func(*testing.T, string, aibridge.Recorder) (*aibridge.RequestBridge, error)
		createRequest  func(t *testing.T, baseURL string, body []byte) *http.Request
	}{
		// Session in header.
		{
			name:           "mux",
			fixture:        fixtures.AntSimple,
			expectedClient: aibridge.ClientMux,
			sessionID:      "mux-workspace-321",
			configureFunc: func(t *testing.T, addr string, client aibridge.Recorder) (*aibridge.RequestBridge, error) {
				t.Helper()
				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
				providers := []aibridge.Provider{provider.NewAnthropic(anthropicCfg(addr, apiKey), nil)}
				return aibridge.NewRequestBridge(t.Context(), providers, client, mcp.NewServerProxyManager(nil, testTracer), logger, nil, testTracer)
			},
			createRequest: func(t *testing.T, baseURL string, body []byte) *http.Request {
				t.Helper()
				req := createAnthropicMessagesReq(t, baseURL, body)
				req.Header.Set("User-Agent", "mux/1.0.0")
				req.Header.Set("X-Mux-Workspace-Id", "mux-workspace-321")
				return req
			},
		},
		// Session in body.
		{
			name:           "claude_code",
			fixture:        fixtures.AntSimple,
			expectedClient: aibridge.ClientClaudeCode,
			sessionID:      "f47ac10b-58cc-4372-a567-0e02b2c3d479",
			configureFunc: func(t *testing.T, addr string, client aibridge.Recorder) (*aibridge.RequestBridge, error) {
				t.Helper()
				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
				providers := []aibridge.Provider{provider.NewAnthropic(anthropicCfg(addr, apiKey), nil)}
				return aibridge.NewRequestBridge(t.Context(), providers, client, mcp.NewServerProxyManager(nil, testTracer), logger, nil, testTracer)
			},
			createRequest: func(t *testing.T, baseURL string, body []byte) *http.Request {
				t.Helper()
				// Claude Code embeds the session ID in metadata.user_id within the body.
				body, err := sjson.SetBytes(body, "metadata.user_id",
					"user_abc123_account_456_session_f47ac10b-58cc-4372-a567-0e02b2c3d479")
				require.NoError(t, err)
				req := createAnthropicMessagesReq(t, baseURL, body)
				req.Header.Set("User-Agent", "claude-cli/2.0.67 (external, cli)")
				return req
			},
		},
		// No session.
		{
			name:           "zed",
			fixture:        fixtures.AntSimple,
			expectedClient: aibridge.ClientZed,
			configureFunc: func(t *testing.T, addr string, client aibridge.Recorder) (*aibridge.RequestBridge, error) {
				t.Helper()
				logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
				providers := []aibridge.Provider{provider.NewAnthropic(anthropicCfg(addr, apiKey), nil)}
				return aibridge.NewRequestBridge(t.Context(), providers, client, mcp.NewServerProxyManager(nil, testTracer), logger, nil, testTracer)
			},
			createRequest: func(t *testing.T, baseURL string, body []byte) *http.Request {
				t.Helper()
				req := createAnthropicMessagesReq(t, baseURL, body)
				req.Header.Set("User-Agent", "Zed/0.219.4+stable.119.abc123 (macos; aarch64)")
				return req
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			fix := fixtures.Parse(t, tc.fixture)
			upstream := testutil.NewMockUpstream(t, ctx, testutil.NewFixtureResponse(fix))

			recorderClient := &testutil.MockRecorder{}

			b, err := tc.configureFunc(t, upstream.URL, recorderClient)
			require.NoError(t, err)
			mockSrv := httptest.NewUnstartedServer(b)
			t.Cleanup(mockSrv.Close)
			mockSrv.Config.BaseContext = func(_ net.Listener) context.Context {
				return aibcontext.AsActor(ctx, userID, nil)
			}
			mockSrv.Start()

			req := tc.createRequest(t, mockSrv.URL, fix.Request())
			resp, err := http.DefaultClient.Do(req)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, resp.StatusCode)
			defer resp.Body.Close()

			// Drain the body to let the stream complete.
			_, err = io.ReadAll(resp.Body)
			require.NoError(t, err)

			interceptions := recorderClient.RecordedInterceptions()
			require.Len(t, interceptions, 1, "expected exactly one interception")
			assert.Equal(t, string(tc.expectedClient), interceptions[0].Client)

			if tc.sessionID == "" {
				assert.Nil(t, interceptions[0].ClientSessionID, "expected nil session ID for %s", tc.name)
			} else {
				require.NotNil(t, interceptions[0].ClientSessionID, "expected non-nil session ID for %s", tc.name)
				assert.Equal(t, tc.sessionID, *interceptions[0].ClientSessionID)
			}

			recorderClient.VerifyAllInterceptionsEnded(t)
		})
	}
}

func TestFallthrough(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name                 string
		providerName         string
		fixture              []byte
		basePath             string
		requestPath          string
		expectedUpstreamPath string
		providerFn           providerFunc
	}{
		{
			name:                 "ant_empty_base_url_path",
			providerName:         config.ProviderAnthropic,
			fixture:              fixtures.AntFallthrough,
			basePath:             "",
			requestPath:          "/anthropic/v1/models",
			expectedUpstreamPath: "/v1/models",
			providerFn:           newAnthropicProvider,
		},
		{
			name:                 "oai_empty_base_url_path",
			providerName:         config.ProviderOpenAI,
			fixture:              fixtures.OaiChatFallthrough,
			basePath:             "",
			requestPath:          "/openai/v1/models",
			expectedUpstreamPath: "/models",
			providerFn:           newOpenAIProvider,
		},
		{
			name:                 "ant_some_base_url_path",
			providerName:         config.ProviderAnthropic,
			fixture:              fixtures.AntFallthrough,
			basePath:             "/api",
			requestPath:          "/anthropic/v1/models",
			expectedUpstreamPath: "/api/v1/models",
			providerFn:           newAnthropicProvider,
		},
		{
			name:                 "oai_some_base_url_path",
			providerName:         config.ProviderOpenAI,
			fixture:              fixtures.OaiChatFallthrough,
			basePath:             "/api",
			requestPath:          "/openai/v1/models",
			expectedUpstreamPath: "/api/models",
			providerFn:           newOpenAIProvider,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			fix := fixtures.Parse(t, tc.fixture)
			upstream := newMockUpstream(t, t.Context(), newFixtureResponse(fix))
			p := tc.providerFn(upstream.URL + tc.basePath)
			ts := newBridgeTestServer(t, t.Context(),
				[]aibridge.Provider{p},
			)

			req, err := http.NewRequestWithContext(t.Context(), "GET", fmt.Sprintf("%s%s", ts.URL, tc.requestPath), nil)
			require.NoError(t, err)

			resp, err := http.DefaultClient.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()

			require.Equal(t, http.StatusOK, resp.StatusCode)

			// Verify upstream received the request at the expected path
			// with the API key header.
			received := upstream.receivedRequests()
			require.Len(t, received, 1)
			require.Equal(t, tc.expectedUpstreamPath, received[0].Path)
			require.Contains(t, received[0].Header.Get(p.AuthHeader()), apiKey)

			gotBytes, err := io.ReadAll(resp.Body)
			require.NoError(t, err)

			// Compare JSON bodies for semantic equality.
			var got any
			var exp any
			require.NoError(t, json.Unmarshal(gotBytes, &got))
			require.NoError(t, json.Unmarshal(fix.NonStreaming(), &exp))
			require.EqualValues(t, exp, got)
		})
	}
}

// setupInjectedToolTest abstracts common setup required for injected-tool integration tests.
func setupInjectedToolTest(
	t *testing.T,
	fixture []byte,
	streaming bool,
	providerFn providerFunc,
	tracer trace.Tracer,
	actorID string,
	createRequestFn func(*testing.T, string, []byte) *http.Request,
	toolRequestValidatorFn func(*http.Request, []byte),
) (*testutil.MockRecorder, *mockMCP, *http.Response) {
	t.Helper()

	ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
	t.Cleanup(cancel)

	fix := fixtures.Parse(t, fixture)

	// Setup mock server for multi-turn interaction.
	// First request → tool call response, second → tool response.
	firstResp := newFixtureResponse(fix)
	toolResp := newFixtureToolResponse(fix)
	toolResp.OnRequest = toolRequestValidatorFn
	upstream := newMockUpstream(t, ctx, firstResp, toolResp)

	mockMCP := setupMCPForTest(t, tracer)

	ts := newBridgeTestServer(t, ctx,
		[]aibridge.Provider{providerFn(upstream.URL)},
		withMCP(mockMCP),
		withTracer(tracer),
		withActor(actorID, nil),
	)

	// Add the stream param to the request.
	reqBody, err := sjson.SetBytes(fix.Request(), "stream", streaming)
	require.NoError(t, err)

	req := createRequestFn(t, ts.URL, reqBody)
	client := &http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode)
	t.Cleanup(func() {
		_ = resp.Body.Close()
	})

	// We must ALWAYS have 2 calls to the bridge for injected tool tests.
	require.Eventually(t, func() bool {
		return upstream.Calls.Load() == 2
	}, time.Second*10, time.Millisecond*50)

	return ts.Recorder, mockMCP, resp
}

func TestErrorHandling(t *testing.T) {
	t.Parallel()

	// Tests that errors which occur *before* a streaming response begins, or in non-streaming requests, are handled as expected.
	t.Run("non-stream error", func(t *testing.T) {
		cases := []struct {
			name              string
			fixture           []byte
			createRequestFunc createRequestFunc
			providerFn        providerFunc
			responseHandlerFn func(resp *http.Response)
		}{
			{
				name:              config.ProviderAnthropic,
				fixture:           fixtures.AntNonStreamError,
				createRequestFunc: createAnthropicMessagesReq,
				providerFn:        newAnthropicProvider,
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
				name:              config.ProviderOpenAI,
				fixture:           fixtures.OaiChatNonStreamError,
				createRequestFunc: createOpenAIChatCompletionsReq,
				providerFn:        newOpenAIProvider,
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

						// Setup mock server. Error fixtures contain raw HTTP
						// responses that may cause the bridge to retry.
						fix := fixtures.Parse(t, tc.fixture)
						upstream := newMockUpstream(t, ctx, newFixtureResponse(fix))

						ts := newBridgeTestServer(t, ctx,
							[]aibridge.Provider{tc.providerFn(upstream.URL)},
						)

						// Add the stream param to the request.
						reqBody, err := sjson.SetBytes(fix.Request(), "stream", streaming)
						require.NoError(t, err)

						req := tc.createRequestFunc(t, ts.URL, reqBody)
						resp, err := http.DefaultClient.Do(req)
						t.Cleanup(func() { _ = resp.Body.Close() })
						require.NoError(t, err)

						tc.responseHandlerFn(resp)
						ts.Recorder.VerifyAllInterceptionsEnded(t)
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
			providerFn        providerFunc
			responseHandlerFn func(resp *http.Response)
		}{
			{
				name:              config.ProviderAnthropic,
				fixture:           fixtures.AntMidStreamError,
				createRequestFunc: createAnthropicMessagesReq,
				providerFn:        newAnthropicProvider,
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
				name:              config.ProviderOpenAI,
				fixture:           fixtures.OaiChatMidStreamError,
				createRequestFunc: createOpenAIChatCompletionsReq,
				providerFn:        newOpenAIProvider,
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

				// Setup mock server.
				fix := fixtures.Parse(t, tc.fixture)
				upstream := newMockUpstream(t, ctx, newFixtureResponse(fix))
				upstream.StatusCode = http.StatusInternalServerError

				ts := newBridgeTestServer(t, ctx,
					[]aibridge.Provider{tc.providerFn(upstream.URL)},
				)

				req := tc.createRequestFunc(t, ts.URL, fix.Request())
				resp, err := http.DefaultClient.Do(req)
				t.Cleanup(func() { _ = resp.Body.Close() })
				require.NoError(t, err)
				ts.Close()

				tc.responseHandlerFn(resp)
				ts.Recorder.VerifyAllInterceptionsEnded(t)
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

	cases := []struct {
		name              string
		fixture           []byte
		createRequestFunc createRequestFunc
		providerFn        providerFunc
	}{
		{
			name:              config.ProviderAnthropic,
			fixture:           fixtures.AntSimple,
			createRequestFunc: createAnthropicMessagesReq,
			providerFn:        newAnthropicProvider,
		},
		{
			name:              config.ProviderOpenAI,
			fixture:           fixtures.OaiChatSimple,
			createRequestFunc: createOpenAIChatCompletionsReq,
			providerFn:        newOpenAIProvider,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			// Setup MCP tools.
			mockMCP := setupMCPForTest(t, defaultTracer)

			fix := fixtures.Parse(t, tc.fixture)

			// Create a mock upstream that serves the same blocking response for each request.
			count := 10
			responses := make([]upstreamResponse, count)
			for i := range count {
				responses[i] = newFixtureResponse(fix)
			}
			upstream := newMockUpstream(t, ctx, responses...)

			ts := newBridgeTestServer(t, ctx,
				[]aibridge.Provider{tc.providerFn(upstream.URL)},
				withMCP(mockMCP),
			)

			// Make multiple requests and verify they all have identical payloads.
			for range count {
				req := tc.createRequestFunc(t, ts.URL, fix.Request())
				client := &http.Client{}
				resp, err := client.Do(req)
				require.NoError(t, err)
				require.Equal(t, http.StatusOK, resp.StatusCode)
				_ = resp.Body.Close()
			}

			// All upstream request bodies should be identical.
			received := upstream.receivedRequests()
			require.Len(t, received, count)
			reference := string(received[0].Body)
			for _, r := range received[1:] {
				assert.JSONEq(t, reference, string(r.Body))
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
		providerFn    providerFunc
		createRequest createRequestFunc
		envVars       map[string]string
		headerName    string
	}{
		{
			name:          config.ProviderAnthropic,
			fixture:       fixtures.AntSimple,
			providerFn:    newAnthropicProvider,
			createRequest: createAnthropicMessagesReq,
			envVars: map[string]string{
				"ANTHROPIC_AUTH_TOKEN": "should-not-leak",
			},
			headerName: "Authorization", // We only send through the X-Api-Key, so this one should not be present.
		},
		{
			name:          config.ProviderOpenAI,
			fixture:       fixtures.OaiChatSimple,
			providerFn:    newOpenAIProvider,
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

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			fix := fixtures.Parse(t, tc.fixture)
			upstream := newMockUpstream(t, ctx, newFixtureResponse(fix))

			// Set environment variables that the SDK would automatically read.
			// These should NOT leak into upstream requests.
			for key, val := range tc.envVars {
				t.Setenv(key, val)
			}

			ts := newBridgeTestServer(t, ctx,
				[]aibridge.Provider{tc.providerFn(upstream.URL)},
			)

			req := tc.createRequest(t, ts.URL, fix.Request())
			client := &http.Client{}
			resp, err := client.Do(req)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, resp.StatusCode)
			defer resp.Body.Close()

			// Verify that environment values did not leak.
			received := upstream.receivedRequests()
			require.Len(t, received, 1)
			require.Empty(t, received[0].Header.Get(tc.headerName))
		})
	}
}

func TestActorHeaders(t *testing.T) {
	t.Parallel()

	actorUsername := "bob"

	cases := []struct {
		name             string
		createRequest    createRequestFunc
		createProviderFn func(url, key string, sendHeaders bool) aibridge.Provider
		fixture          []byte
		streaming        bool
	}{
		{
			name:          "openai/v1/chat/completions",
			createRequest: createOpenAIChatCompletionsReq,
			createProviderFn: func(url, key string, sendHeaders bool) aibridge.Provider {
				cfg := openAICfg(url, key)
				cfg.SendActorHeaders = sendHeaders
				return provider.NewOpenAI(cfg)
			},
			fixture:   fixtures.OaiChatSimple,
			streaming: true,
		},
		{
			name:          "openai/v1/chat/completions",
			createRequest: createOpenAIChatCompletionsReq,
			createProviderFn: func(url, key string, sendHeaders bool) aibridge.Provider {
				cfg := openAICfg(url, key)
				cfg.SendActorHeaders = sendHeaders
				return provider.NewOpenAI(cfg)
			},
			fixture:   fixtures.OaiChatSimple,
			streaming: false,
		},
		{
			name:          "openai/v1/responses",
			createRequest: createOpenAIResponsesReq,
			createProviderFn: func(url, key string, sendHeaders bool) aibridge.Provider {
				cfg := openAICfg(url, key)
				cfg.SendActorHeaders = sendHeaders
				return provider.NewOpenAI(cfg)
			},
			fixture:   fixtures.OaiResponsesStreamingSimple,
			streaming: true,
		},
		{
			name:          "openai/v1/responses",
			createRequest: createOpenAIResponsesReq,
			createProviderFn: func(url, key string, sendHeaders bool) aibridge.Provider {
				cfg := openAICfg(url, key)
				cfg.SendActorHeaders = sendHeaders
				return provider.NewOpenAI(cfg)
			},
			fixture:   fixtures.OaiResponsesBlockingSimple,
			streaming: false,
		},
		{
			name:          "anthropic/v1/messages",
			createRequest: createAnthropicMessagesReq,
			createProviderFn: func(url, key string, sendHeaders bool) aibridge.Provider {
				cfg := anthropicCfg(url, key)
				cfg.SendActorHeaders = sendHeaders
				return provider.NewAnthropic(cfg, nil)
			},
			fixture:   fixtures.AntSimple,
			streaming: true,
		},
		{
			name:          "anthropic/v1/messages",
			createRequest: createAnthropicMessagesReq,
			createProviderFn: func(url, key string, sendHeaders bool) aibridge.Provider {
				cfg := anthropicCfg(url, key)
				cfg.SendActorHeaders = sendHeaders
				return provider.NewAnthropic(cfg, nil)
			},
			fixture:   fixtures.AntSimple,
			streaming: false,
		},
	}

	for _, tc := range cases {
		for _, send := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/streaming=%v/send-headers=%v", tc.name, tc.streaming, send), func(t *testing.T) {
				t.Parallel()

				ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
				t.Cleanup(cancel)

				// Track headers received by the upstream server.
				var receivedHeaders http.Header
				srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					receivedHeaders = r.Header.Clone()
					w.WriteHeader(http.StatusTeapot)
				}))
				srv.Config.BaseContext = func(_ net.Listener) context.Context {
					return ctx
				}
				srv.Start()
				t.Cleanup(srv.Close)

				p := tc.createProviderFn(srv.URL, apiKey, send)

				metadataKey := "Username"
				ts := newBridgeTestServer(t, ctx,
					[]aibridge.Provider{p},
					withActor(defaultActorID, recorder.Metadata{
						metadataKey: actorUsername,
					}),
				)

				// Add the stream param to the request.
				reqBody, err := sjson.SetBytes(fixtures.Request(t, tc.fixture), "stream", tc.streaming)
				require.NoError(t, err)

				req := tc.createRequest(t, ts.URL, reqBody)
				client := &http.Client{}
				resp, err := client.Do(req)
				require.NoError(t, err)
				require.NotEmpty(t, receivedHeaders)
				defer resp.Body.Close()

				// Verify that the actor headers were only received if intended.
				found := make(map[string][]string)
				for k, v := range receivedHeaders {
					k = strings.ToLower(k)
					if intercept.IsActorHeader(k) {
						found[k] = v
					}
				}

				if send {
					require.Equal(t, found[strings.ToLower(intercept.ActorIDHeader())], []string{defaultActorID})
					require.Equal(t, found[strings.ToLower(intercept.ActorMetadataHeader(metadataKey))], []string{actorUsername})
				} else {
					require.Empty(t, found)
				}
			})
		}
	}
}

func calculateTotalInputTokens(in []*recorder.TokenUsageRecord) int64 {
	var total int64
	for _, el := range in {
		total += el.Input
	}
	return total
}

func calculateTotalOutputTokens(in []*recorder.TokenUsageRecord) int64 {
	var total int64
	for _, el := range in {
		total += el.Output
	}
	return total
}
