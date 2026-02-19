package aibridge_test

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"cdr.dev/slog/v3"
	"cdr.dev/slog/v3/sloggers/slogtest"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/config"
	aibcontext "github.com/coder/aibridge/context"
	"github.com/coder/aibridge/fixtures"
	"github.com/coder/aibridge/intercept/apidump"
	"github.com/coder/aibridge/internal/testutil"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/provider"
	"github.com/stretchr/testify/require"
	"golang.org/x/tools/txtar"
)

func openaiCfgWithAPIDump(url, key, dumpDir string) config.OpenAI {
	return config.OpenAI{
		BaseURL:    url,
		Key:        key,
		APIDumpDir: dumpDir,
	}
}

func anthropicCfgWithAPIDump(url, key, dumpDir string) config.Anthropic {
	return config.Anthropic{
		BaseURL:    url,
		Key:        key,
		APIDumpDir: dumpDir,
	}
}

func TestAPIDump(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name              string
		fixture           []byte
		providersFunc     func(addr, dumpDir string) []aibridge.Provider
		createRequestFunc createRequestFunc
	}{
		{
			name:    "anthropic",
			fixture: fixtures.AntSimple,
			providersFunc: func(addr, dumpDir string) []aibridge.Provider {
				return []aibridge.Provider{provider.NewAnthropic(anthropicCfgWithAPIDump(addr, apiKey, dumpDir), nil)}
			},
			createRequestFunc: createAnthropicMessagesReq,
		},
		{
			name:    "openai_chat_completions",
			fixture: fixtures.OaiChatSimple,
			providersFunc: func(addr, dumpDir string) []aibridge.Provider {
				return []aibridge.Provider{provider.NewOpenAI(openaiCfgWithAPIDump(addr, apiKey, dumpDir))}
			},
			createRequestFunc: createOpenAIChatCompletionsReq,
		},
		{
			name:    "openai_responses",
			fixture: fixtures.OaiResponsesBlockingSimple,
			providersFunc: func(addr, dumpDir string) []aibridge.Provider {
				return []aibridge.Provider{provider.NewOpenAI(openaiCfgWithAPIDump(addr, apiKey, dumpDir))}
			},
			createRequestFunc: createOpenAIResponsesReq,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			arc := txtar.Parse(tc.fixture)
			files := filesMap(arc)
			require.Contains(t, files, fixtureRequest)
			require.Contains(t, files, fixtureNonStreamingResponse)

			reqBody := files[fixtureRequest]

			// Setup mock upstream server.
			srv := newMockServer(ctx, t, files, nil, nil)
			t.Cleanup(srv.Close)

			// Create temp dir for API dumps.
			dumpDir := t.TempDir()

			recorderClient := &testutil.MockRecorder{}
			b, err := aibridge.NewRequestBridge(t.Context(), tc.providersFunc(srv.URL, dumpDir), recorderClient, mcp.NewServerProxyManager(nil, testTracer), logger, nil, testTracer)
			require.NoError(t, err)

			mockSrv := httptest.NewUnstartedServer(b)
			t.Cleanup(mockSrv.Close)
			mockSrv.Config.BaseContext = func(_ net.Listener) context.Context {
				return aibcontext.AsActor(ctx, userID, nil)
			}
			mockSrv.Start()

			req := tc.createRequestFunc(t, mockSrv.URL, reqBody)
			resp, err := http.DefaultClient.Do(req)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, resp.StatusCode)
			defer resp.Body.Close()
			_, _ = io.ReadAll(resp.Body)

			// Verify dump files were created.
			interceptions := recorderClient.RecordedInterceptions()
			require.Len(t, interceptions, 1)
			interceptionID := interceptions[0].ID

			// Find dump files for this interception by walking the dump directory.
			var reqDumpFile, respDumpFile string
			err = filepath.Walk(dumpDir, func(path string, info os.FileInfo, err error) error {
				if err != nil {
					return err
				}
				if info.IsDir() {
					return nil
				}
				// Files are named: {timestamp}-{interceptionID}.{req|resp}.txt
				if strings.Contains(path, interceptionID) {
					if strings.HasSuffix(path, apidump.SuffixRequest) {
						reqDumpFile = path
					} else if strings.HasSuffix(path, apidump.SuffixResponse) {
						respDumpFile = path
					}
				}
				return nil
			})
			require.NoError(t, err)
			require.NotEmpty(t, reqDumpFile, "request dump file should exist")
			require.NotEmpty(t, respDumpFile, "response dump file should exist")

			// Verify request dump contains expected HTTP request format.
			reqDumpData, err := os.ReadFile(reqDumpFile)
			require.NoError(t, err)

			// Parse the dumped HTTP request.
			dumpReq, err := http.ReadRequest(bufio.NewReader(bytes.NewReader(reqDumpData)))
			require.NoError(t, err)
			dumpBody, err := io.ReadAll(dumpReq.Body)
			require.NoError(t, err)

			// Compare requests semantically (key order may differ).
			require.JSONEq(t, string(dumpBody), string(reqBody), "request body JSON should match semantically")

			// Verify response dump contains expected HTTP response format.
			respDumpData, err := os.ReadFile(respDumpFile)
			require.NoError(t, err)

			// Parse the dumped HTTP response.
			dumpResp, err := http.ReadResponse(bufio.NewReader(bytes.NewReader(respDumpData)), nil)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, dumpResp.StatusCode)
			dumpRespBody, err := io.ReadAll(dumpResp.Body)
			require.NoError(t, err)

			// Compare responses semantically (key order may differ).
			expectedRespBody := files[fixtureNonStreamingResponse]
			require.JSONEq(t, string(expectedRespBody), string(dumpRespBody), "response body JSON should match semantically")

			recorderClient.VerifyAllInterceptionsEnded(t)
		})
	}
}

func TestAPIDumpPassthrough(t *testing.T) {
	t.Parallel()

	const responseBody = `{"object":"list","data":[{"id":"gpt-4","object":"model"}]}`

	cases := []struct {
		name         string
		providerFunc func(addr string, dumpDir string) aibridge.Provider
		requestPath  string
	}{
		{
			name: "anthropic",
			providerFunc: func(addr string, dumpDir string) aibridge.Provider {
				return provider.NewAnthropic(anthropicCfgWithAPIDump(addr, apiKey, dumpDir), nil)
			},
			requestPath: "/anthropic/v1/models",
		},
		{
			name: "openai",
			providerFunc: func(addr string, dumpDir string) aibridge.Provider {
				return provider.NewOpenAI(openaiCfgWithAPIDump(addr, apiKey, dumpDir))
			},
			requestPath: "/openai/v1/models",
		},
		{
			name: "copilot",
			providerFunc: func(addr string, dumpDir string) aibridge.Provider {
				return provider.NewCopilot(config.Copilot{BaseURL: addr, APIDumpDir: dumpDir})
			},
			requestPath: "/copilot/models",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)

			ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
			t.Cleanup(cancel)

			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.Write([]byte(responseBody))
			}))
			t.Cleanup(upstream.Close)

			dumpDir := t.TempDir()

			recorderClient := &testutil.MockRecorder{}
			prov := tc.providerFunc(upstream.URL, dumpDir)
			provs := []aibridge.Provider{prov}
			b, err := aibridge.NewRequestBridge(t.Context(), provs, recorderClient, mcp.NewServerProxyManager(nil, testTracer), logger, nil, testTracer)
			require.NoError(t, err)

			bridgeSrv := httptest.NewUnstartedServer(b)
			t.Cleanup(bridgeSrv.Close)
			bridgeSrv.Config.BaseContext = func(_ net.Listener) context.Context {
				return aibcontext.AsActor(ctx, userID, nil)
			}
			bridgeSrv.Start()

			req, err := http.NewRequestWithContext(ctx, http.MethodGet, bridgeSrv.URL+tc.requestPath, nil)
			require.NoError(t, err)

			resp, err := http.DefaultClient.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()

			// Find dump files in the passthrough directory.
			passthroughDir := filepath.Join(dumpDir, tc.name, "passthrough")
			var reqDumpFile, respDumpFile string
			err = filepath.Walk(passthroughDir, func(path string, info os.FileInfo, err error) error {
				if err != nil {
					return err
				}
				if info.IsDir() {
					return nil
				}
				if strings.HasSuffix(path, apidump.SuffixRequest) {
					reqDumpFile = path
				} else if strings.HasSuffix(path, apidump.SuffixResponse) {
					respDumpFile = path
				}
				return nil
			})
			require.NoError(t, err, "walking failed: %v", err)
			require.NotEmpty(t, reqDumpFile, "request dump file should exist")
			require.NotEmpty(t, respDumpFile, "response dump file should exist")

			// Verify request dump.
			reqDumpData, err := os.ReadFile(reqDumpFile)
			require.NoError(t, err)
			dumpReq, err := http.ReadRequest(bufio.NewReader(bytes.NewReader(reqDumpData)))
			require.NoError(t, err)
			require.Equal(t, http.MethodGet, dumpReq.Method)

			// Verify response dump.
			respDumpData, err := os.ReadFile(respDumpFile)
			require.NoError(t, err)
			dumpResp, err := http.ReadResponse(bufio.NewReader(bytes.NewReader(respDumpData)), nil)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, dumpResp.StatusCode)
			dumpRespBody, err := io.ReadAll(dumpResp.Body)
			require.NoError(t, err)
			require.JSONEq(t, responseBody, string(dumpRespBody))
		})
	}
}
