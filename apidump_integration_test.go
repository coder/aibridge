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
		providerName      string
		providersFunc     func(addr, dumpDir string) []aibridge.Provider
		createRequestFunc createRequestFunc
	}{
		{
			name:         config.ProviderAnthropic,
			fixture:      fixtures.AntSimple,
			providerName: config.ProviderAnthropic,
			providersFunc: func(addr, dumpDir string) []aibridge.Provider {
				return []aibridge.Provider{provider.NewAnthropic(anthropicCfgWithAPIDump(addr, apiKey, dumpDir), nil)}
			},
			createRequestFunc: createAnthropicMessagesReq,
		},
		{
			name:         config.ProviderOpenAI,
			fixture:      fixtures.OaiChatSimple,
			providerName: config.ProviderOpenAI,
			providersFunc: func(addr, dumpDir string) []aibridge.Provider {
				return []aibridge.Provider{provider.NewOpenAI(openaiCfgWithAPIDump(addr, apiKey, dumpDir))}
			},
			createRequestFunc: createOpenAIChatCompletionsReq,
		},
		{
			name:         config.ProviderOpenAI,
			fixture:      fixtures.OaiResponsesBlockingSimple,
			providerName: config.ProviderOpenAI,
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
