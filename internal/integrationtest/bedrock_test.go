package integrationtest

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/fixtures"
	"github.com/coder/aibridge/provider"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func TestAWSBedrockIntegration(t *testing.T) {
	t.Parallel()

	t.Run("invalid config", func(t *testing.T) {
		t.Parallel()

		ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
		t.Cleanup(cancel)

		// Invalid bedrock config - missing region & base url
		bedrockCfg := &config.AWSBedrock{
			Region:          "",
			AccessKey:       "test-key",
			AccessKeySecret: "test-secret",
			Model:           "test-model",
			SmallFastModel:  "test-haiku",
		}

		ts := newBridgeTestServer(t, ctx, "http://unused",
			withCustomProvider(provider.NewAnthropic(anthropicCfg("http://unused", apiKey), bedrockCfg)),
			withLogger(newLogger(t)),
		)

		req := ts.newRequest(t, pathAnthropicMessages, fixtures.Request(t, fixtures.AntSingleBuiltinTool))
		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		defer resp.Body.Close()

		require.Equal(t, http.StatusInternalServerError, resp.StatusCode)
		body, err := io.ReadAll(resp.Body)
		require.NoError(t, err)
		require.Contains(t, string(body), "create anthropic client")
		require.Contains(t, string(body), "region or base url required")
	})

	t.Run("/v1/messages", func(t *testing.T) {
		for _, streaming := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/streaming=%v", t.Name(), streaming), func(t *testing.T) {
				t.Parallel()

				ctx, cancel := context.WithTimeout(t.Context(), time.Second*30)
				t.Cleanup(cancel)

				fix := fixtures.Parse(t, fixtures.AntSingleBuiltinTool)
				upstream := newMockUpstream(t, ctx, newFixtureResponse(fix))

				// We define region here to validate that with Region & BaseURL defined, the latter takes precedence.
				bedrockCfg := &config.AWSBedrock{
					Region:          "us-west-2",
					AccessKey:       "test-access-key",
					AccessKeySecret: "test-secret-key",
					Model:           "danthropic",      // This model should override the request's given one.
					SmallFastModel:  "danthropic-mini", // Unused but needed for validation.
					BaseURL:         upstream.URL,      // Use the mock server.
				}

				ts := newBridgeTestServer(t, ctx, upstream.URL,
					withCustomProvider(provider.NewAnthropic(anthropicCfg(upstream.URL, apiKey), bedrockCfg)),
					withLogger(newLogger(t)),
				)

				// Make API call to aibridge for Anthropic /v1/messages, which will be routed via AWS Bedrock.
				// We override the AWS Bedrock client to route requests through our mock server.
				reqBody, err := sjson.SetBytes(fix.Request(), "stream", streaming)
				require.NoError(t, err)
				req := ts.newRequest(t, pathAnthropicMessages, reqBody)
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
				received := upstream.receivedRequests()
				require.Len(t, received, 1)

				// The Anthropic SDK's Bedrock middleware extracts "model" and "stream"
				// from the JSON body and encodes them in the URL path.
				// See: https://github.com/anthropics/anthropic-sdk-go/blob/4d669338f2041f3c60640b6dd317c4895dc71cd4/bedrock/bedrock.go#L247-L248
				pathParts := strings.Split(received[0].Path, "/")
				require.True(t, len(pathParts) >= 3 && pathParts[1] == "model", "unexpected path: %s", received[0].Path)
				require.Equal(t, bedrockCfg.Model, pathParts[2])
				require.False(t, gjson.GetBytes(received[0].Body, "model").Exists(), "model should be stripped from body")
				require.False(t, gjson.GetBytes(received[0].Body, "stream").Exists(), "stream should be stripped from body")

				interceptions := ts.Recorder.RecordedInterceptions()
				require.Len(t, interceptions, 1)
				require.Equal(t, interceptions[0].Model, bedrockCfg.Model)
				ts.Recorder.VerifyAllInterceptionsEnded(t)
			})
		}
	})
}
