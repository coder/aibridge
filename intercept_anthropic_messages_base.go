package aibridge

import (
	"context"
	"net/http"
	"net/url"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/bedrock"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/coder/aibridge/mcp"
	"github.com/google/uuid"

	"cdr.dev/slog"
)

type AnthropicMessagesInterceptionBase struct {
	id  uuid.UUID
	req *MessageNewParamsWrapper

	cfg        AnthropicConfig
	bedrockCfg *AWSBedrockConfig

	logger slog.Logger

	recorder Recorder
	mcpProxy mcp.ServerProxier
}

func (i *AnthropicMessagesInterceptionBase) ID() uuid.UUID {
	return i.id
}

func (i *AnthropicMessagesInterceptionBase) Setup(logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier) {
	i.logger = logger
	i.recorder = recorder
	i.mcpProxy = mcpProxy
}

func (i *AnthropicMessagesInterceptionBase) Model() string {
	if i.req == nil {
		return "coder-aibridge-unknown"
	}

	if i.bedrockCfg != nil {
		model := i.bedrockCfg.Model
		if i.isSmallFastModel() {
			model = i.bedrockCfg.SmallFastModel
		}
		return model
	}

	return string(i.req.Model)
}

func (i *AnthropicMessagesInterceptionBase) injectTools() {
	if i.req == nil || i.mcpProxy == nil {
		return
	}

	// Inject tools.
	for _, tool := range i.mcpProxy.ListTools() {
		i.req.Tools = append(i.req.Tools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: tool.Params,
					Required:   tool.Required,
				},
				Name:        tool.ID,
				Description: anthropic.String(tool.Description),
				Type:        anthropic.ToolTypeCustom,
			},
		})
	}

	// Note: Parallel tool calls are disabled to avoid tool_use/tool_result block mismatches.
	i.req.ToolChoice = anthropic.ToolChoiceUnionParam{
		OfAny: &anthropic.ToolChoiceAnyParam{
			Type:                   "auto",
			DisableParallelToolUse: anthropic.Bool(true),
		},
	}
}

// isSmallFastModel checks if the model is a small/fast model (Haiku 3.5).
// These models are optimized for tasks like code autocomplete and other small, quick operations.
// See `ANTHROPIC_SMALL_FAST_MODEL`: https://docs.anthropic.com/en/docs/claude-code/settings#environment-variables
// https://docs.claude.com/en/docs/claude-code/costs#background-token-usage
func (i *AnthropicMessagesInterceptionBase) isSmallFastModel() bool {
	return strings.Contains(string(i.req.Model), "haiku")
}

func (i *AnthropicMessagesInterceptionBase) newAnthropicClient(cfg AnthropicConfig, bedrockCfg *AWSBedrockConfig, opts ...option.RequestOption) anthropic.Client {
	opts = append(opts, option.WithAPIKey(cfg.Key))
	opts = append(opts, option.WithBaseURL(cfg.BaseURL))

	if i.bedrockCfg != nil {
		// TODO: deadline.
		opts = append(opts, i.withAWSBedrock(context.Background(), i.bedrockCfg))
		i.augmentRequestForBedrock()

		// If an endpoint override is set (for testing), add a custom HTTP client AFTER the bedrock config
		// This overrides any HTTP client set by the bedrock middleware
		if i.bedrockCfg.EndpointOverride != "" {
			opts = append(opts, option.WithHTTPClient(&http.Client{
				Transport: &redirectTransport{
					base:          http.DefaultTransport,
					redirectToURL: i.bedrockCfg.EndpointOverride,
				},
			}))
		}
	}

	return anthropic.NewClient(opts...)
}

func (i *AnthropicMessagesInterceptionBase) withAWSBedrock(ctx context.Context, cfg *AWSBedrockConfig) option.RequestOption {
	opts := []func(*config.LoadOptions) error{
		config.WithRegion(cfg.Region),
		config.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(
				cfg.AccessKey,
				cfg.AccessKeySecret,
				"",
			),
		),
	}

	// Load the AWS config with our options, then pass it to bedrock.WithConfig
	// This is the recommended approach when you have custom config settings
	awsCfg, err := config.LoadDefaultConfig(ctx, opts...)
	if err != nil {
		// If config loading fails, fall back to WithLoadDefaultConfig with options
		return bedrock.WithLoadDefaultConfig(ctx, opts...)
	}

	return bedrock.WithConfig(awsCfg)
}

// augmentRequestForBedrock will change the model used for the request since AWS Bedrock doesn't support
// Anthropics' model names.
func (i *AnthropicMessagesInterceptionBase) augmentRequestForBedrock() {
	if i.bedrockCfg == nil {
		return
	}

	i.req.MessageNewParams.Model = anthropic.Model(i.Model())
}

// redirectTransport is an HTTP RoundTripper that redirects requests to a different endpoint.
// This is useful for testing when we need to redirect AWS Bedrock requests to a mock server.
type redirectTransport struct {
	base          http.RoundTripper
	redirectToURL string
}

func (t *redirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Parse the redirect URL
	redirectURL, err := url.Parse(t.redirectToURL)
	if err != nil {
		return nil, err
	}

	// Redirect the request to the mock server
	req.URL.Scheme = redirectURL.Scheme
	req.URL.Host = redirectURL.Host
	req.Host = redirectURL.Host

	return t.base.RoundTrip(req)
}
