package aibridge

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

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

func (i *AnthropicMessagesInterceptionBase) newMessagesService(ctx context.Context, opts ...option.RequestOption) (anthropic.MessageService, error) {
	opts = append(opts, option.WithAPIKey(i.cfg.Key))
	opts = append(opts, option.WithBaseURL(i.cfg.BaseURL))

	if i.bedrockCfg != nil {
		ctx, cancel := context.WithTimeout(ctx, time.Second*30)
		defer cancel()
		bedrockOpt, err := i.withAWSBedrock(ctx, i.bedrockCfg)
		if err != nil {
			return anthropic.MessageService{}, err
		}
		opts = append(opts, bedrockOpt)
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

	return anthropic.NewMessageService(opts...), nil
}

func (i *AnthropicMessagesInterceptionBase) withAWSBedrock(ctx context.Context, cfg *AWSBedrockConfig) (option.RequestOption, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config given")
	}
	if cfg.Region == "" {
		return nil, fmt.Errorf("region required")
	}
	if cfg.AccessKey == "" {
		return nil, fmt.Errorf("access key required")
	}
	if cfg.AccessKeySecret == "" {
		return nil, fmt.Errorf("access key secret required")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("model required")
	}
	if cfg.SmallFastModel == "" {
		return nil, fmt.Errorf("small fast model required")
	}

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

	awsCfg, err := config.LoadDefaultConfig(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to load AWS Bedrock config: %w", err)
	}

	return bedrock.WithConfig(awsCfg), nil
}

// augmentRequestForBedrock will change the model used for the request since AWS Bedrock doesn't support
// Anthropics' model names.
func (i *AnthropicMessagesInterceptionBase) augmentRequestForBedrock() {
	if i.bedrockCfg == nil {
		return
	}

	i.req.MessageNewParams.Model = anthropic.Model(i.Model())
}

// writeUpstreamError marshals and writes a given error.
func (i *AnthropicMessagesInterceptionBase) writeUpstreamError(w http.ResponseWriter, antErr *AnthropicErrorResponse) {
	if antErr == nil {
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(antErr.StatusCode)

	out, err := json.Marshal(antErr)
	if err != nil {
		i.logger.Warn(context.Background(), "failed to marshal upstream error", slog.Error(err), slog.F("error_payload", slog.F("%+v", antErr)))
		// Response has to match expected format.
		// See https://docs.claude.com/en/api/errors#error-shapes.
		_, _ = w.Write([]byte(fmt.Sprintf(`{
	"type":"error",
	"error": {
		"type": "error",
		"message":"error marshaling upstream error"
	},
	"request_id": "%s"
}`, i.ID().String())))
	} else {
		_, _ = w.Write(out)
	}
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
