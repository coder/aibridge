package aibridge

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/google/uuid"
)

var _ Provider = &AnthropicProvider{}

// AnthropicProvider allows for interactions with the Anthropic API.
type AnthropicProvider struct {
	cfg        AnthropicConfig
	bedrockCfg *AWSBedrockConfig
}

const (
	ProviderAnthropic = "anthropic"

	routeMessages = "/anthropic/v1/messages" // https://docs.anthropic.com/en/api/messages
)

// parseCustomHeaders parses the ANTHROPIC_CUSTOM_HEADERS environment variable.
// The format is "Name: Value" with one header per line.
// Multiple lines can be separated by newlines (\n).
func parseCustomHeaders(envValue string) map[string]string {
	if envValue == "" {
		return nil
	}

	headers := make(map[string]string)
	lines := strings.Split(envValue, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Split on the first colon to handle values that contain colons (e.g., URLs)
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			// Skip malformed headers
			continue
		}

		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])

		if key != "" {
			headers[key] = value
		}
	}

	if len(headers) == 0 {
		return nil
	}

	return headers
}

func NewAnthropicProvider(cfg AnthropicConfig, bedrockCfg *AWSBedrockConfig) *AnthropicProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.anthropic.com/"
	}
	if cfg.Key == "" {
		cfg.Key = os.Getenv("ANTHROPIC_API_KEY")
	}
	if cfg.CustomHeaders == nil {
		cfg.CustomHeaders = parseCustomHeaders(os.Getenv("ANTHROPIC_CUSTOM_HEADERS"))
	}

	return &AnthropicProvider{
		cfg:        cfg,
		bedrockCfg: bedrockCfg,
	}
}

func (p *AnthropicProvider) Name() string {
	return ProviderAnthropic
}

func (p *AnthropicProvider) BridgedRoutes() []string {
	return []string{routeMessages}
}

func (p *AnthropicProvider) PassthroughRoutes() []string {
	return []string{
		"/v1/models",
		"/v1/models/", // See https://pkg.go.dev/net/http#hdr-Trailing_slash_redirection-ServeMux.
		"/v1/messages/count_tokens",
	}
}

func (p *AnthropicProvider) CreateInterceptor(w http.ResponseWriter, r *http.Request) (Interceptor, error) {
	payload, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	id := uuid.New()

	switch r.URL.Path {
	case routeMessages:
		var req MessageNewParamsWrapper
		if err := json.Unmarshal(payload, &req); err != nil {
			return nil, fmt.Errorf("failed to unmarshal request: %w", err)
		}

		if req.Stream {
			return NewAnthropicMessagesStreamingInterception(id, &req, p.cfg, p.bedrockCfg), nil
		}

		return NewAnthropicMessagesBlockingInterception(id, &req, p.cfg, p.bedrockCfg), nil
	}

	return nil, UnknownRoute
}

func (p *AnthropicProvider) BaseURL() string {
	return p.cfg.BaseURL
}

func (p *AnthropicProvider) AuthHeader() string {
	return "X-Api-Key"
}

func (p *AnthropicProvider) InjectAuthHeader(headers *http.Header) {
	if headers == nil {
		headers = &http.Header{}
	}

	headers.Set(p.AuthHeader(), p.cfg.Key)
}

func getAnthropicErrorResponse(err error) *AnthropicErrorResponse {
	var apierr *anthropic.Error
	if !errors.As(err, &apierr) {
		return nil
	}

	msg := apierr.Error()
	typ := string(constant.ValueOf[constant.APIError]())

	var detail *anthropic.APIErrorObject
	if field, ok := apierr.JSON.ExtraFields["error"]; ok {
		_ = json.Unmarshal([]byte(field.Raw()), &detail)
	}
	if detail != nil {
		msg = detail.Message
		typ = string(detail.Type)
	}

	return &AnthropicErrorResponse{
		ErrorResponse: &anthropic.ErrorResponse{
			Error: anthropic.ErrorObjectUnion{
				Message: msg,
				Type:    typ,
			},
			Type: constant.ValueOf[constant.Error](),
		},
		StatusCode: apierr.StatusCode,
	}
}

var _ error = &AnthropicErrorResponse{}

type AnthropicErrorResponse struct {
	*anthropic.ErrorResponse

	StatusCode int `json:"-"`
}

func newAnthropicErr(msg error) *AnthropicErrorResponse {
	return &AnthropicErrorResponse{
		ErrorResponse: &shared.ErrorResponse{
			Error: shared.ErrorObjectUnion{
				Message: msg.Error(),
				Type:    "error",
			},
		},
	}
}

func (a *AnthropicErrorResponse) Error() string {
	if a.ErrorResponse == nil {
		return ""
	}
	return a.ErrorResponse.Error.Message
}
