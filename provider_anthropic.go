package aibridge

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	aibtrace "github.com/coder/aibridge/aibtrace"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
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

func NewAnthropicProvider(cfg AnthropicConfig, bedrockCfg *AWSBedrockConfig) *AnthropicProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.anthropic.com/"
	}
	if cfg.Key == "" {
		cfg.Key = os.Getenv("ANTHROPIC_API_KEY")
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

func (p *AnthropicProvider) CreateInterceptor(tracer trace.Tracer, w http.ResponseWriter, r *http.Request) (_ Interceptor, outErr error) {
	id := uuid.New()
	_, span := tracer.Start(r.Context(), "Intercept.CreateInterceptor")
	defer aibtrace.EndSpanErr(span, &outErr)

	payload, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	switch r.URL.Path {
	case routeMessages:
		var req MessageNewParamsWrapper
		if err := json.Unmarshal(payload, &req); err != nil {
			return nil, fmt.Errorf("failed to unmarshal request: %w", err)
		}

		var interceptor Interceptor
		if req.Stream {
			interceptor = NewAnthropicMessagesStreamingInterception(id, &req, p.cfg, p.bedrockCfg, tracer)
		} else {
			interceptor = NewAnthropicMessagesBlockingInterception(id, &req, p.cfg, p.bedrockCfg, tracer)
		}
		span.SetAttributes(interceptor.TraceAttributes(r.Context())...)
		return interceptor, nil
	}

	span.SetStatus(codes.Error, "unknown route: "+r.URL.Path)
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
