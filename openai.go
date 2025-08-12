package aibridge

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/anthropics/anthropic-sdk-go/shared"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/coder/aibridge/utils"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
)

// ChatCompletionNewParamsWrapper exists because the "stream" param is not included in openai.ChatCompletionNewParams.
type ChatCompletionNewParamsWrapper struct {
	openai.ChatCompletionNewParams `json:""`
	Stream                         bool `json:"stream,omitempty"`
}

func (c ChatCompletionNewParamsWrapper) MarshalJSON() ([]byte, error) {
	type shadow ChatCompletionNewParamsWrapper
	return param.MarshalWithExtras(c, (*shadow)(&c), map[string]any{
		"stream": c.Stream,
	})
}

func (c *ChatCompletionNewParamsWrapper) UnmarshalJSON(raw []byte) error {
	err := c.ChatCompletionNewParams.UnmarshalJSON(raw)
	if err != nil {
		return err
	}

	if stream := utils.ExtractJSONField[bool](raw, "stream"); stream {
		c.Stream = stream
		if c.Stream {
			c.ChatCompletionNewParams.StreamOptions = openai.ChatCompletionStreamOptionsParam{
				IncludeUsage: openai.Bool(true), // Always include usage when streaming.
			}
		} else {
			c.ChatCompletionNewParams.StreamOptions = openai.ChatCompletionStreamOptionsParam{}
		}
	} else {
		c.ChatCompletionNewParams.StreamOptions = openai.ChatCompletionStreamOptionsParam{}
	}

	return nil
}

func (c *ChatCompletionNewParamsWrapper) LastUserPrompt() (*string, error) {
	if c == nil {
		return nil, errors.New("nil struct")
	}

	if len(c.Messages) == 0 {
		return nil, errors.New("no messages")
	}

	var msg *openai.ChatCompletionUserMessageParam
	for i := len(c.Messages) - 1; i >= 0; i-- {
		m := c.Messages[i]
		if m.OfUser != nil {
			msg = m.OfUser
			break
		}
	}

	if msg == nil {
		return nil, nil
	}

	return utils.PtrTo(strings.TrimSpace(
		msg.Content.OfString.String(),
	)), nil
}

func sumUsage(ref, in openai.CompletionUsage) openai.CompletionUsage {
	return openai.CompletionUsage{
		CompletionTokens: ref.CompletionTokens + in.CompletionTokens,
		PromptTokens:     ref.PromptTokens + in.PromptTokens,
		TotalTokens:      ref.TotalTokens + in.TotalTokens,
		CompletionTokensDetails: openai.CompletionUsageCompletionTokensDetails{
			AcceptedPredictionTokens: ref.CompletionTokensDetails.AcceptedPredictionTokens + in.CompletionTokensDetails.AcceptedPredictionTokens,
			AudioTokens:              ref.CompletionTokensDetails.AudioTokens + in.CompletionTokensDetails.AudioTokens,
			ReasoningTokens:          ref.CompletionTokensDetails.ReasoningTokens + in.CompletionTokensDetails.ReasoningTokens,
			RejectedPredictionTokens: ref.CompletionTokensDetails.RejectedPredictionTokens + in.CompletionTokensDetails.RejectedPredictionTokens,
		},
		PromptTokensDetails: openai.CompletionUsagePromptTokensDetails{
			AudioTokens:  ref.PromptTokensDetails.AudioTokens + in.PromptTokensDetails.AudioTokens,
			CachedTokens: ref.PromptTokensDetails.CachedTokens + in.PromptTokensDetails.CachedTokens,
		},
	}
}

func getOpenAIErrorResponse(err error) *OpenAIErrorResponse {
	var apierr *openai.Error
	if !errors.As(err, &apierr) {
		return nil
	}

	msg := apierr.Error()
	typ := string(constant.ValueOf[constant.APIError]())

	var detail *shared.APIErrorObject
	if field, ok := apierr.JSON.ExtraFields["error"]; ok {
		_ = json.Unmarshal([]byte(field.Raw()), &detail)
	}
	if detail != nil {
		msg = detail.Message
		typ = string(detail.Type)
	}

	return &OpenAIErrorResponse{
		ErrorResponse: &shared.ErrorResponse{
			Error: shared.ErrorObjectUnion{
				Message: msg,
				Type:    typ,
			},
			Type: constant.ValueOf[constant.Error](),
		},
		StatusCode: apierr.StatusCode,
	}
}

var _ error = &OpenAIErrorResponse{}

type OpenAIErrorResponse struct {
	*shared.ErrorResponse

	StatusCode int `json:"-"`
}

func newOpenAIErr(msg error) *OpenAIErrorResponse {
	return &OpenAIErrorResponse{
		ErrorResponse: &shared.ErrorResponse{
			Error: shared.ErrorObjectUnion{
				Message: msg.Error(),
				Type:    "error",
			},
		},
	}
}

func (a *OpenAIErrorResponse) Error() string {
	if a.ErrorResponse == nil {
		return ""
	}
	return a.ErrorResponse.Error.Message
}
