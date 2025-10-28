package aibridge

import (
	"encoding/json"
	"errors"

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
	MaxCompletionTokens            *int `json:"max_completion_tokens,omitempty"`
}

func (c ChatCompletionNewParamsWrapper) MarshalJSON() ([]byte, error) {
	type shadow ChatCompletionNewParamsWrapper
	extras := map[string]any{
		"stream": c.Stream,
	}
	if c.MaxCompletionTokens != nil {
		extras["max_completion_tokens"] = *c.MaxCompletionTokens
	}
	return param.MarshalWithExtras(c, (*shadow)(&c), extras)
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

	// Extract max_completion_tokens if present
	if maxCompletionTokens := utils.ExtractJSONField[float64](raw, "max_completion_tokens"); maxCompletionTokens > 0 {
		tokens := int(maxCompletionTokens)
		c.MaxCompletionTokens = &tokens
		// Set it in the underlying params as well
		c.ChatCompletionNewParams.MaxCompletionTokens = openai.Int(int64(tokens))
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

	// We only care if the last message was issued by a user.
	msg := c.Messages[len(c.Messages)-1]
	if msg.OfUser == nil {
		return nil, nil
	}

	if msg.OfUser.Content.OfString.String() != "" {
		return utils.PtrTo(msg.OfUser.Content.OfString.String()), nil
	}

	// Walk backwards on "user"-initiated message content. Clients often inject
	// content ahead of the actual prompt to provide context to the model,
	// so the last item in the slice is most likely the user's prompt.
	for i := len(msg.OfUser.Content.OfArrayOfContentParts) - 1; i >= 0; i-- {
		// Only text content is supported currently.
		if textContent := msg.OfUser.Content.OfArrayOfContentParts[i].OfText; textContent != nil {
			return &textContent.Text, nil
		}
	}

	return nil, nil
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

// calculateActualInputTokenUsage accounts for cached tokens which are included in [openai.CompletionUsage].PromptTokens.
func calculateActualInputTokenUsage(in openai.CompletionUsage) int64 {
	// Input *includes* the cached tokens, so we subtract them here to reflect actual input token usage.
	// The original value can be reconstructed by referencing the "prompt_cached" field in metadata.
	// See https://platform.openai.com/docs/api-reference/usage/completions_object#usage/completions_object-input_tokens.
	return in.PromptTokens /* The aggregated number of text input tokens used, including cached tokens. */ -
		in.PromptTokensDetails.CachedTokens /* The aggregated number of text input tokens that has been cached from previous requests. */
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
