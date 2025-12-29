package chatcompletions

import (
	"errors"

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

func (c *ChatCompletionNewParamsWrapper) lastUserPrompt() (*string, error) {
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
