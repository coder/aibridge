package responses

import (
	"fmt"

	"github.com/openai/openai-go/v3/responses"
	"github.com/tidwall/gjson"
)

// ResponsesNewParamsWrapper exists because the "stream" param is not included
// in responses.ResponseNewParams.
type ResponsesNewParamsWrapper struct {
	responses.ResponseNewParams
	Stream bool `json:"stream,omitempty"`
}

func (c *ResponsesNewParamsWrapper) UnmarshalJSON(raw []byte) error {
	err := c.ResponseNewParams.UnmarshalJSON(raw)
	if err != nil {
		return fmt.Errorf("failed to unmarshal response params: %w", err)
	}

	c.Stream = false
	if stream := gjson.Get(string(raw), "stream"); stream.Bool() {
		c.Stream = stream.Bool()
	}
	return nil
}
