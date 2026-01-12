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

func (r *ResponsesNewParamsWrapper) UnmarshalJSON(raw []byte) error {
	err := r.ResponseNewParams.UnmarshalJSON(raw)
	if err != nil {
		return fmt.Errorf("failed to unmarshal response params: %w", err)
	}

	r.Stream = false
	if stream := gjson.Get(string(raw), "stream"); stream.Bool() {
		r.Stream = stream.Bool()
	}
	return nil
}
