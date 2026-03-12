package responses

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	reqPathBackground        = "background"
	reqPathInput             = "input"
	reqPathParallelToolCalls = "parallel_tool_calls"
	reqPathStream            = "stream"
	reqPathTools             = "tools"
)

var reqPathModel = string(constant.ValueOf[constant.Model]())

// ResponsesRequestPayload is raw JSON bytes of a Responses API request.
// Methods provide package-specific reads and rewrites while preserving the
// original body for upstream pass-through.
type ResponsesRequestPayload []byte

func NewResponsesRequestPayload(raw []byte) (ResponsesRequestPayload, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, fmt.Errorf("responses empty request body")
	}
	if !json.Valid(raw) {
		return nil, fmt.Errorf("responses invalid JSON request body")
	}

	return ResponsesRequestPayload(raw), nil
}

func (p ResponsesRequestPayload) Stream() bool {
	return gjson.GetBytes(p, reqPathStream).Bool()
}

func (p ResponsesRequestPayload) model() string {
	return gjson.GetBytes(p, reqPathModel).String()
}

func (p ResponsesRequestPayload) background() bool {
	return gjson.GetBytes(p, reqPathBackground).Bool()
}

func (p ResponsesRequestPayload) withInjectedTools(injected []responses.ToolUnionParam) (ResponsesRequestPayload, error) {
	if len(injected) == 0 {
		return p, nil
	}

	existing, err := p.existingToolItems()
	if err != nil {
		return p, err
	}

	allTools := make([]any, 0, len(existing)+len(injected))
	for _, item := range existing {
		allTools = append(allTools, item)
	}
	for _, tool := range injected {
		allTools = append(allTools, tool)
	}

	return p.set(reqPathTools, allTools)
}

func (p ResponsesRequestPayload) withParallelToolCallsDisabled() (ResponsesRequestPayload, error) {
	existing, err := p.existingToolItems()
	if err != nil {
		return p, err
	}
	if len(existing) == 0 {
		return p, nil
	}

	return p.set(reqPathParallelToolCalls, false)
}

func (p ResponsesRequestPayload) withAppendedInputItems(items []responses.ResponseInputItemUnionParam) (ResponsesRequestPayload, error) {
	if len(items) == 0 {
		return p, nil
	}

	existing, err := p.inputItemsForRewrite()
	if err != nil {
		return p, err
	}

	allInput := make([]any, 0, len(existing)+len(items))
	allInput = append(allInput, existing...)
	for _, item := range items {
		allInput = append(allInput, item)
	}

	return p.set(reqPathInput, allInput)
}

func (p ResponsesRequestPayload) inputItemsForRewrite() ([]any, error) {
	input := gjson.GetBytes(p, reqPathInput)
	if !input.Exists() || input.Type == gjson.Null {
		return []any{}, nil
	}

	if input.Type == gjson.String {
		return []any{responses.ResponseInputItemParamOfMessage(input.String(), responses.EasyInputMessageRoleUser)}, nil
	}

	if !input.IsArray() {
		return nil, fmt.Errorf("unsupported input type: %s", input.Type)
	}

	items := input.Array()
	existing := make([]any, 0, len(items))
	for _, item := range items {
		existing = append(existing, json.RawMessage(item.Raw))
	}

	return existing, nil
}

func (p ResponsesRequestPayload) existingToolItems() ([]json.RawMessage, error) {
	tools := gjson.GetBytes(p, reqPathTools)
	if !tools.Exists() {
		return nil, nil
	}
	if !tools.IsArray() {
		return nil, fmt.Errorf("unsupported tools type: %s", tools.Type)
	}

	items := tools.Array()
	existing := make([]json.RawMessage, 0, len(items))
	for _, item := range items {
		existing = append(existing, json.RawMessage(item.Raw))
	}

	return existing, nil
}

func (p ResponsesRequestPayload) set(path string, value any) (ResponsesRequestPayload, error) {
	b, err := sjson.SetBytes(p, path, value)
	return ResponsesRequestPayload(b), err
}
