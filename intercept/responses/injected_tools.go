package responses

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/recorder"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/tidwall/sjson"
)

func (i *responsesInterceptionBase) injectTools() {
	if i.req == nil || i.mcpProxy == nil {
		return
	}

	tools := i.mcpProxy.ListTools()
	if len(tools) == 0 {
		return
	}

	// Inject tools.
	for _, tool := range i.mcpProxy.ListTools() {
		params := map[string]any{
			"type":       "object",
			"properties": tool.Params,
			// "additionalProperties": false, // Only relevant when strict=true.
		}

		// Otherwise the request fails with "None is not of type 'array'" if a nil slice is given.
		if len(tool.Required) > 0 {
			// Must list ALL properties when strict=true.
			params["required"] = tool.Required
		}

		fn := responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        tool.ID,
				Strict:      openai.Bool(false), // TODO: configurable.
				Description: openai.String(tool.Description),
				Parameters:  params,
			},
		}

		i.req.Tools = append(i.req.Tools, fn)
	}

	var err error
	i.reqPayload, err = sjson.SetBytes(i.reqPayload, "tools", i.req.Tools)
	if err != nil {
		i.logger.Warn(context.Background(), "failed to set tools", slog.Error(err))
	}
}

// disableParallelToolCalls disables parallel tool calls, to simplify the inner agentic loop.
// This is best-effort, and failing to set this flag does not fail the request.
// TODO: implement parallel tool calls.
func (i *responsesInterceptionBase) disableParallelToolCalls() {
	// Disable parallel tool calls to simplify inner agentic loop; best-effort.
	if len(i.req.Tools) > 0 {
		var err error
		i.reqPayload, err = sjson.SetBytes(i.reqPayload, "parallel_tool_calls", false)
		if err != nil {
			i.logger.Warn(context.Background(), "failed to disable parallel_tool_calls", slog.Error(err))
		}
	}
}

// handleInjectedToolCalls checks for function calls that we need to handle in our inner agentic loop.
// These are functions injected by the MCP proxy.
// Returns a list of tool call results.
func (i *BlockingResponsesInterceptor) handleInjectedToolCalls(ctx context.Context, pending []responses.ResponseFunctionToolCall, response *responses.Response) ([]responses.ResponseInputItemUnionParam, error) {
	if response == nil {
		return nil, fmt.Errorf("empty response")
	}

	// MCP proxy has not been configured; no way to handle injected functions.
	if i.mcpProxy == nil {
		return nil, nil
	}

	var results []responses.ResponseInputItemUnionParam
	for _, fc := range pending {
		results = append(results, i.invokeInjectedTool(ctx, response.ID, fc))
	}

	return results, nil
}

// prepareRequestForAgenticLoop prepares the request by setting the output of the given
// response as input to the next request, in order for the tool call result(s) to make function correctly.
func (i *BlockingResponsesInterceptor) prepareRequestForAgenticLoop(response *responses.Response) {
	// Unset the string input; we need a list now.
	i.req.Input.OfString = param.Opt[string]{}

	// OutputText is also available, but by definition the trigger for a function call is not a simple
	// text response from the model.
	for _, output := range response.Output {
		i.appendOutputToInput(i.req, output)
	}
}

// getPendingInjectedToolCalls extracts function calls from the response that are managed by MCP proxy
func (i *BlockingResponsesInterceptor) getPendingInjectedToolCalls(ctx context.Context, response *responses.Response) []responses.ResponseFunctionToolCall {
	var calls []responses.ResponseFunctionToolCall

	for _, item := range response.Output {
		if item.Type != string(constant.ValueOf[constant.FunctionCall]()) {
			continue
		}

		// Injected functions are defined by MCP, and MCP tools have to have a schema
		// for their inputs. The Responses API also supports "Custom Tools":
		// https://platform.openai.com/docs/guides/function-calling#custom-tools
		// These are like regular functions but their inputs are not schematized.
		// As such, custom tools are not considered here.
		fc := item.AsFunctionCall()

		// Check if this is a tool managed by our MCP proxy
		if i.mcpProxy != nil && i.mcpProxy.GetTool(fc.Name) != nil {
			calls = append(calls, fc)
		} else {
			// Record tool usage for non-managed tools
			_ = i.recorder.RecordToolUsage(ctx, &recorder.ToolUsageRecord{
				InterceptionID: i.ID().String(),
				MsgID:          response.ID,
				Tool:           fc.Name,
				Args:           fc.Arguments,
				Injected:       false,
			})
		}
	}

	return calls
}

func (i *BlockingResponsesInterceptor) invokeInjectedTool(ctx context.Context, responseID string, fc responses.ResponseFunctionToolCall) responses.ResponseInputItemUnionParam {
	tool := i.mcpProxy.GetTool(fc.Name)
	if tool == nil {
		return responses.ResponseInputItemParamOfFunctionCallOutput(fc.CallID, fmt.Sprintf("error: unknown injected function %q", fc.ID))
	}

	args := i.unmarshalArgs(fc.Arguments)
	res, err := tool.Call(ctx, args, i.tracer)
	_ = i.recorder.RecordToolUsage(ctx, &recorder.ToolUsageRecord{
		InterceptionID:  i.ID().String(),
		MsgID:           responseID,
		ServerURL:       &tool.ServerURL,
		Tool:            tool.Name,
		Args:            args,
		Injected:        true,
		InvocationError: err,
	})

	var output string
	if err != nil {
		// Results have no fixed structure; if an error occurs, we can just pass back the error.
		// https://platform.openai.com/docs/guides/function-calling?strict-mode=enabled#formatting-results
		output = fmt.Sprintf("invocation error: %q", err.Error())
	} else {
		var out strings.Builder
		if encErr := json.NewEncoder(&out).Encode(res); encErr != nil {
			i.logger.Warn(ctx, "failed to encode tool response", slog.Error(encErr))
			output = fmt.Sprintf("result encode error: %q", encErr.Error())
		} else {
			output = out.String()
		}
	}

	return responses.ResponseInputItemParamOfFunctionCallOutput(fc.CallID, output)
}

// appendOutputToInput converts a response output item to an input item and appends it to the
// request's input list. This is used in agentic loops where we need to feed the model's output
// back as input for the next iteration (e.g., when processing tool call results).
//
// The conversion uses the openai-go library's ToParam() methods where available, which leverage
// param.Override() with raw JSON to preserve all fields. For types without ToParam(), we use
// the ResponseInputItemParamOf* helper functions.
func (i *BlockingResponsesInterceptor) appendOutputToInput(req *ResponsesNewParamsWrapper, item responses.ResponseOutputItemUnion) {
	var inputItem responses.ResponseInputItemUnionParam

	switch item.Type {
	case string(constant.ValueOf[constant.Message]()):
		p := item.AsMessage().ToParam()
		inputItem = responses.ResponseInputItemUnionParam{OfOutputMessage: &p}

	case string(constant.ValueOf[constant.FileSearchCall]()):
		p := item.AsFileSearchCall().ToParam()
		inputItem = responses.ResponseInputItemUnionParam{OfFileSearchCall: &p}

	case string(constant.ValueOf[constant.FunctionCall]()):
		p := item.AsFunctionCall().ToParam()
		inputItem = responses.ResponseInputItemUnionParam{OfFunctionCall: &p}

	case string(constant.ValueOf[constant.WebSearchCall]()):
		p := item.AsWebSearchCall().ToParam()
		inputItem = responses.ResponseInputItemUnionParam{OfWebSearchCall: &p}

	case "computer_call": // No constant.ComputerCall type exists
		p := item.AsComputerCall().ToParam()
		inputItem = responses.ResponseInputItemUnionParam{OfComputerCall: &p}

	case string(constant.ValueOf[constant.Reasoning]()):
		p := item.AsReasoning().ToParam()
		inputItem = responses.ResponseInputItemUnionParam{OfReasoning: &p}

	case string(constant.ValueOf[constant.Compaction]()):
		c := item.AsCompaction()
		inputItem = responses.ResponseInputItemParamOfCompaction(c.EncryptedContent)

	case string(constant.ValueOf[constant.ImageGenerationCall]()):
		c := item.AsImageGenerationCall()
		inputItem = responses.ResponseInputItemParamOfImageGenerationCall(c.ID, c.Result, c.Status)

	case string(constant.ValueOf[constant.CodeInterpreterCall]()):
		p := item.AsCodeInterpreterCall().ToParam()
		inputItem = responses.ResponseInputItemUnionParam{OfCodeInterpreterCall: &p}

	case "custom_tool_call": // No constant.CustomToolCall type exists
		p := item.AsCustomToolCall().ToParam()
		inputItem = responses.ResponseInputItemUnionParam{OfCustomToolCall: &p}

	// Output-only types that don't have direct input equivalents or are handled separately:
	// - local_shell_call, shell_call, shell_call_output: Shell tool outputs
	// - apply_patch_call, apply_patch_call_output: Apply patch outputs
	// - mcp_call, mcp_list_tools, mcp_approval_request: MCP-specific outputs
	default:
		i.logger.Debug(context.Background(), "skipping output item type for input", slog.F("type", item.Type))
		return
	}

	req.Input.OfInputItemList = append(req.Input.OfInputItemList, inputItem)
}

// unmarshalArgs unmarshals JSON arguments string into a map
func (i *BlockingResponsesInterceptor) unmarshalArgs(in string) (args recorder.ToolArgs) {
	if len(strings.TrimSpace(in)) == 0 {
		return args
	}

	if err := json.Unmarshal([]byte(in), &args); err != nil {
		i.logger.Warn(context.Background(), "failed to unmarshal tool args", slog.Error(err))
	}

	return args
}
