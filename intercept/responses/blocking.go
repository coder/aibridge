package responses

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/recorder"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/tidwall/sjson"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type BlockingResponsesInterceptor struct {
	responsesInterceptionBase
}

func NewBlockingInterceptor(id uuid.UUID, req *ResponsesNewParamsWrapper, reqPayload []byte, cfg config.OpenAI, model string, tracer trace.Tracer) *BlockingResponsesInterceptor {
	return &BlockingResponsesInterceptor{
		responsesInterceptionBase: responsesInterceptionBase{
			id:         id,
			req:        req,
			reqPayload: reqPayload,
			cfg:        cfg,
			model:      model,
			tracer:     tracer,
		},
	}
}

func (i *BlockingResponsesInterceptor) Setup(logger slog.Logger, recorder recorder.Recorder, mcpProxy mcp.ServerProxier) {
	i.responsesInterceptionBase.Setup(logger.Named("blocking"), recorder, mcpProxy)
}

func (i *BlockingResponsesInterceptor) Streaming() bool {
	return false
}

func (i *BlockingResponsesInterceptor) TraceAttributes(r *http.Request) []attribute.KeyValue {
	return i.responsesInterceptionBase.baseTraceAttributes(r, false)
}

func (i *BlockingResponsesInterceptor) ProcessRequest(w http.ResponseWriter, r *http.Request) error {
	ctx := r.Context()
	if err := i.validateRequest(ctx, w); err != nil {
		return err
	}

	i.injectTools()

	var (
		response    *responses.Response
		upstreamErr error
		respCopy    responseCopier
	)

	for {
		srv := i.newResponsesService()
		respCopy = responseCopier{}

		opts := i.requestOptions(&respCopy)
		opts = append(opts, option.WithRequestTimeout(time.Second*600))
		response, upstreamErr = srv.New(ctx, i.req.ResponseNewParams, opts...)

		if upstreamErr != nil {
			break
		}

		// response could be nil eg. fixtures/openai/responses/blocking/wrong_response_format.txtar
		if response == nil {
			break
		}

		// Record prompt usage on first successful response.
		i.recordUserPrompt(ctx, response.ID)
		i.recordToolUsage(ctx, response)
		i.recordTokenUsage(ctx, response)

		// Invoke any injected function calls.
		// The Responses API refers to what we call "tools" as "functions", so we keep the terminology
		// consistent in this package.
		// See https://platform.openai.com/docs/guides/function-calling

		nextRequest, err := i.handleInjectedFunctionCalls(ctx, response)
		if err != nil {
			i.logger.Error(ctx, "failed to invoke injected function call", slog.Error(err))
			i.sendCustomErr(ctx, w, http.StatusInternalServerError, fmt.Errorf("failed to invoke injected function call"))
			break
		}

		// No next request, flow is complete.
		if nextRequest == nil {
			break
		}

		i.reqPayload, err = sjson.SetBytes(i.reqPayload, "input", nextRequest.Input)
		if err != nil {
			// TODO: handle error properly.
			i.logger.Error(ctx, "failure to marshal new input in inner agentic loop", slog.Error(err))
			break
		}
	}

	if upstreamErr != nil && !respCopy.responseReceived.Load() {
		// no response received from upstream, return custom error
		i.sendCustomErr(ctx, w, http.StatusInternalServerError, upstreamErr)
	}

	err := respCopy.forwardResp(w)

	return errors.Join(upstreamErr, err)
}

// handleInjectedFunctionCalls checks for function calls that we need to handle in our inner agentic loop.
// These are functions injected by the MCP proxy.
func (i *BlockingResponsesInterceptor) handleInjectedFunctionCalls(ctx context.Context, response *responses.Response) (*ResponsesNewParamsWrapper, error) {
	if response == nil {
		return nil, fmt.Errorf("empty response")
	}

	// MCP proxy has not been configured; no way to handle injected functions.
	if i.mcpProxy == nil {
		return nil, nil
	}

	pending := i.getPendingInjectedFunctionCalls(ctx, response)
	if len(pending) == 0 {
		// No injected function calls need to be invoked, flow is complete.
		return nil, nil
	}

	// TODO: clone?
	nextRequest := i.req

	// We need to inject the output of the last response as input to the next request, in order for
	// the tool call result(s) to make sense.

	// Unset the string input, we need a list now.
	nextRequest.Input.OfString = param.Opt[string]{}

	// TODO: check for OutputText
	for _, output := range response.Output {
		// nextRequest.Input.OfInputItemList = append(nextRequest.Input.OfInputItemList, responses.ResponseInputItemParamOfOutputMessage(output.AsMessage().ToParam().Content, response.ID, responses.ResponseOutputMessageStatus(response.Status)))
		// TODO: this is a pretty janky vibe-coded func by Claude; it had the args in the wrong order, needs a LOT of verification.
		i.appendOutputToInput(nextRequest, output)
	}

	for _, fc := range pending {
		res, err := i.invokeInjectedFunc(ctx, response.ID, fc)
		if err != nil {
			i.logger.Error(ctx, "invoke injected tool", slog.Error(err), slog.F("id", fc.ID), slog.F("call_id", fc.CallID))
			// ALWAYS include response.
		}

		nextRequest.Input.OfInputItemList = append(nextRequest.Input.OfInputItemList, res)
	}

	return nextRequest, nil
}

// getPendingInjectedFunctionCalls extracts function calls from the response that are managed by MCP proxy
func (i *BlockingResponsesInterceptor) getPendingInjectedFunctionCalls(ctx context.Context, response *responses.Response) []responses.ResponseFunctionToolCall {
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

func (i *BlockingResponsesInterceptor) invokeInjectedFunc(ctx context.Context, responseID string, fc responses.ResponseFunctionToolCall) (responses.ResponseInputItemUnionParam, error) {
	tool := i.mcpProxy.GetTool(fc.Name)
	if tool == nil {
		return responses.ResponseInputItemParamOfFunctionCallOutput(fc.CallID, "error: unknown injected function"), fmt.Errorf("unknown injected tool: %q", fc.Name)
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

	return responses.ResponseInputItemParamOfFunctionCallOutput(fc.CallID, output), nil
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
