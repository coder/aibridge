package responses

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/config"
	aibcontext "github.com/coder/aibridge/context"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/metrics"
	"github.com/coder/aibridge/recorder"
	"github.com/coder/aibridge/tracing"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"go.opentelemetry.io/otel/attribute"
)

const (
	requestTimeout = time.Second * 60 // TODO: configurable timeout
)

type responsesInterceptionBase struct {
	id         uuid.UUID
	req        *ResponsesNewParamsWrapper
	reqPayload []byte
	baseURL    string
	apiKey     string
	model      string
	recorder   recorder.Recorder
	mcpProxy   mcp.ServerProxier
	logger     slog.Logger
	metrics    metrics.Metrics
}

func (i *responsesInterceptionBase) newResponsesService() responses.ResponseService {
	opts := []option.RequestOption{
		option.WithBaseURL(i.baseURL),
		option.WithAPIKey(i.apiKey),
	}

	return responses.NewResponseService(opts...)
}

func (i *responsesInterceptionBase) ID() uuid.UUID {
	return i.id
}

func (i *responsesInterceptionBase) Setup(logger slog.Logger, recorder recorder.Recorder, mcpProxy mcp.ServerProxier) {
	i.logger = logger.With(slog.F("model", i.model))
	i.recorder = recorder
	i.mcpProxy = mcpProxy
}

func (i *responsesInterceptionBase) Model() string {
	return i.model
}

func (i *responsesInterceptionBase) baseTraceAttributes(r *http.Request, streaming bool) []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.String(tracing.RequestPath, r.URL.Path),
		attribute.String(tracing.InterceptionID, i.id.String()),
		attribute.String(tracing.InitiatorID, aibcontext.ActorFromContext(r.Context()).ID),
		attribute.String(tracing.Provider, config.ProviderOpenAI),
		attribute.String(tracing.Model, i.Model()),
		attribute.Bool(tracing.Streaming, streaming),
	}
}

func (i *responsesInterceptionBase) validateRequest(ctx context.Context, w http.ResponseWriter) error {
	if i.req == nil {
		err := errors.New("developer error: req is nil")
		i.sendCustomErr(ctx, w, http.StatusInternalServerError, err)
		return err
	}

	if i.req.Background.Value {
		err := fmt.Errorf("background requests are currently not supported by AI Bridge")
		i.sendCustomErr(ctx, w, http.StatusNotImplemented, err)
		return err
	}

	// keeping the same logic for 'parallel_tool_calls' as in chat-completions
	// https://github.com/coder/aibridge/blob/7535a71e91a1d214a31a9b59bb810befb26141bc/intercept/chatcompletions/streaming.go#L99
	if len(i.req.Tools) > 0 {
		var err error
		i.reqPayload, err = sjson.SetBytes(i.reqPayload, "parallel_tool_calls", false)
		if err != nil {
			err = fmt.Errorf("failed set parallel_tool_calls parameter: %w", err)
			i.sendCustomErr(ctx, w, http.StatusInternalServerError, err)
			return err
		}
	}

	return nil
}

// sendCustomErr sends custom responses.Error error to the client
// it should only be called before any data is sent back to the client
func (i *responsesInterceptionBase) sendCustomErr(ctx context.Context, w http.ResponseWriter, code int, err error) {
	respErr := responses.Error{
		Code:    strconv.Itoa(code),
		Message: err.Error(),
	}
	if b, err := json.Marshal(respErr); err != nil {
		i.logger.Warn(ctx, "failed to marshal custom error: ", slog.Error(err))
	} else {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(code)
		if _, err := w.Write(b); err != nil {
			i.logger.Warn(ctx, "failed to send custom error: ", slog.Error(err))
		}
	}
}

func (i *responsesInterceptionBase) requestOptions(respCopy *responseCopier) []option.RequestOption {
	opts := []option.RequestOption{
		// Sends original payload to solve json re-encoding issues
		// eg. Codex CLI produces requests without ID set in reasoning items: https://platform.openai.com/docs/api-reference/responses/create#responses_create-input-input_item_list-item-reasoning-id
		// when re-encoded, ID field is set to empty string which results
		// in bad request while not sending ID field at all somehow works.
		option.WithRequestBody("application/json", i.reqPayload),

		// copyMiddleware copies body of original response body to the buffer in responseCopier,
		// also reference to headers and status code is kept responseCopier.
		// responseCopier is used by interceptors to forward response as it was received,
		// eliminating any possibility of JSON re-encoding issues.
		option.WithMiddleware(respCopy.copyMiddleware),
	}
	if !i.req.Stream {
		opts = append(opts, option.WithRequestTimeout(requestTimeout))
	}
	return opts
}

// lastUserPrompt returns last input message with "user" role
func (i *responsesInterceptionBase) lastUserPrompt() (string, error) {
	if i == nil {
		return "", errors.New("cannot get last user prompt: nil struct")
	}
	if i.req == nil {
		return "", errors.New("cannot get last user prompt: nil req struct")
	}

	// 'input' field can be a string or array of objects:
	// https://platform.openai.com/docs/api-reference/responses/create#responses_create-input

	// Check string variant
	if i.req.Input.OfString.Valid() {
		return i.req.Input.OfString.Value, nil
	}

	// fallback to parsing original bytes since golang SDK doensn't properly decode 'Input' field.
	// If 'type' field of input item is not set it will be omitted from 'Input.OfInputItemList'
	// It is an optional field according to API: https://platform.openai.com/docs/api-reference/responses/create#responses_create-input-input_item_list-input_message
	// example: fixtures/openai/responses/blocking/builtin_tool.txtar
	inputItems := gjson.GetBytes(i.reqPayload, "input").Array()
	for i := len(inputItems) - 1; i >= 0; i-- {
		item := inputItems[i]
		if item.Get("role").Str == "user" {
			var sb strings.Builder

			// content can be a string or array of objects:
			// https://platform.openai.com/docs/api-reference/responses/create#responses_create-input-input_item_list-input_message-content
			content := item.Get("content")
			if content.Str != "" {
				return content.Str, nil
			}
			for _, c := range content.Array() {
				if c.Get("type").Str == "input_text" {
					sb.WriteString(c.Get("text").Str)
				}
			}
			if sb.Len() > 0 {
				return sb.String(), nil
			}
		}
	}

	return "", errors.New("failed to find last user prompt")
}

func (i *responsesInterceptionBase) recordUserPrompt(ctx context.Context, responseID string) {
	prompt, err := i.lastUserPrompt()
	if err != nil {
		i.logger.Warn(ctx, "failed to get last user prompt", slog.Error(err))
		return
	}

	if prompt != "" && responseID != "" {
		promptUsage := &recorder.PromptUsageRecord{
			InterceptionID: i.ID().String(),
			MsgID:          responseID,
			Prompt:         prompt,
		}
		if err := i.recorder.RecordPromptUsage(ctx, promptUsage); err != nil {
			i.logger.Warn(ctx, "failed to record prompt usage", slog.Error(err))
		}
		return
	}

	if prompt == "" {
		i.logger.Warn(ctx, "got empty last prompt, skipping prompt recording")
		return
	}
	i.logger.Warn(ctx, "got empty response ID, skipping prompt recording")
}

// responseCopier helper struct to send original response to the client
type responseCopier struct {
	buff            deltaBuffer
	responseStatus  int
	responseHeaders http.Header

	// responseBody keeps reference to original ReadCloser.
	// TeeReader in copyMiddleware copies read bytes from
	// response body (read by SDK) to the buffer. In case
	// SDK doesns't read everything readAll method reads from
	// this closer to makes sure whole response body is in the buffer.
	responseBody io.ReadCloser

	// responseReceived flag is used to determine if AI Bridge needs to write custom error:
	// - If responseReceived is true, the upstream response is forwarded as-is.
	// - If responseReceived is false, no response was returned and there is nothing to forward (eg. connection/client error). Custom error will be returned.
	responseReceived atomic.Bool
}

func (r *responseCopier) copyMiddleware(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	resp, err := next(req)
	if err != nil || resp == nil {
		return resp, err
	}

	r.responseReceived.Store(true)
	r.responseStatus = resp.StatusCode
	r.responseHeaders = resp.Header
	resp.Body = io.NopCloser(io.TeeReader(resp.Body, &r.buff))
	r.responseBody = resp.Body
	return resp, nil
}

// readAll reads all data from resp.Body returned by so TeeReader
// so it appends all read data to the buffer and returns buffer contents.
func (r *responseCopier) readAll() ([]byte, error) {
	if r.responseBody == nil {
		return []byte{}, nil
	}

	_, err := io.ReadAll(r.responseBody)
	return r.buff.readDelta(), err
}

// forwardResp writes whole response as received to ResponseWriter
func (r *responseCopier) forwardResp(w http.ResponseWriter) error {
	w.Header().Set("Content-Type", r.responseHeaders.Get("Content-Type"))
	w.WriteHeader(r.responseStatus)

	b, err := r.readAll()
	if err != nil {
		return fmt.Errorf("failed to read response body: %w", err)
	}

	if _, err := w.Write(b); err != nil {
		return fmt.Errorf("failed to write response body: %w", err)
	}
	return nil
}

// deltaBuffer is a thread safe byte buffer
// supports reading incremental data (added after last read)
type deltaBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (d *deltaBuffer) Write(p []byte) (int, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.buf.Write(p)
}

// readDelta returns only the bytes appended
// after the last readDelta call.
func (d *deltaBuffer) readDelta() []byte {
	d.mu.Lock()
	defer d.mu.Unlock()

	b := bytes.Clone(d.buf.Bytes())
	d.buf.Reset()
	return b
}
