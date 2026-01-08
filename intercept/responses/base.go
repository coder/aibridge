package responses

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/config"
	aibcontext "github.com/coder/aibridge/context"
	"github.com/coder/aibridge/intercept/eventstream"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/metrics"
	"github.com/coder/aibridge/recorder"
	"github.com/coder/aibridge/tracing"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/tidwall/sjson"
	"go.opentelemetry.io/otel/attribute"
)

type responsesInterceptionBase struct {
	id         uuid.UUID
	req        *ResponsesNewParamsWrapper
	reqPayload []byte
	baseURL    string
	key        string
	recorder   recorder.Recorder
	mcpProxy   mcp.ServerProxier

	logger  slog.Logger
	metrics metrics.Metrics
}

func NewResponsesService(baseURL string, key string, logger slog.Logger) responses.ResponseService {
	opts := []option.RequestOption{
		option.WithAPIKey(key),
		option.WithBaseURL(baseURL),
	}

	return responses.NewResponseService(opts...)
}

func (i *responsesInterceptionBase) ID() uuid.UUID {
	return i.id
}

func (i *responsesInterceptionBase) Setup(logger slog.Logger, recorder recorder.Recorder, mcpProxy mcp.ServerProxier) {
	i.logger = logger.With(slog.F("model", i.req.Model))
	i.recorder = recorder
	i.mcpProxy = mcpProxy
}

func (i *responsesInterceptionBase) Model() string {
	if i.req == nil {
		return "coder-aibridge-unknown"
	}

	return string(i.req.Model)
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
		return fmt.Errorf("developer error: req is nil")
	}

	if i.req.Background.Value {
		err := fmt.Errorf("background requests are currently not supported by aibridge")
		http.Error(w, err.Error(), http.StatusNotImplemented)
		return err
	}

	if len(i.req.Tools) > 0 {
		var err error
		i.reqPayload, err = sjson.SetBytes(i.reqPayload, "parallel_tool_calls", false)
		if err != nil {
			i.logger.Warn(ctx, "failed set parallel_tool_calls parameter", slog.Error(err))
			return fmt.Errorf("failed to prepare request: %w", err)
		}
	}

	return nil
}

func (i *responsesInterceptionBase) requestOptions(respBody *deltaBuffer) []option.RequestOption {
	opts := []option.RequestOption{
		// Sends original payload to solve json re-encoding issues
		// eg. Codex CLI produces requests without ID set in reasoning items: https://platform.openai.com/docs/api-reference/responses/create#responses_create-input-input_item_list-item-reasoning-id
		// when re-encoded, ID field is set to empty string which results
		// in bad request while not sending ID field at all somehow works.
		option.WithRequestBody("application/json", i.reqPayload),

		// Reads response body into given buffer
		option.WithMiddleware(teeMiddleware(respBody)),
	}
	if !i.req.Stream {
		opts = append(opts, option.WithRequestTimeout(time.Second*60)) // TODO: configurable timeout
	}
	return opts
}

// handleUpstreamError checks error if it is an openAI error and if ProcessRequest should exit early.
// If it is openAI responses error -> sets proper http response code and returns false to not exit early
// response body will be sent using the same method as non-error response, using teeMiddleware -> deltaBuffer.
// If it is a connection error or unnknown error -> returns given error + true to indicate early exit
func (i *responsesInterceptionBase) handleUpstreamError(ctx context.Context, upstreamErr error, w http.ResponseWriter) (error, bool) {
	if upstreamErr == nil {
		return nil, false
	}

	if eventstream.IsConnError(upstreamErr) {
		http.Error(w, upstreamErr.Error(), http.StatusInternalServerError)
		return fmt.Errorf("upstream connection closed: %w", upstreamErr), true
	}

	i.logger.Warn(ctx, "openai responses API error", slog.Error(upstreamErr))
	var rspErr *responses.Error
	if errors.As(upstreamErr, &rspErr) {
		w.WriteHeader(rspErr.StatusCode)
		return fmt.Errorf("responses API error: %w", rspErr), false
	}

	w.WriteHeader(http.StatusInternalServerError)
	return fmt.Errorf("responses API failed: %w", upstreamErr), true
}

// teeMiddleware copies response body to given buffer leaving original response intact/consumable for openAI SDK
func teeMiddleware(respBody io.Writer) func(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	return func(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
		resp, err := next(req)
		if err != nil || resp == nil || resp.Body == nil {
			return resp, err
		}

		resp.Body = io.NopCloser(io.TeeReader(resp.Body, respBody))
		return resp, nil
	}
}

// deltaBuffer stores everything written to it and lets you read only the new bytes since last drain.
type deltaBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (d *deltaBuffer) Write(p []byte) (int, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.buf.Write(p)
}

func (d *deltaBuffer) Read(p []byte) (int, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	c, err := d.buf.Read(p)
	return c, err
}

// readDelta returns only the bytes appended since the last readDelta call.
func (d *deltaBuffer) readDelta() []byte {
	d.mu.Lock()
	defer d.mu.Unlock()

	b := bytes.Clone(d.buf.Bytes())
	d.buf.Reset()
	return b
}
