package responses

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
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
	"github.com/tidwall/sjson"
	"go.opentelemetry.io/otel/attribute"
)

const (
	requestTimeout = time.Second * 60 // TODO: configurable timeout
)

type responsesInterceptionBase struct {
	id          uuid.UUID
	req         *ResponsesNewParamsWrapper
	reqPayload  []byte
	baseURL     string
	apiKey      string
	model       string
	gotResponse atomic.Bool
	recorder    recorder.Recorder
	mcpProxy    mcp.ServerProxier
	logger      slog.Logger
	metrics     metrics.Metrics
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
	i.logger = logger.With(slog.F("model", i.req.Model))
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
		return errors.New("developer error: req is nil")
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

func (i *responsesInterceptionBase) requestOptions(payloadBuff *deltaBuffer, w http.ResponseWriter) []option.RequestOption {
	opts := []option.RequestOption{
		// Sends original payload to solve json re-encoding issues
		// eg. Codex CLI produces requests without ID set in reasoning items: https://platform.openai.com/docs/api-reference/responses/create#responses_create-input-input_item_list-item-reasoning-id
		// when re-encoded, ID field is set to empty string which results
		// in bad request while not sending ID field at all somehow works.
		option.WithRequestBody("application/json", i.reqPayload),

		// Reads response body into given buffer
		option.WithMiddleware(i.copyMiddleware(payloadBuff, w)),
	}
	if !i.req.Stream {
		opts = append(opts, option.WithRequestTimeout(requestTimeout))
	}
	return opts
}

// copyMiddleware copies Content-Type header, status code and
// response body to given buffer leaving original response
// intact/consumable for openAI SDK
func (i *responsesInterceptionBase) copyMiddleware(payloadBuff *deltaBuffer, w http.ResponseWriter) func(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	return func(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
		resp, err := next(req)
		if err != nil || resp == nil {
			return resp, err
		}

		// mark that some response has been received
		i.gotResponse.Store(true)

		w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
		w.WriteHeader(resp.StatusCode)
		payloadBuff.closer = resp.Body
		resp.Body = io.NopCloser(io.TeeReader(resp.Body, payloadBuff))
		return resp, nil
	}
}

// handleUpstreamError checks if upstream error is from http response or not.
//   - If it is error from response, original status code/body is forwarded as usual.
//   - If it is a client/connection error, internal server error is returned with error as plain text.
func (i *responsesInterceptionBase) handleUpstreamError(ctx context.Context, upstreamErr error, w http.ResponseWriter) (error, bool) {
	if upstreamErr == nil {
		return nil, false
	}

	i.logger.Warn(ctx, "openai responses API upstream error", slog.Error(upstreamErr))

	if i.gotResponse.Load() {
		// if copyMiddleware received some response forward it as is
		return fmt.Errorf("responses API error: %w", upstreamErr), false
	}

	// if no response has been received (eg. client/connection error) return internal server error
	http.Error(w, upstreamErr.Error(), http.StatusInternalServerError)
	return fmt.Errorf("upstream connection error: %w", upstreamErr), true
}

// deltaBuffer is a thread safe byte buffer
// supports reading incremental data (added after last read)
type deltaBuffer struct {
	mu     sync.RWMutex
	buf    bytes.Buffer
	closer io.ReadCloser
}

func (d *deltaBuffer) Write(p []byte) (int, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.buf.Write(p)
}

// Reads all data from original resqusts body
// so it is properly copied by TeeReader to buffer
func (d *deltaBuffer) drainCloser() error {
	if d.closer == nil {
		return nil
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	_, err := io.ReadAll(d.closer)
	return err
}

// readDelta returns only the bytes appended
// after the last readDelta call.
func (d *deltaBuffer) readDelta() []byte {
	d.mu.RLock()
	defer d.mu.RUnlock()

	b := bytes.Clone(d.buf.Bytes())
	d.buf.Reset()
	return b
}
