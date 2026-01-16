package responses

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/intercept/eventstream"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/recorder"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v3/responses"
	oaiconst "github.com/openai/openai-go/v3/shared/constant"
	"go.opentelemetry.io/otel/attribute"
)

const (
	streamShutdownTimeout = time.Second * 30 // TODO: configurable
)

type StreamingResponsesInterceptor struct {
	responsesInterceptionBase
}

func NewStreamingInterceptor(id uuid.UUID, req *ResponsesNewParamsWrapper, reqPayload []byte, baseURL string, key string, model string) *StreamingResponsesInterceptor {
	return &StreamingResponsesInterceptor{
		responsesInterceptionBase: responsesInterceptionBase{
			id:         id,
			req:        req,
			reqPayload: reqPayload,
			baseURL:    baseURL,
			apiKey:     key,
			model:      model,
		},
	}
}

func (i *StreamingResponsesInterceptor) Setup(logger slog.Logger, recorder recorder.Recorder, mcpProxy mcp.ServerProxier) {
	i.responsesInterceptionBase.Setup(logger.Named("streaming"), recorder, mcpProxy)
}

func (i *StreamingResponsesInterceptor) Streaming() bool {
	return true
}

func (i *StreamingResponsesInterceptor) TraceAttributes(r *http.Request) []attribute.KeyValue {
	return i.responsesInterceptionBase.baseTraceAttributes(r, true)
}

func (i *StreamingResponsesInterceptor) ProcessRequest(w http.ResponseWriter, r *http.Request) error {
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	r = r.WithContext(ctx) // Rewire context for SSE cancellation.

	if err := i.validateRequest(ctx, w); err != nil {
		return err
	}

	events := eventstream.NewEventStream(ctx, i.logger.Named("sse-sender"), nil)
	go events.Start(w, r)
	defer func() {
		shutdownCtx, shutdownCancel := context.WithTimeout(ctx, streamShutdownTimeout)
		defer shutdownCancel()
		_ = events.Shutdown(shutdownCtx)
	}()

	var respCopy responseCopier
	var responseID string
	var completedResponse *responses.Response

	srv := i.newResponsesService()
	opts := i.requestOptions(&respCopy)
	stream := srv.NewStreaming(ctx, i.req.ResponseNewParams, opts...)
	defer stream.Close()

	if upstreamErr := stream.Err(); upstreamErr != nil {
		// events stream should never be initialized
		if events.IsStreaming() {
			i.logger.Warn(ctx, "event stream was initialized when no response was received from upstream")
			return upstreamErr
		}

		// no response received from upstream (eg. client/connection error), return custom error
		if !respCopy.responseReceived.Load() {
			i.sendCustomErr(ctx, w, http.StatusInternalServerError, upstreamErr)
			return upstreamErr
		}

		// forward received response as-is
		err := respCopy.forwardResp(w)
		return errors.Join(upstreamErr, err)
	}

	for stream.Next() {
		ev := stream.Current()

		// Not every event has response.id set (eg: fixtures/openai/responses/streaming/simple.txtar).
		// First event should be of 'response.created' type and have response.id set.
		// Set responseID to the first response.id that is set.
		if responseID == "" && ev.Response.ID != "" {
			responseID = ev.Response.ID
		}

		// Capture the response from the response.completed event.
		// Only response.completed event type have 'usage' field set.
		if ev.Type == string(oaiconst.ValueOf[oaiconst.ResponseCompleted]()) {
			completedEvent := ev.AsResponseCompleted()
			completedResponse = &completedEvent.Response
		}
		if err := events.Send(ctx, respCopy.buff.readDelta()); err != nil {
			err = fmt.Errorf("failed to relay chunk: %w", err)
			return err
		}
	}
	i.recordUserPrompt(ctx, responseID)
	if completedResponse != nil {
		i.recordToolUsage(ctx, completedResponse)
		i.recordTokenUsage(ctx, completedResponse)
	} else {
		i.logger.Warn(ctx, "got empty response, skipping tool and token usage recording")
	}

	b, err := respCopy.readAll()
	if err != nil {
		return fmt.Errorf("failed to read response body: %w", err)
	}

	err = events.Send(ctx, b)
	return errors.Join(err, stream.Err())
}
