package aibridge

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"cdr.dev/slog"
	aibtrace "github.com/coder/aibridge/aibtrace"
	"github.com/coder/aibridge/mcp"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// Interceptor describes a (potentially) stateful interaction with an AI provider.
type Interceptor interface {
	// ID returns the unique identifier for this interception.
	ID() uuid.UUID
	// Setup injects some required dependencies. This MUST be called before using the interceptor
	// to process requests.
	Setup(logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier)

	// Model returns the model in use for this [Interceptor].
	Model() string
	// ProcessRequest handles the HTTP request.
	ProcessRequest(w http.ResponseWriter, r *http.Request) error

	// TraceAttributes returns tacing attributes for this [Inteceptor]
	TraceAttributes(context.Context) []attribute.KeyValue
}

var UnknownRoute = errors.New("unknown route")

// The duration after which an async recording will be aborted.
const recordingTimeout = time.Second * 5

// newInterceptionProcessor returns an [http.HandlerFunc] which is capable of creating a new interceptor and processing a given request
// using [Provider] p, recording all usage events using [Recorder] recorder.
func newInterceptionProcessor(p Provider, logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx, span := tracer.Start(r.Context(), "Intercept", trace.WithAttributes(
			attribute.String(aibtrace.TraceProviderKey, p.Name()),
		))
		defer span.End()
		r = r.WithContext(ctx)

		interceptor, err := p.CreateInterceptor(w, r)
		if err != nil {
			span.SetStatus(codes.Error, fmt.Sprintf("failed to create interceptor: %v", err))
			logger.Warn(ctx, "failed to create interceptor", slog.Error(err), slog.F("path", r.URL.Path))
			http.Error(w, fmt.Sprintf("failed to create %q interceptor", r.URL.Path), http.StatusInternalServerError)
			return
		}
		span.SetAttributes(attribute.String(aibtrace.TraceInterceptionIDKey, interceptor.ID().String()))
		span.SetAttributes(attribute.String(aibtrace.TraceModelKey, interceptor.Model()))

		// Record usage in the background to not block request flow.
		asyncRecorder := NewAsyncRecorder(logger, recorder, recordingTimeout)
		interceptor.Setup(logger, asyncRecorder, mcpProxy)

		actor := actorFromContext(ctx)
		if actor == nil {
			logger.Warn(ctx, "no actor found in context")
			http.Error(w, "no actor found", http.StatusBadRequest)
			return
		}
		span.SetAttributes(attribute.String(aibtrace.TraceUserIDKey, actor.id))
		ctx = aibtrace.WithTraceInterceptionAttributesInContext(ctx, interceptor.TraceAttributes(ctx))

		if err := recorder.RecordInterception(ctx, &InterceptionRecord{
			ID:          interceptor.ID().String(),
			Metadata:    actor.metadata,
			InitiatorID: actor.id,
			Provider:    p.Name(),
			Model:       interceptor.Model(),
		}); err != nil {
			span.SetStatus(codes.Error, fmt.Sprintf("failed to record interception: %v", err))
			logger.Warn(ctx, "failed to record interception", slog.Error(err))
			http.Error(w, "failed to record interception", http.StatusInternalServerError)
			return
		}

		log := logger.With(
			slog.F("route", r.URL.Path),
			slog.F("provider", p.Name()),
			slog.F("interception_id", interceptor.ID()),
			slog.F("user_agent", r.UserAgent()),
		)

		log.Debug(ctx, "interception started")
		if err := interceptor.ProcessRequest(w, r); err != nil {
			span.SetStatus(codes.Error, fmt.Sprintf("interception failed: %v", err))
			log.Warn(ctx, "interception failed", slog.Error(err))
		} else {
			log.Debug(ctx, "interception ended")
		}
		asyncRecorder.RecordInterceptionEnded(ctx, &InterceptionRecordEnded{ID: interceptor.ID().String()})

		// Ensure all recording have completed before completing request.
		asyncRecorder.Wait()
	}
}
