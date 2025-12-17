package aibridge

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"cdr.dev/slog"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/tracing"
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
	// Specifies whether an interceptor handles streaming or not.
	Streaming() bool
	// TraceAttributes returns tracing attributes for this [Interceptor]
	TraceAttributes(*http.Request) []attribute.KeyValue
}

var UnknownRoute = errors.New("unknown route")

// The duration after which an async recording will be aborted.
const recordingTimeout = time.Second * 5

// newInterceptionProcessor returns an [http.HandlerFunc] which is capable of creating a new interceptor and processing a given request
// using [Provider] p, recording all usage events using [Recorder] recorder.
func newInterceptionProcessor(p Provider, recorder Recorder, mcpProxy mcp.ServerProxier, logger slog.Logger, metrics *Metrics, tracer trace.Tracer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx, span := tracer.Start(r.Context(), "Intercept")
		defer span.End()

		route := strings.TrimPrefix(r.URL.Path, fmt.Sprintf("/%s", p.Name()))

		interceptor, err := p.CreateInterceptor(w, r.WithContext(ctx), tracer)
		if err != nil {
			span.SetStatus(codes.Error, fmt.Sprintf("failed to create interceptor: %v", err))
			logger.Warn(ctx, "failed to create interceptor", slog.Error(err), slog.F("path", r.URL.Path))
			http.Error(w, fmt.Sprintf("failed to create %q interceptor", r.URL.Path), http.StatusInternalServerError)
			return
		}

		if metrics != nil {
			start := time.Now()
			defer func() {
				metrics.InterceptionDuration.WithLabelValues(p.Name(), interceptor.Model()).Observe(time.Since(start).Seconds())
			}()
		}

		actor := actorFromContext(ctx)
		if actor == nil {
			logger.Warn(ctx, "no actor found in context")
			http.Error(w, "no actor found", http.StatusBadRequest)
			return
		}

		traceAttrs := interceptor.TraceAttributes(r)
		span.SetAttributes(traceAttrs...)
		ctx = tracing.WithInterceptionAttributesInContext(ctx, traceAttrs)
		r = r.WithContext(ctx)

		// Record usage in the background to not block request flow.
		asyncRecorder := NewAsyncRecorder(logger, recorder, recordingTimeout)
		asyncRecorder.WithMetrics(metrics)
		asyncRecorder.WithProvider(p.Name())
		asyncRecorder.WithModel(interceptor.Model())
		asyncRecorder.WithInitiatorID(actor.id)
		interceptor.Setup(logger, asyncRecorder, mcpProxy)

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
			slog.F("route", route),
			slog.F("provider", p.Name()),
			slog.F("interception_id", interceptor.ID()),
			slog.F("user_agent", r.UserAgent()),
			slog.F("streaming", interceptor.Streaming()),
		)

		log.Debug(ctx, "interception started")
		if metrics != nil {
			metrics.InterceptionsInflight.WithLabelValues(p.Name(), interceptor.Model(), route).Add(1)
			defer func() {
				metrics.InterceptionsInflight.WithLabelValues(p.Name(), interceptor.Model(), route).Sub(1)
			}()
		}

		if err := interceptor.ProcessRequest(w, r); err != nil {
			if metrics != nil {
				metrics.InterceptionCount.WithLabelValues(p.Name(), interceptor.Model(), InterceptionCountStatusFailed, route, r.Method, actor.id).Add(1)
			}
			span.SetStatus(codes.Error, fmt.Sprintf("interception failed: %v", err))
			log.Warn(ctx, "interception failed", slog.Error(err))
		} else {
			if metrics != nil {
				metrics.InterceptionCount.WithLabelValues(p.Name(), interceptor.Model(), InterceptionCountStatusCompleted, route, r.Method, actor.id).Add(1)
			}
			log.Debug(ctx, "interception ended")
		}
		asyncRecorder.RecordInterceptionEnded(ctx, &InterceptionRecordEnded{ID: interceptor.ID().String()})

		// Ensure all recording have completed before completing request.
		asyncRecorder.Wait()
	}
}
