package aibridge

import (
	"errors"
	"fmt"
	"net/http"
	"time"

	"cdr.dev/slog"
	"github.com/coder/aibridge/mcp"
	"github.com/google/uuid"
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
}

var UnknownRoute = errors.New("unknown route")

// The duration after which an async recording will be aborted.
const recordingTimeout = time.Second * 5

// newInterceptionProcessor returns an [http.HandlerFunc] which is capable of creating a new interceptor and processing a given request
// using [Provider] p, recording all usage events using [Recorder] recorder.
func newInterceptionProcessor(p Provider, logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		interceptor, err := p.CreateInterceptor(w, r)
		if err != nil {
			logger.Warn(r.Context(), "failed to create interceptor", slog.Error(err), slog.F("path", r.URL.Path))
			http.Error(w, fmt.Sprintf("failed to create %q interceptor", r.URL.Path), http.StatusInternalServerError)
			return
		}

		// Record usage in the background to not block request flow.
		asyncRecorder := NewAsyncRecorder(logger, recorder, recordingTimeout)
		interceptor.Setup(logger, asyncRecorder, mcpProxy)

		actor := actorFromContext(r.Context())
		if actor == nil {
			logger.Warn(r.Context(), "no actor found in context")
			http.Error(w, "no actor found", http.StatusBadRequest)
			return
		}

		if err := recorder.RecordInterception(r.Context(), &InterceptionRecord{
			ID:          interceptor.ID().String(),
			Metadata:    actor.metadata,
			InitiatorID: actor.id,
			Provider:    p.Name(),
			Model:       interceptor.Model(),
		}); err != nil {
			logger.Warn(r.Context(), "failed to record interception", slog.Error(err))
			http.Error(w, "failed to record interception", http.StatusInternalServerError)
			return
		}

		log := logger.With(slog.F("route", r.URL.Path), slog.F("provider", p.Name()), slog.F("interception_id", interceptor.ID()))

		log.Debug(r.Context(), "interception started")
		if err := interceptor.ProcessRequest(w, r); err != nil {
			log.Warn(r.Context(), "interception failed", slog.Error(err))
		} else {
			log.Debug(r.Context(), "interception ended")
		}
		asyncRecorder.RecordInterceptionEnded(r.Context(), &InterceptionRecordEnded{ID: interceptor.ID().String()})

		// Ensure all recording have completed before completing request.
		asyncRecorder.Wait()
	}
}
