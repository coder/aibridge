package aibridge

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"

	"cdr.dev/slog"
	"github.com/coder/aibridge/mcp"
	"go.opentelemetry.io/otel/trace"

	"github.com/hashicorp/go-multierror"
)

// RequestBridge is an [http.Handler] which is capable of masquerading as AI providers' APIs;
// specifically, OpenAI's & Anthropic's at present.
// RequestBridge intercepts requests to - and responses from - these upstream services to provide
// a centralized governance layer.
//
// RequestBridge has no concept of authentication or authorization. It does have a concept of identity,
// in the narrow sense that it expects an [actor] to be defined in the context, to record the initiator
// of each interception.
//
// RequestBridge is safe for concurrent use.
type RequestBridge struct {
	mux    *http.ServeMux
	logger slog.Logger

	mcpProxy mcp.ServerProxier

	// circuitBreakers manages circuit breakers for upstream providers.
	// When enabled, it protects against cascading failures from upstream rate limits.
	circuitBreakers *CircuitBreakers

	inflightReqs atomic.Int32
	inflightWG   sync.WaitGroup // For graceful shutdown.

	inflightCtx    context.Context
	inflightCancel func()

	shutdownOnce sync.Once
	closed       chan struct{}
}

var _ http.Handler = &RequestBridge{}

// NewRequestBridge creates a new *[RequestBridge] and registers the HTTP routes defined by the given providers.
// Any routes which are requested but not registered will be reverse-proxied to the upstream service.
//
// A [Recorder] is also required to record prompt, tool, and token use.
//
// mcpProxy will be closed when the [RequestBridge] is closed.
func NewRequestBridge(ctx context.Context, providers []Provider, recorder Recorder, mcpProxy mcp.ServerProxier, logger slog.Logger, metrics *Metrics, tracer trace.Tracer) (*RequestBridge, error) {
	return NewRequestBridgeWithCircuitBreaker(ctx, providers, recorder, mcpProxy, logger, metrics, tracer, DefaultCircuitBreakerConfig())
}

// NewRequestBridgeWithCircuitBreaker creates a new *[RequestBridge] with custom circuit breaker configuration.
// See [NewRequestBridge] for more details.
func NewRequestBridgeWithCircuitBreaker(ctx context.Context, providers []Provider, recorder Recorder, mcpProxy mcp.ServerProxier, logger slog.Logger, metrics *Metrics, tracer trace.Tracer, cbConfig CircuitBreakerConfig) (*RequestBridge, error) {
	mux := http.NewServeMux()

	// Create circuit breakers with metrics callback
	var onChange func(name string, from, to CircuitState)
	if metrics != nil {
		onChange = func(name string, from, to CircuitState) {
			provider, endpoint, _ := strings.Cut(name, ":")
			metrics.CircuitBreakerState.WithLabelValues(provider, endpoint).Set(float64(to))
			if to == CircuitOpen {
				metrics.CircuitBreakerTrips.WithLabelValues(provider, endpoint).Inc()
			}
		}
	}
	cbs := NewCircuitBreakers(cbConfig, onChange)

	for _, provider := range providers {

		// Add the known provider-specific routes which are bridged (i.e. intercepted and augmented).
		for _, path := range provider.BridgedRoutes() {
			mux.HandleFunc(path, newInterceptionProcessor(provider, recorder, mcpProxy, logger, metrics, tracer, cbs))
		}

		// Any requests which passthrough to this will be reverse-proxied to the upstream.
		//
		// We have to whitelist the known-safe routes because an API key with elevated privileges (i.e. admin) might be
		// configured, so we should just reverse-proxy known-safe routes.
		ftr := newPassthroughRouter(provider, logger.Named(fmt.Sprintf("passthrough.%s", provider.Name())), metrics, tracer)
		for _, path := range provider.PassthroughRoutes() {
			prefix := fmt.Sprintf("/%s", provider.Name())
			route := fmt.Sprintf("%s%s", prefix, path)
			mux.HandleFunc(route, http.StripPrefix(prefix, ftr).ServeHTTP)
		}
	}

	// Catch-all.
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		logger.Warn(r.Context(), "route not supported", slog.F("path", r.URL.Path), slog.F("method", r.Method))
		http.Error(w, fmt.Sprintf("route not supported: %s %s", r.Method, r.URL.Path), http.StatusNotFound)
	})

	inflightCtx, cancel := context.WithCancel(context.Background())
	return &RequestBridge{
		mux:             mux,
		logger:          logger,
		mcpProxy:        mcpProxy,
		circuitBreakers: cbs,
		inflightCtx:     inflightCtx,
		inflightCancel:  cancel,

		closed: make(chan struct{}, 1),
	}, nil
}

// ServeHTTP exposes the internal http.Handler, which has all [Provider]s' routes registered.
// It also tracks inflight requests.
func (b *RequestBridge) ServeHTTP(rw http.ResponseWriter, r *http.Request) {
	select {
	case <-b.closed:
		http.Error(rw, "server closed", http.StatusInternalServerError)
		return
	default:
	}

	// We want to abide by the context passed in without losing any of its
	// functionality, but we still want to link our shutdown context to each
	// request.
	ctx := mergeContexts(r.Context(), b.inflightCtx)

	b.inflightReqs.Add(1)
	b.inflightWG.Add(1)
	defer func() {
		b.inflightReqs.Add(-1)
		b.inflightWG.Done()
	}()

	b.mux.ServeHTTP(rw, r.WithContext(ctx))
}

// Shutdown will attempt to gracefully shutdown. This entails waiting for all requests to
// complete, and shutting down the MCP server proxier.
// TODO: add tests.
func (b *RequestBridge) Shutdown(ctx context.Context) error {
	var err error
	b.shutdownOnce.Do(func() {
		// Prevent any new requests from being accepted.
		close(b.closed)

		// Wait for inflight requests to complete or context cancellation.
		done := make(chan struct{})
		go func() {
			b.inflightWG.Wait()
			close(done)
		}()

		select {
		case <-ctx.Done():
			// Cancel all inflight requests, if any are still running.
			b.logger.Debug(ctx, "shutdown context canceled; cancelling inflight requests", slog.Error(ctx.Err()))
			b.inflightCancel()
			<-done
			err = ctx.Err()
		case <-done:
		}

		if b.mcpProxy != nil {
			// It's ok that we reuse the ctx here even if it's done, since the
			// Shutdown method will just immediately use the more aggressive close
			// since the ctx is already expired.
			err = multierror.Append(err, b.mcpProxy.Shutdown(ctx))
		}
	})

	return err
}

func (b *RequestBridge) InflightRequests() int32 {
	return b.inflightReqs.Load()
}

// CircuitBreakers returns the circuit breakers for this bridge.
func (b *RequestBridge) CircuitBreakers() *CircuitBreakers {
	return b.circuitBreakers
}

// mergeContexts merges two contexts together, so that if either is cancelled
// the returned context is cancelled. The context values will only be used from
// the first context.
func mergeContexts(base, other context.Context) context.Context {
	ctx, cancel := context.WithCancel(base)
	go func() {
		defer cancel()
		select {
		case <-base.Done():
		case <-other.Done():
		}
	}()
	return ctx
}
