package integrationtest

import (
	"context"
	"net"
	"net/http/httptest"
	"testing"

	"cdr.dev/slog/v3"
	"cdr.dev/slog/v3/sloggers/slogtest"
	"github.com/coder/aibridge"
	aibcontext "github.com/coder/aibridge/context"
	"github.com/coder/aibridge/internal/testutil"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/metrics"
	"github.com/coder/aibridge/recorder"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/trace"
)

// DefaultActorID is the actor ID used by default in test servers.
const DefaultActorID = "ae235cc1-9f8f-417d-a636-a7b170bac62e"

// DefaultTracer is the default OTel tracer used in integration tests.
var DefaultTracer = otel.Tracer("integrationtest")

// NewLogger creates a test logger at Debug level.
// Eliminates the repeated slogtest.Make(t, &slogtest.Options{...}).Leveled(slog.LevelDebug) pattern.
func NewLogger(t *testing.T, opts ...*slogtest.Options) slog.Logger {
	t.Helper()
	var o *slogtest.Options
	if len(opts) > 0 {
		o = opts[0]
	} else {
		o = &slogtest.Options{}
	}
	return slogtest.Make(t, o).Leveled(slog.LevelDebug)
}

// BridgeTestServer wraps an httptest.Server running a RequestBridge.
type BridgeTestServer struct {
	*httptest.Server
	Recorder *testutil.MockRecorder
	Bridge   *aibridge.RequestBridge
}

// BridgeOption configures a [BridgeTestServer].
type BridgeOption func(*bridgeConfig)

type bridgeConfig struct {
	metrics      *metrics.Metrics
	tracer       trace.Tracer
	mcpProxy     mcp.ServerProxier
	userID       string
	metadata     recorder.Metadata
	logger       slog.Logger
	loggerSet    bool
	wrapRecorder bool
}

// WithMetrics sets the Prometheus metrics for the bridge.
func WithMetrics(m *metrics.Metrics) BridgeOption {
	return func(c *bridgeConfig) { c.metrics = m }
}

// WithTracer overrides the default tracer.
func WithTracer(t trace.Tracer) BridgeOption {
	return func(c *bridgeConfig) { c.tracer = t }
}

// WithMCP sets the MCP server proxier (default: NoopMCPManager).
func WithMCP(p mcp.ServerProxier) BridgeOption {
	return func(c *bridgeConfig) { c.mcpProxy = p }
}

// WithActor sets the actor ID and metadata for the BaseContext.
func WithActor(id string, md recorder.Metadata) BridgeOption {
	return func(c *bridgeConfig) { c.userID = id; c.metadata = md }
}

// WithLogger overrides the default slogtest debug logger.
func WithLogger(l slog.Logger) BridgeOption {
	return func(c *bridgeConfig) { c.logger = l; c.loggerSet = true }
}

// WithWrappedRecorder wraps the MockRecorder through aibridge.NewRecorder
// (the production recorder wrapper). Use when testing the recorder pipeline.
func WithWrappedRecorder() BridgeOption {
	return func(c *bridgeConfig) { c.wrapRecorder = true }
}

// NewBridgeTestServer creates a fully configured test server running
// a RequestBridge with sensible defaults:
//   - MockRecorder (raw, unless WithWrappedRecorder)
//   - NoopMCPManager (unless WithMCP)
//   - slogtest debug logger (unless WithLogger)
//   - DefaultTracer (unless WithTracer)
//   - DefaultActorID (unless WithActor)
func NewBridgeTestServer(
	t *testing.T,
	ctx context.Context,
	providers []aibridge.Provider,
	opts ...BridgeOption,
) *BridgeTestServer {
	t.Helper()

	cfg := &bridgeConfig{
		userID: DefaultActorID,
	}
	for _, o := range opts {
		o(cfg)
	}
	if cfg.tracer == nil {
		cfg.tracer = DefaultTracer
	}
	if !cfg.loggerSet {
		cfg.logger = NewLogger(t)
	}
	if cfg.mcpProxy == nil {
		cfg.mcpProxy = NewNoopMCPManager()
	}

	mockRec := &testutil.MockRecorder{}
	var rec aibridge.Recorder = mockRec
	if cfg.wrapRecorder {
		rec = aibridge.NewRecorder(cfg.logger, cfg.tracer, func() (aibridge.Recorder, error) {
			return mockRec, nil
		})
	}

	bridge, err := aibridge.NewRequestBridge(
		ctx, providers, rec, cfg.mcpProxy,
		cfg.logger, cfg.metrics, cfg.tracer,
	)
	require.NoError(t, err)

	actorID, md := cfg.userID, cfg.metadata
	srv := httptest.NewUnstartedServer(bridge)
	srv.Config.BaseContext = func(_ net.Listener) context.Context {
		return aibcontext.AsActor(ctx, actorID, md)
	}
	srv.Start()
	t.Cleanup(srv.Close)

	return &BridgeTestServer{
		Server:   srv,
		Recorder: mockRec,
		Bridge:   bridge,
	}
}
