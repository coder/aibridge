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

// defaultActorID is the actor ID used by default in test servers.
const defaultActorID = "ae235cc1-9f8f-417d-a636-a7b170bac62e"

// defaultTracer is the default OTel tracer used in integration tests.
var defaultTracer = otel.Tracer("integrationtest")

// newLogger creates a test logger at Debug level.
func newLogger(t *testing.T) slog.Logger {
	t.Helper()
	return slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)
}

// bridgeTestServer wraps an httptest.Server running a RequestBridge.
type bridgeTestServer struct {
	*httptest.Server
	Recorder *testutil.MockRecorder
	Bridge   *aibridge.RequestBridge
}

// bridgeOption configures a [bridgeTestServer].
type bridgeOption func(*bridgeConfig)

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

// withMetrics sets the Prometheus metrics for the bridge.
func withMetrics(m *metrics.Metrics) bridgeOption {
	return func(c *bridgeConfig) { c.metrics = m }
}

// withTracer overrides the default tracer.
func withTracer(t trace.Tracer) bridgeOption {
	return func(c *bridgeConfig) { c.tracer = t }
}

// withMCP sets the MCP server proxier (default: NoopMCPManager).
func withMCP(p mcp.ServerProxier) bridgeOption {
	return func(c *bridgeConfig) { c.mcpProxy = p }
}

// withActor sets the actor ID and metadata for the BaseContext.
func withActor(id string, md recorder.Metadata) bridgeOption {
	return func(c *bridgeConfig) { c.userID = id; c.metadata = md }
}

// withLogger overrides the default slogtest debug logger.
func withLogger(l slog.Logger) bridgeOption {
	return func(c *bridgeConfig) { c.logger = l; c.loggerSet = true }
}

// withWrappedRecorder wraps the MockRecorder through aibridge.NewRecorder
// (the production recorder wrapper). Use when testing the recorder pipeline.
func withWrappedRecorder() bridgeOption {
	return func(c *bridgeConfig) { c.wrapRecorder = true }
}

// newBridgeTestServer creates a fully configured test server running
// a RequestBridge with sensible defaults:
//   - MockRecorder (raw, unless withWrappedRecorder)
//   - NoopMCPManager (unless withMCP)
//   - slogtest debug logger (unless withLogger)
//   - defaultTracer (unless withTracer)
//   - defaultActorID (unless withActor)
func newBridgeTestServer(
	t *testing.T,
	ctx context.Context,
	providers []aibridge.Provider,
	opts ...bridgeOption,
) *bridgeTestServer {
	t.Helper()

	cfg := &bridgeConfig{
		userID: defaultActorID,
	}
	for _, o := range opts {
		o(cfg)
	}
	if cfg.tracer == nil {
		cfg.tracer = defaultTracer
	}
	if !cfg.loggerSet {
		cfg.logger = newLogger(t)
	}
	if cfg.mcpProxy == nil {
		cfg.mcpProxy = newNoopMCPManager()
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

	return &bridgeTestServer{
		Server:   srv,
		Recorder: mockRec,
		Bridge:   bridge,
	}
}
