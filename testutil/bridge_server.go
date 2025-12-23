package testutil

import (
	"bytes"
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"cdr.dev/slog"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/mcp"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
)

type BridgeConfig struct {
	Ctx     context.Context
	ActorID string

	// Exactly one of Handler or Providers must be set.
	Handler   http.Handler
	Providers []aibridge.Provider

	Recorder aibridge.Recorder

	MCPProxiers map[string]mcp.ServerProxier

	Logger  slog.Logger
	Metrics *aibridge.Metrics
	Tracer  trace.Tracer
}

type BridgeServer struct {
	*httptest.Server
	Client *http.Client
}

func NewBridgeServer(t testing.TB, cfg BridgeConfig) *BridgeServer {
	t.Helper()

	ctx := cfg.Ctx
	if ctx == nil {
		ctx = context.Background()
	}

	if cfg.Tracer == nil {
		cfg.Tracer = noop.NewTracerProvider().Tracer("aibridge/testutil")
	}

	if cfg.Handler == nil {
		if len(cfg.Providers) == 0 {
			t.Fatalf("BridgeConfig: must set either Handler or Providers")
		}
		if cfg.Recorder == nil {
			t.Fatalf("BridgeConfig: Recorder is required when building a RequestBridge")
		}

		mgr := mcp.NewServerProxyManager(cfg.MCPProxiers, cfg.Tracer)
		// Only init when there are proxiers. This keeps trace output consistent with
		// tests that intentionally pass a nil/empty proxy map.
		if len(cfg.MCPProxiers) > 0 {
			if err := mgr.Init(ctx); err != nil {
				t.Fatalf("init MCP manager: %v", err)
			}
		}

		bridge, err := aibridge.NewRequestBridge(ctx, cfg.Providers, cfg.Recorder, mgr, cfg.Logger, cfg.Metrics, cfg.Tracer)
		if err != nil {
			t.Fatalf("create RequestBridge: %v", err)
		}
		cfg.Handler = bridge
	}

	srv := httptest.NewUnstartedServer(cfg.Handler)
	srv.Config.BaseContext = func(_ net.Listener) context.Context {
		if cfg.ActorID == "" {
			return ctx
		}
		return aibridge.AsActor(ctx, cfg.ActorID, nil)
	}
	srv.Start()
	t.Cleanup(srv.Close)

	return &BridgeServer{
		Server: srv,
		Client: &http.Client{},
	}
}

func (b *BridgeServer) NewProviderRequest(t testing.TB, provider string, body []byte) *http.Request {
	t.Helper()

	path := ""
	switch provider {
	case aibridge.ProviderAnthropic:
		path = "/anthropic/v1/messages"
	case aibridge.ProviderOpenAI:
		path = "/openai/v1/chat/completions"
	default:
		t.Fatalf("unknown provider %q", provider)
	}

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, b.URL+path, bytes.NewReader(body))
	if err != nil {
		t.Fatalf("create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	return req
}
