// This is an example server demonstrating aibridge usage.
// Run with: go run ./example
package main

import (
	"context"
	"database/sql"
	"net/http"
	"os"
	"regexp"

	"cdr.dev/slog"
	"cdr.dev/slog/sloggers/sloghuman"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/mcp"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	_ "modernc.org/sqlite"
)

func main() {
	ctx := context.Background()
	logger := slog.Make(sloghuman.Sink(os.Stderr)).Leveled(slog.LevelDebug)

	// Initialize SQLite database with WAL mode for better concurrency.
	db, err := sql.Open("sqlite", "aibridge.db?_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		logger.Fatal(ctx, "open database", slog.Error(err))
	}
	defer db.Close()
	db.SetMaxOpenConns(1) // SQLite only supports one writer at a time.

	if err := initSchema(db); err != nil {
		logger.Fatal(ctx, "init schema", slog.Error(err))
	}

	recorder, err := NewSQLiteRecorder(db, logger)
	if err != nil {
		logger.Fatal(ctx, "create recorder", slog.Error(err))
	}
	defer recorder.Close()

	// Configure providers.
	providers := []aibridge.Provider{
		aibridge.NewAnthropicProvider(aibridge.AnthropicConfig{
			Key: os.Getenv("ANTHROPIC_API_KEY"),
		}, nil),
		aibridge.NewOpenAIProvider(aibridge.OpenAIConfig{
			Key: os.Getenv("OPENAI_API_KEY"),
		}),
	}

	// Setup metrics.
	reg := prometheus.NewRegistry()
	metrics := aibridge.NewMetrics(reg)

	// Optional: Configure MCP server for centralized tool injection.
	// DeepWiki provides free access to public GitHub repo documentation.
	// See: https://mcp.deepwiki.com
	var mcpProxy mcp.ServerProxier
	deepwikiProxy, err := mcp.NewStreamableHTTPServerProxy(
		logger.Named("mcp.deepwiki"),
		"deepwiki",                           // server name (tools prefixed as bmcp_deepwiki_*)
		"https://mcp.deepwiki.com/mcp",       // no auth required for public repos
		nil,                                  // headers
		regexp.MustCompile(`^ask_question$`), // allowlist: only ask_question tool
		nil,                                  // denylist
	)
	if err != nil {
		logger.Fatal(ctx, "create deepwiki mcp proxy", slog.Error(err))
	}

	mcpProxy = mcp.NewServerProxyManager(map[string]mcp.ServerProxier{"deepwiki": deepwikiProxy})
	if err := mcpProxy.Init(ctx); err != nil {
		logger.Warn(ctx, "mcp init warning", slog.Error(err))
	}

	// Create the bridge with SQLite recorder.
	bridge, err := aibridge.NewRequestBridge(
		ctx,
		providers,
		recorder,
		mcpProxy,
		metrics,
		logger,
	)
	if err != nil {
		logger.Fatal(ctx, "create bridge", slog.Error(err))
	}
	defer bridge.Shutdown(ctx)

	// Setup HTTP routes.
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
	mux.Handle("/", actorMiddleware(bridge))

	logger.Info(ctx, "listening on :8080")
	if err := http.ListenAndServe(":8080", mux); err != nil {
		logger.Fatal(ctx, "http server error", slog.Error(err))
	}
}

// actorMiddleware injects actor identity into request context.
// In production, extract user ID from auth headers/tokens.
func actorMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		userID := r.Header.Get("X-User-ID")
		if userID == "" {
			userID = "anonymous"
		}
		ctx := aibridge.AsActor(r.Context(), userID, nil)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}
