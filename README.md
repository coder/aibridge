# AIBridge

*Intercept AI requests, track usage, inject MCP tools centrally*

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Go Reference](https://pkg.go.dev/badge/github.com/coder/aibridge.svg)](https://pkg.go.dev/github.com/coder/aibridge)

## Overview

AIBridge is a Go library that provides a centralized governance layer for AI provider interactions. It acts as an HTTP proxy between AI clients and upstream AI providers (OpenAI, Anthropic, AWS Bedrock), enabling:

- **Usage Tracking**: Record token consumption, prompts, and tool usage across all AI interactions
- **MCP Tool Injection**: Centrally inject Model Context Protocol (MCP) tools into AI requests without client-side changes
- **Multi-Provider Support**: Unified interface for OpenAI, Anthropic, and AWS Bedrock APIs
- **Request Interception**: Transform requests and responses in-flight for governance and augmentation
- **Graceful Shutdown**: Proper handling of in-flight requests and resource cleanup

## Architecture

![AIBridge Architecture](https://blink.so/api/files/5102455c-33cc-4b3f-9413-11d8510c9d01)

### Core Components

#### RequestBridge
The main HTTP handler that routes requests to appropriate providers and manages interception lifecycle.

```go
type RequestBridge struct {
    // HTTP multiplexer for routing
    mux *http.ServeMux
    
    // MCP server proxy for tool injection
    mcpProxy mcp.ServerProxier
    
    // Tracks in-flight requests for graceful shutdown
    inflightReqs atomic.Int32
    inflightWG   sync.WaitGroup
}
```

#### Provider Interface
Defines how AI providers interact with the bridge:

```go
type Provider interface {
    Name() string
    BaseURL() string
    CreateInterceptor(w http.ResponseWriter, r *http.Request) (Interceptor, error)
    BridgedRoutes() []string
    PassthroughRoutes() []string
    AuthHeader() string
    InjectAuthHeader(*http.Header)
}
```

Supported providers:
- `OpenAIProvider` - OpenAI API (including Azure OpenAI)
- `AnthropicProvider` - Anthropic Claude API
- AWS Bedrock support via Anthropic provider

#### Interceptor
Handles individual request/response flows with provider-specific logic:

```go
type Interceptor interface {
    ID() uuid.UUID
    Setup(logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier)
    Model() string
    ProcessRequest(w http.ResponseWriter, r *http.Request) error
}
```

Implementations:
- Streaming and non-streaming variants
- Tool injection and result handling
- Token usage calculation
- Error handling and retry logic

#### Recorder
Captures all usage data for governance and analytics:

```go
type Recorder interface {
    RecordInterception(ctx context.Context, req *InterceptionRecord) error
    RecordInterceptionEnded(ctx context.Context, req *InterceptionRecordEnded) error
    RecordTokenUsage(ctx context.Context, req *TokenUsageRecord) error
    RecordPromptUsage(ctx context.Context, req *PromptUsageRecord) error
    RecordToolUsage(ctx context.Context, req *ToolUsageRecord) error
}
```

#### MCP Integration
Model Context Protocol (MCP) server integration enables centralized tool management:

- `ServerProxyManager` - Aggregates tools from multiple MCP servers
- `ServerProxier` - Interface for MCP server communication
- Tool namespacing with `bmcp_` prefix to avoid conflicts
- Automatic tool injection into AI requests
- Tool allowlist/denylist filtering

## Usage

### Basic Setup

```go
import (
    "context"
    "net/http"
    
    "cdr.dev/slog"
    "github.com/coder/aibridge"
    "github.com/coder/aibridge/mcp"
)

func main() {
    ctx := context.Background()
    logger := slog.Make(sloghuman.Sink(os.Stdout))
    
    // Configure providers
    providers := []aibridge.Provider{
        aibridge.NewOpenAIProvider(aibridge.OpenAIConfig{
            BaseURL: "https://api.openai.com/v1/",
            Key:     os.Getenv("OPENAI_API_KEY"),
        }),
        aibridge.NewAnthropicProvider(aibridge.AnthropicConfig{
            BaseURL: "https://api.anthropic.com/",
            Key:     os.Getenv("ANTHROPIC_API_KEY"),
        }, nil),
    }
    
    // Create recorder (implement your own or use a mock)
    recorder := NewDatabaseRecorder(db)
    
    // Setup MCP tool injection (optional)
    mcpProxy := mcp.NewServerProxyManager(nil)
    
    // Create the bridge
    bridge, err := aibridge.NewRequestBridge(
        ctx,
        providers,
        logger,
        recorder,
        mcpProxy,
    )
    if err != nil {
        log.Fatal(err)
    }
    
    // Wrap with actor context (for user tracking)
    handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Extract user ID from your auth system
        userID := getUserFromAuth(r)
        ctx := aibridge.AsActor(r.Context(), userID, map[string]any{
            "org_id": "org-123",
        })
        bridge.ServeHTTP(w, r.WithContext(ctx))
    })
    
    // Start server
    server := &http.Server{
        Addr:    ":8080",
        Handler: handler,
    }
    
    log.Fatal(server.ListenAndServe())
}
```

### Using with Coder

The [coder/coder](https://github.com/coder/coder) repository uses AIBridge for its AI features. Example from `enterprise/cli/aibridged.go`:

```go
// Setup supported providers
providers := []aibridge.Provider{
    aibridge.NewOpenAIProvider(aibridge.OpenAIConfig{
        BaseURL: deploymentConfig.AI.BridgeConfig.OpenAI.BaseURL.String(),
        Key:     deploymentConfig.AI.BridgeConfig.OpenAI.Key.String(),
    }),
    aibridge.NewAnthropicProvider(aibridge.AnthropicConfig{
        BaseURL: deploymentConfig.AI.BridgeConfig.Anthropic.BaseURL.String(),
        Key:     deploymentConfig.AI.BridgeConfig.Anthropic.Key.String(),
    }, getBedrockConfig(deploymentConfig.AI.BridgeConfig.Bedrock)),
}

// Create pool for reusable stateful RequestBridge instances (one per user)
pool, err := aibridged.NewCachedBridgePool(
    aibridged.DefaultPoolOptions,
    providers,
    logger.Named("pool"),
)

// Create daemon that integrates with Coder's database
srv, err := aibridged.New(ctx, pool, dialFunc, logger)
```

### AWS Bedrock Support

```go
bedrockConfig := &aibridge.AWSBedrockConfig{
    Region:          "us-east-1",
    AccessKey:       os.Getenv("AWS_ACCESS_KEY_ID"),
    AccessKeySecret: os.Getenv("AWS_SECRET_ACCESS_KEY"),
    Model:           "anthropic.claude-3-5-sonnet-20241022-v2:0",
    SmallFastModel:  "anthropic.claude-3-5-haiku-20241022-v1:0",
}

provider := aibridge.NewAnthropicProvider(
    aibridge.AnthropicConfig{},
    bedrockConfig,
)
```

### Actor Context

AIBridge requires actor context for tracking who initiated each request:

```go
// Create actor context with user ID and optional metadata
ctx := aibridge.AsActor(ctx, "user-123", map[string]any{
    "org_id":    "org-456",
    "workspace": "dev-env",
})

// Use in request
req = req.WithContext(ctx)
```

### Implementing a Recorder

```go
type DatabaseRecorder struct {
    db *sql.DB
}

func (r *DatabaseRecorder) RecordInterception(ctx context.Context, req *aibridge.InterceptionRecord) error {
    _, err := r.db.ExecContext(ctx,
        `INSERT INTO aibridge_interceptions (id, initiator_id, provider, model, metadata, started_at)
         VALUES ($1, $2, $3, $4, $5, $6)`,
        req.ID, req.InitiatorID, req.Provider, req.Model, req.Metadata, req.StartedAt,
    )
    return err
}

func (r *DatabaseRecorder) RecordTokenUsage(ctx context.Context, req *aibridge.TokenUsageRecord) error {
    _, err := r.db.ExecContext(ctx,
        `INSERT INTO aibridge_token_usages (interception_id, msg_id, input, output, metadata, created_at)
         VALUES ($1, $2, $3, $4, $5, $6)`,
        req.InterceptionID, req.MsgID, req.Input, req.Output, req.Metadata, req.CreatedAt,
    )
    return err
}

// Implement remaining Recorder methods...
```

### MCP Tool Integration

```go
import "github.com/coder/aibridge/mcp"

// Create MCP server proxies
proxiers := map[string]mcp.ServerProxier{
    "filesystem": NewFileSystemMCPProxy(),
    "git":        NewGitMCPProxy(),
}

mcpProxy := mcp.NewServerProxyManager(proxiers)

// Initialize MCP servers and load tools
if err := mcpProxy.Init(ctx); err != nil {
    log.Fatal(err)
}

// Tools are now automatically injected into AI requests
bridge, err := aibridge.NewRequestBridge(
    ctx,
    providers,
    logger,
    recorder,
    mcpProxy,
)
```

### Graceful Shutdown

```go
// Create shutdown context
shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

// Shutdown waits for in-flight requests and closes MCP servers
if err := bridge.Shutdown(shutdownCtx); err != nil {
    logger.Error(ctx, "shutdown error", slog.Error(err))
}
```

## Supported Routes

### OpenAI Provider

**Bridged (Intercepted):**
- `POST /openai/v1/chat/completions` - Chat completions with streaming support

**Passthrough:**
- `GET /openai/v1/models` - List available models
- `POST /openai/v1/responses` - Responses API (TODO: full support)

### Anthropic Provider

**Bridged (Intercepted):**
- `POST /anthropic/v1/messages` - Messages API with streaming support

**Passthrough:**
- `GET /anthropic/v1/models` - List available models
- `POST /anthropic/v1/messages/count_tokens` - Token counting

## LLM Provider Compatibility

| Provider | Status | Features |
|----------|--------|----------|
| OpenAI | ✅ Full | Chat completions, streaming, tool calls, vision |
| Anthropic | ✅ Full | Messages API, streaming, tool use, Claude models |
| AWS Bedrock (Anthropic) | ✅ Full | Cross-region inference, all Claude models |
| Azure OpenAI | ✅ Full | Use OpenAI provider with Azure base URL |

## Key Features

### Request Interception
- Transparent proxy between client and upstream API
- Transform requests to inject MCP tools
- Parse streaming and non-streaming responses
- Extract usage data (tokens, prompts, tool calls)

### Tool Injection
- Automatic injection of MCP tools into requests
- Tool namespacing to prevent conflicts (`bmcp_` prefix)
- Allowlist/denylist filtering
- Multi-server tool aggregation

### Usage Tracking
- Token consumption (input/output)
- User prompts and AI responses
- Tool invocations and results
- Error tracking and debugging

### Streaming Support
- Both providers support streaming responses
- SSE (Server-Sent Events) parsing
- Incremental token counting
- Tool call detection in streams

## Testing

The repository includes comprehensive integration tests with fixtures:

```bash
go test -v ./...
```

Test fixtures are in `fixtures/` directory using txtar format:
- `anthropic/simple.txtar` - Basic message flow
- `anthropic/single_builtin_tool.txtar` - Tool usage
- `anthropic/single_injected_tool.txtar` - MCP tool injection
- `openai/simple.txtar` - OpenAI chat completions
- `openai/single_builtin_tool.txtar` - Function calling

## Database Schema

Example schema used in coder/coder (see `coderd/database/migrations/000370_aibridge.up.sql`):

```sql
CREATE TABLE aibridge_interceptions (
    id UUID PRIMARY KEY,
    initiator_id UUID NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    metadata JSONB,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP
);

CREATE TABLE aibridge_token_usages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interception_id UUID NOT NULL REFERENCES aibridge_interceptions(id),
    msg_id TEXT NOT NULL,
    input BIGINT NOT NULL,
    output BIGINT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE aibridge_user_prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interception_id UUID NOT NULL REFERENCES aibridge_interceptions(id),
    msg_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE aibridge_tool_usages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interception_id UUID NOT NULL REFERENCES aibridge_interceptions(id),
    msg_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    server_url TEXT,
    args JSONB,
    injected BOOLEAN NOT NULL,
    invocation_error TEXT,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL
);
```

## Monitoring

Coder provides Grafana dashboards for AIBridge monitoring:
- Token usage by user/model
- Cost tracking with LiteLLM pricing data
- Tool usage analytics
- Error rates and latency

See: `examples/monitoring/dashboards/grafana/aibridge/` in coder/coder repo

## Contributing

Contributions are welcome! Please:
1. Add tests for new features
2. Follow existing code patterns
3. Update documentation
4. Run `make test` before submitting

## License

AGPL v3.0 - See [LICENSE](LICENSE) file

## Links

- [Coder Documentation](https://coder.com/docs)
- [Model Context Protocol](https://spec.modelcontextprotocol.io/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic API Reference](https://docs.anthropic.com/en/api)
