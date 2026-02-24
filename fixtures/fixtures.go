package fixtures

import (
	_ "embed"
	"testing"

	"golang.org/x/tools/txtar"
)

var (
	//go:embed anthropic/simple.txtar
	AntSimple []byte

	//go:embed anthropic/single_builtin_tool.txtar
	AntSingleBuiltinTool []byte

	//go:embed anthropic/single_injected_tool.txtar
	AntSingleInjectedTool []byte

	//go:embed anthropic/fallthrough.txtar
	AntFallthrough []byte

	//go:embed anthropic/stream_error.txtar
	AntMidStreamError []byte

	//go:embed anthropic/non_stream_error.txtar
	AntNonStreamError []byte
)

var (
	//go:embed openai/chatcompletions/simple.txtar
	OaiChatSimple []byte

	//go:embed openai/chatcompletions/single_builtin_tool.txtar
	OaiChatSingleBuiltinTool []byte

	//go:embed openai/chatcompletions/single_injected_tool.txtar
	OaiChatSingleInjectedTool []byte

	//go:embed openai/chatcompletions/fallthrough.txtar
	OaiChatFallthrough []byte

	//go:embed openai/chatcompletions/stream_error.txtar
	OaiChatMidStreamError []byte

	//go:embed openai/chatcompletions/non_stream_error.txtar
	OaiChatNonStreamError []byte

	//go:embed openai/chatcompletions/streaming_injected_tool_no_preamble.txtar
	OaiChatStreamingInjectedToolNoPreamble []byte

	//go:embed openai/chatcompletions/streaming_injected_tool_nonzero_index.txtar
	OaiChatStreamingInjectedToolNonzeroIndex []byte
)

var (
	//go:embed openai/responses/blocking/simple.txtar
	OaiResponsesBlockingSimple []byte

	//go:embed openai/responses/blocking/single_builtin_tool.txtar
	OaiResponsesBlockingSingleBuiltinTool []byte

	//go:embed openai/responses/blocking/cached_input_tokens.txtar
	OaiResponsesBlockingCachedInputTokens []byte

	//go:embed openai/responses/blocking/custom_tool.txtar
	OaiResponsesBlockingCustomTool []byte

	//go:embed openai/responses/blocking/conversation.txtar
	OaiResponsesBlockingConversation []byte

	//go:embed openai/responses/blocking/http_error.txtar
	OaiResponsesBlockingHttpErr []byte

	//go:embed openai/responses/blocking/prev_response_id.txtar
	OaiResponsesBlockingPrevResponseID []byte

	//go:embed openai/responses/blocking/single_injected_tool.txtar
	OaiResponsesBlockingSingleInjectedTool []byte

	//go:embed openai/responses/blocking/single_injected_tool_error.txtar
	OaiResponsesBlockingSingleInjectedToolError []byte

	//go:embed openai/responses/blocking/wrong_response_format.txtar
	OaiResponsesBlockingWrongResponseFormat []byte
)

var (
	//go:embed openai/responses/streaming/simple.txtar
	OaiResponsesStreamingSimple []byte

	//go:embed openai/responses/streaming/codex_example.txtar
	OaiResponsesStreamingCodex []byte

	//go:embed openai/responses/streaming/builtin_tool.txtar
	OaiResponsesStreamingBuiltinTool []byte

	//go:embed openai/responses/streaming/cached_input_tokens.txtar
	OaiResponsesStreamingCachedInputTokens []byte

	//go:embed openai/responses/streaming/custom_tool.txtar
	OaiResponsesStreamingCustomTool []byte

	//go:embed openai/responses/streaming/conversation.txtar
	OaiResponsesStreamingConversation []byte

	//go:embed openai/responses/streaming/http_error.txtar
	OaiResponsesStreamingHttpErr []byte

	//go:embed openai/responses/streaming/prev_response_id.txtar
	OaiResponsesStreamingPrevResponseID []byte

	//go:embed openai/responses/streaming/single_injected_tool.txtar
	OaiResponsesStreamingSingleInjectedTool []byte

	//go:embed openai/responses/streaming/single_injected_tool_error.txtar
	OaiResponsesStreamingSingleInjectedToolError []byte

	//go:embed openai/responses/streaming/stream_error.txtar
	OaiResponsesStreamingStreamError []byte

	//go:embed openai/responses/streaming/stream_failure.txtar
	OaiResponsesStreamingStreamFailure []byte

	//go:embed openai/responses/streaming/wrong_response_format.txtar
	OaiResponsesStreamingWrongResponseFormat []byte
)

// Archive file name constants matching the file names used in txtar fixtures.
const (
	fileRequest              = "request"
	fileStreamingResponse    = "streaming"
	fileNonStreamingResponse = "non-streaming"
	fileStreamingToolCall    = "streaming/tool-call"
	fileNonStreamingToolCall = "non-streaming/tool-call"
)

// Files maps txtar archive file names to their contents.
type Files map[string][]byte

func (f Files) Request() []byte {
	return f[fileRequest]
}

func (f Files) Streaming() []byte {
	return f[fileStreamingResponse]
}

func (f Files) NonStreaming() []byte {
	return f[fileNonStreamingResponse]
}

func (f Files) StreamingToolCall() []byte {
	return f[fileStreamingToolCall]
}

func (f Files) NonStreamingToolCall() []byte {
	return f[fileNonStreamingToolCall]
}

// ParseFiles parses raw txtar data into a Files map.
func ParseFiles(t *testing.T, data []byte) Files {
	t.Helper()

	archive := txtar.Parse(data)
	if len(archive.Files) == 0 {
		return nil
	}

	out := make(Files, len(archive.Files))
	for _, f := range archive.Files {
		out[f.Name] = f.Data
	}
	return out
}

// Request extracts the "request" fixture from raw txtar data.
func Request(t *testing.T, fixture []byte) []byte {
	t.Helper()
	return ParseFiles(t, fixture).Request()
}
