package fixtures

import (
	_ "embed"
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
)

var (
	//go:embed openai/responses/blocking/simple.txtar
	OaiResponsesBlockingSimple []byte

	//go:embed openai/responses/blocking/builtin_tool.txtar
	OaiResponsesBlockingBuiltinTool []byte

	//go:embed openai/responses/blocking/conversation.txtar
	OaiResponsesBlockingConversation []byte

	//go:embed openai/responses/blocking/prev_response_id.txtar
	OaiResponsesBlockingPrevResponseID []byte

	//go:embed openai/responses/blocking/wrong_response_format.txtar
	OaiResponsesBlockingWrongResponseFormat []byte
)

var (
	//go:embed openai/responses/streaming/simple.txtar
	OaiResponsesStreamingSimple []byte

	//go:embed openai/responses/streaming/codex_example.txtar
	ResponsesStreamingCodex []byte

	//go:embed openai/responses/streaming/builtin_tool.txtar
	OaiResponsesStreamingBuiltinTool []byte

	//go:embed openai/responses/streaming/conversation.txtar
	OaiResponsesStreamingConversation []byte

	//go:embed openai/responses/streaming/prev_response_id.txtar
	OaiResponsesStreamingPrevResponseID []byte

	//go:embed openai/responses/streaming/stream_error.txtar
	OaiResponsesStreamingStreamError []byte

	//go:embed openai/responses/streaming/stream_failure.txtar
	OaiResponsesStreamingStreamFailure []byte

	//go:embed openai/responses/streaming/wrong_response_format.txtar
	OaiResponsesStreamingWrongResponseFormat []byte
)
