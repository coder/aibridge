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
	OaiSimple []byte

	//go:embed openai/chatcompletions/single_builtin_tool.txtar
	OaiSingleBuiltinTool []byte

	//go:embed openai/chatcompletions/single_injected_tool.txtar
	OaiSingleInjectedTool []byte

	//go:embed openai/chatcompletions/fallthrough.txtar
	OaiFallthrough []byte

	//go:embed openai/chatcompletions/stream_error.txtar
	OaiMidStreamError []byte

	//go:embed openai/chatcompletions/non_stream_error.txtar
	OaiNonStreamError []byte
)

var (
	//go:embed openai/responses/blocking/simple.txtar
	ResponsesBlockingSimple []byte

	//go:embed openai/responses/blocking/builtin_tool.txtar
	ResponsesBlockingBuiltinTool []byte

	//go:embed openai/responses/blocking/conversation.txtar
	ResponsesBlockingConversation []byte

	//go:embed openai/responses/blocking/prev_response_id.txtar
	ResponsesBlockingPrevResponseID []byte

	//go:embed openai/responses/blocking/wrong_response_format.txtar
	ResponsesBlockingWrongResponseFormat []byte
)

var (
	//go:embed openai/responses/streaming/simple.txtar
	ResponsesStreamingSimple []byte

	//go:embed openai/responses/streaming/builtin_tool.txtar
	ResponsesStreamingBuiltinTool []byte

	//go:embed openai/responses/streaming/conversation.txtar
	ResponsesStreamingConversation []byte

	//go:embed openai/responses/streaming/prev_response_id.txtar
	ResponsesStreamingPrevResponseID []byte

	//go:embed openai/responses/streaming/stream_error.txtar
	ResponsesStreamingStreamError []byte

	//go:embed openai/responses/streaming/stream_failure.txtar
	ResponsesStreamingStreamFailure []byte

	//go:embed openai/responses/streaming/wrong_response_format.txtar
	ResponsesStreamingWrongResponseFormat []byte
)
