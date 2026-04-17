// Package bedrock provides a SigV4-signing reverse proxy interceptor
// for native Bedrock API requests. It forwards requests to AWS Bedrock
// with centralized AWS credentials and extracts audit metadata from
// the response stream.
package bedrock

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream/eventstreamapi"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/google/uuid"
	"github.com/tidwall/gjson"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/xerrors"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/intercept"
	"github.com/coder/aibridge/intercept/messages"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/recorder"
	"github.com/coder/aibridge/tracing"
)

var _ intercept.Interceptor = &Interceptor{}

// Interceptor is a SigV4-signing reverse proxy for native Bedrock API
// requests. It forwards the request body as-is, signs with centralized
// AWS credentials, and extracts audit metadata from the response.
type Interceptor struct {
	id           uuid.UUID
	modelID      string
	streaming    bool
	reqBody      []byte
	originalPath string
	bedrockCfg   config.AWSBedrock
	providerName string
	dumpDir      string
	tracer       trace.Tracer
	credential   intercept.CredentialInfo

	logger   slog.Logger
	recorder recorder.Recorder
}

func NewInterceptor(
	id uuid.UUID,
	modelID string,
	streaming bool,
	reqBody []byte,
	originalPath string,
	bedrockCfg config.AWSBedrock,
	providerName string,
	dumpDir string,
	tracer trace.Tracer,
	cred intercept.CredentialInfo,
) *Interceptor {
	return &Interceptor{
		id:           id,
		modelID:      modelID,
		streaming:    streaming,
		reqBody:      reqBody,
		originalPath: originalPath,
		bedrockCfg:   bedrockCfg,
		providerName: providerName,
		dumpDir:      dumpDir,
		tracer:       tracer,
		credential:   cred,
	}
}

func (i *Interceptor) ID() uuid.UUID { return i.id }
func (i *Interceptor) Model() string { return i.modelID }

func (i *Interceptor) Setup(logger slog.Logger, rec recorder.Recorder, _ mcp.ServerProxier) {
	i.logger = logger
	i.recorder = rec
}

func (i *Interceptor) Streaming() bool                      { return i.streaming }
func (i *Interceptor) Credential() intercept.CredentialInfo { return i.credential }
func (i *Interceptor) CorrelatingToolCallID() *string       { return nil }

func (i *Interceptor) TraceAttributes(r *http.Request) []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.String(tracing.Provider, i.providerName),
		attribute.String(tracing.Model, i.modelID),
		attribute.Bool(tracing.Streaming, i.streaming),
		attribute.Bool(tracing.IsBedrock, true),
	}
}

func (i *Interceptor) ProcessRequest(w http.ResponseWriter, r *http.Request) error {
	ctx := r.Context()
	_, span := i.tracer.Start(ctx, "bedrock.ProcessRequest")
	defer span.End()

	// Extract user prompt before sending the request.
	var promptText string
	var promptFound bool
	if reqPayload, err := messages.NewRequestPayload(i.reqBody); err == nil {
		promptText, promptFound, _ = reqPayload.LastUserPrompt()
	}

	baseURL := i.bedrockCfg.BaseURL
	if baseURL == "" {
		baseURL = "https://bedrock-runtime." + i.bedrockCfg.Region + ".amazonaws.com"
	}

	outReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+i.originalPath, bytes.NewReader(i.reqBody))
	if err != nil {
		return xerrors.Errorf("create outbound request: %w", err)
	}

	outReq.Header = intercept.PrepareClientHeaders(r.Header)
	outReq.Header.Set("Content-Type", "application/json")

	awsCreds, err := i.loadCredentials(ctx)
	if err != nil {
		return xerrors.Errorf("load AWS credentials: %w", err)
	}

	hash := sha256.Sum256(i.reqBody)
	signer := v4.NewSigner()
	if err = signer.SignHTTP(ctx, awsCreds, outReq, hex.EncodeToString(hash[:]), "bedrock", i.bedrockCfg.Region, time.Now()); err != nil {
		return xerrors.Errorf("sign request: %w", err)
	}

	resp, err := http.DefaultClient.Do(outReq)
	if err != nil {
		return xerrors.Errorf("send request to bedrock: %w", err)
	}
	defer resp.Body.Close()

	for key, values := range resp.Header {
		for _, val := range values {
			w.Header().Add(key, val)
		}
	}
	w.WriteHeader(resp.StatusCode)

	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(w, resp.Body)
		return nil
	}

	// Buffer the response while streaming to the client so we can
	// parse it for audit data after the stream completes.
	var auditBuf bytes.Buffer
	tee := io.TeeReader(resp.Body, &auditBuf)

	if i.streaming {
		flusher, ok := w.(http.Flusher)
		buf := make([]byte, 32*1024)
		for {
			n, readErr := tee.Read(buf)
			if n > 0 {
				if _, writeErr := w.Write(buf[:n]); writeErr != nil {
					return xerrors.Errorf("write streaming chunk: %w", writeErr)
				}
				if ok {
					flusher.Flush()
				}
			}
			if readErr != nil {
				if readErr == io.EOF {
					break
				}
				return xerrors.Errorf("read streaming chunk: %w", readErr)
			}
		}
	} else {
		if _, err = io.Copy(w, tee); err != nil {
			return xerrors.Errorf("copy response body: %w", err)
		}
	}

	respBytes := auditBuf.Bytes()

	// Dump request and response for debugging/fixture generation.
	i.dumpRequestResponse(ctx, respBytes)

	// Extract audit metadata from the buffered response.
	if i.streaming {
		i.extractStreamingAudit(ctx, respBytes, promptText, promptFound)
	} else {
		i.extractBlockingAudit(ctx, respBytes, promptText, promptFound)
	}

	return nil
}

// dumpRequestResponse writes the raw request body and response bytes
// to files for debugging and test fixture generation.
func (i *Interceptor) dumpRequestResponse(ctx context.Context, respBytes []byte) {
	if i.dumpDir == "" {
		return
	}

	safeModel := strings.ReplaceAll(i.modelID, "/", "-")
	dir := filepath.Join(i.dumpDir, i.providerName, safeModel)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		i.logger.Warn(ctx, "failed to create dump dir", slog.Error(err))
		return
	}

	base := filepath.Join(dir, fmt.Sprintf("%d-%s", time.Now().UTC().UnixMilli(), i.id))

	if err := os.WriteFile(base+".req.json", i.reqBody, 0o644); err != nil {
		i.logger.Warn(ctx, "failed to dump request", slog.Error(err))
	}

	suffix := ".resp.json"
	if i.streaming {
		suffix = ".resp.bin"
	}
	if err := os.WriteFile(base+suffix, respBytes, 0o644); err != nil {
		i.logger.Warn(ctx, "failed to dump response", slog.Error(err))
	}
}

// extractStreamingAudit parses a buffered AWS EventStream response
// and records audit metadata.
func (i *Interceptor) extractStreamingAudit(ctx context.Context, data []byte, promptText string, promptFound bool) {
	decoder := eventstream.NewDecoder()
	reader := bytes.NewReader(data)

	var msgID string

	// Accumulators for content blocks indexed by block position.
	type toolBlock struct {
		id   string
		name string
		args bytes.Buffer
	}
	var toolBlocks []toolBlock
	thinkingBlocks := map[int]*bytes.Buffer{}
	blockTypes := map[int]string{}

	for {
		msg, err := decoder.Decode(reader, nil)
		if err != nil {
			break
		}

		messageType := msg.Headers.Get(eventstreamapi.MessageTypeHeader)
		if messageType == nil || messageType.String() != eventstreamapi.EventMessageType {
			continue
		}
		eventType := msg.Headers.Get(eventstreamapi.EventTypeHeader)
		if eventType == nil || eventType.String() != "chunk" {
			continue
		}

		var chunk struct {
			Bytes string `json:"bytes"`
		}
		if err := json.Unmarshal(msg.Payload, &chunk); err != nil {
			continue
		}
		decoded, err := base64.StdEncoding.DecodeString(chunk.Bytes)
		if err != nil {
			continue
		}

		eventKind := gjson.GetBytes(decoded, "type").String()

		switch eventKind {
		case "message_start":
			msgID = gjson.GetBytes(decoded, "message.id").String()
			usage := gjson.GetBytes(decoded, "message.usage")
			if usage.Exists() {
				_ = i.recorder.RecordTokenUsage(ctx, &recorder.TokenUsageRecord{
					InterceptionID:        i.id.String(),
					MsgID:                 msgID,
					Input:                 usage.Get("input_tokens").Int(),
					Output:                usage.Get("output_tokens").Int(),
					CacheReadInputTokens:  usage.Get("cache_read_input_tokens").Int(),
					CacheWriteInputTokens: usage.Get("cache_creation_input_tokens").Int(),
				})
			}
			if promptFound {
				_ = i.recorder.RecordPromptUsage(ctx, &recorder.PromptUsageRecord{
					InterceptionID: i.id.String(),
					MsgID:          msgID,
					Prompt:         promptText,
				})
				promptFound = false
			}

		case "message_delta":
			usage := gjson.GetBytes(decoded, "usage")
			if usage.Exists() {
				_ = i.recorder.RecordTokenUsage(ctx, &recorder.TokenUsageRecord{
					InterceptionID: i.id.String(),
					MsgID:          msgID,
					Output:         usage.Get("output_tokens").Int(),
				})
			}

		case "content_block_start":
			idx := int(gjson.GetBytes(decoded, "index").Int())
			blockType := gjson.GetBytes(decoded, "content_block.type").String()
			blockTypes[idx] = blockType

			if blockType == "tool_use" {
				toolBlocks = append(toolBlocks, toolBlock{
					id:   gjson.GetBytes(decoded, "content_block.id").String(),
					name: gjson.GetBytes(decoded, "content_block.name").String(),
				})
			}
			if blockType == "thinking" {
				thinkingBlocks[idx] = &bytes.Buffer{}
			}

		case "content_block_delta":
			idx := int(gjson.GetBytes(decoded, "index").Int())
			switch blockTypes[idx] {
			case "tool_use":
				partialJSON := gjson.GetBytes(decoded, "delta.partial_json").String()
				for ti := range toolBlocks {
					if toolBlocks[ti].id != "" {
						toolBlocks[len(toolBlocks)-1].args.WriteString(partialJSON)
						break
					}
				}
			case "thinking":
				if buf, ok := thinkingBlocks[idx]; ok {
					buf.WriteString(gjson.GetBytes(decoded, "delta.thinking").String())
				}
			}

		case "message_stop":
			for _, tb := range toolBlocks {
				var args json.RawMessage
				if tb.args.Len() > 0 {
					args = json.RawMessage(tb.args.Bytes())
				}
				_ = i.recorder.RecordToolUsage(ctx, &recorder.ToolUsageRecord{
					InterceptionID: i.id.String(),
					MsgID:          msgID,
					ToolCallID:     tb.id,
					Tool:           tb.name,
					Args:           args,
					Injected:       false,
				})
			}
			for _, buf := range thinkingBlocks {
				if buf.Len() > 0 {
					_ = i.recorder.RecordModelThought(ctx, &recorder.ModelThoughtRecord{
						InterceptionID: i.id.String(),
						Content:        buf.String(),
						Metadata:       recorder.Metadata{"source": recorder.ThoughtSourceThinking},
					})
				}
			}
		}
	}
}

// extractBlockingAudit parses a JSON response body and records audit
// metadata.
func (i *Interceptor) extractBlockingAudit(ctx context.Context, data []byte, promptText string, promptFound bool) {
	msgID := gjson.GetBytes(data, "id").String()

	if promptFound {
		_ = i.recorder.RecordPromptUsage(ctx, &recorder.PromptUsageRecord{
			InterceptionID: i.id.String(),
			MsgID:          msgID,
			Prompt:         promptText,
		})
	}

	usage := gjson.GetBytes(data, "usage")
	if usage.Exists() {
		_ = i.recorder.RecordTokenUsage(ctx, &recorder.TokenUsageRecord{
			InterceptionID:        i.id.String(),
			MsgID:                 msgID,
			Input:                 usage.Get("input_tokens").Int(),
			Output:                usage.Get("output_tokens").Int(),
			CacheReadInputTokens:  usage.Get("cache_read_input_tokens").Int(),
			CacheWriteInputTokens: usage.Get("cache_creation_input_tokens").Int(),
		})
	}

	content := gjson.GetBytes(data, "content")
	if content.IsArray() {
		content.ForEach(func(_, block gjson.Result) bool {
			switch block.Get("type").String() {
			case "tool_use":
				_ = i.recorder.RecordToolUsage(ctx, &recorder.ToolUsageRecord{
					InterceptionID: i.id.String(),
					MsgID:          msgID,
					ToolCallID:     block.Get("id").String(),
					Tool:           block.Get("name").String(),
					Args:           json.RawMessage(block.Get("input").Raw),
					Injected:       false,
				})
			case "thinking":
				thinking := block.Get("thinking").String()
				if thinking != "" {
					_ = i.recorder.RecordModelThought(ctx, &recorder.ModelThoughtRecord{
						InterceptionID: i.id.String(),
						Content:        thinking,
						Metadata:       recorder.Metadata{"source": recorder.ThoughtSourceThinking},
					})
				}
			}
			return true
		})
	}
}

func (i *Interceptor) loadCredentials(ctx context.Context) (aws.Credentials, error) {
	loadOpts := []func(*awsconfig.LoadOptions) error{
		awsconfig.WithRegion(i.bedrockCfg.Region),
	}

	switch {
	case i.bedrockCfg.AccessKey != "" && i.bedrockCfg.AccessKeySecret != "":
		loadOpts = append(loadOpts, awsconfig.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(
				i.bedrockCfg.AccessKey,
				i.bedrockCfg.AccessKeySecret,
				i.bedrockCfg.SessionToken,
			),
		))
	case i.bedrockCfg.AccessKey != "" || i.bedrockCfg.AccessKeySecret != "":
		return aws.Credentials{}, xerrors.New("both access key and access key secret must be provided together")
	}

	cfg, err := awsconfig.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		return aws.Credentials{}, xerrors.Errorf("load AWS config: %w", err)
	}

	creds, err := cfg.Credentials.Retrieve(ctx)
	if err != nil {
		return aws.Credentials{}, xerrors.Errorf("retrieve AWS credentials: %w", err)
	}

	return creds, nil
}
