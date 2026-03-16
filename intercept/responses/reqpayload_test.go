package responses

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestNewResponsesRequestPayload(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		raw    []byte
		want   string
		model  string
		stream bool
		err    string
	}{
		{
			name: "empty payload",
			raw:  nil,
			err:  "empty request body",
		},
		{
			name: "invalid json",
			raw:  []byte(`{broken`),
			err:  "invalid JSON payload",
		},
		{
			// ResponsesRequestPayload just checks for JSON validity,
			// schema errors are not surfaced here and
			// the original body is preserved for upstream handling
			// similar to how reverse proxy would behave.
			name:   "wrong field types still wrap",
			raw:    []byte(`{"model":123,"stream":"yes","input":42}`),
			want:   `{"model":123,"stream":"yes","input":42}`,
			model:  "123",
			stream: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			payload, err := NewResponsesRequestPayload(tc.raw)
			if tc.err != "" {
				require.ErrorContains(t, err, tc.err)
				assert.Nil(t, payload)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tc.want, string(payload))
			assert.Equal(t, tc.model, payload.model())
			assert.Equal(t, tc.stream, payload.Stream())
		})
	}
}

func TestWithInjectedTools(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		raw       []byte
		injected  []responses.ToolUnionParam
		wantNames []string
		wantErr   string
		wantSame  bool
	}{
		{
			name:      "appends to existing tools",
			raw:       []byte(`{"model":"gpt-4o","input":"hello","tools":[{"type":"function","name":"existing"}]}`),
			injected:  []responses.ToolUnionParam{injectedFunctionTool("injected")},
			wantNames: []string{"existing", "injected"},
		},
		{
			name:      "adds tools when none exist",
			raw:       []byte(`{"model":"gpt-4o","input":"hello"}`),
			injected:  []responses.ToolUnionParam{injectedFunctionTool("injected")},
			wantNames: []string{"injected"},
		},
		{
			name: "appends multiple injected tools",
			raw:  []byte(`{"model":"gpt-4o","input":"hello","tools":[{"type":"function","name":"existing"}]}`),
			injected: []responses.ToolUnionParam{
				injectedFunctionTool("injected-one"),
				injectedFunctionTool("injected-two"),
			},
			wantNames: []string{"existing", "injected-one", "injected-two"},
		},
		{
			name:     "empty injected tools is no op",
			raw:      []byte(`{"model":"gpt-4o","input":"hello","tools":[{"type":"function","name":"existing"}]}`),
			wantSame: true,
		},
		{
			name:     "errors on unsupported tools shape",
			raw:      []byte(`{"model":"gpt-4o","input":"hello","tools":"bad"}`),
			injected: []responses.ToolUnionParam{injectedFunctionTool("injected")},
			wantErr:  "failed to get existing tools: unsupported 'tools' type: String",
			wantSame: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			p := mustPayload(t, tc.raw)
			updated, err := p.injectTools(tc.injected)
			if tc.wantErr != "" {
				require.EqualError(t, err, tc.wantErr)
			} else {
				require.NoError(t, err)
			}

			if tc.wantSame {
				require.Equal(t, p, updated)
			}
			for i, wantName := range tc.wantNames {
				path := fmt.Sprintf("tools.%d.name", i) // name of the i-th element in tools array
				require.Equal(t, wantName, gjson.GetBytes(updated, path).String())
			}
		})
	}
}

func TestWithParallelToolCallsDisabled(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		raw         []byte
		wantErr     string
		wantSame    bool
		wantFlagSet bool
	}{
		{
			name:        "sets flag when tools exist",
			raw:         []byte(`{"tools":[{"type":"function","name":"existing"}]}`),
			wantFlagSet: true,
		},
		{
			name:     "no tools is no op",
			raw:      []byte(`{"model":"gpt-4o"}`),
			wantSame: true,
		},
		{
			name:     "invalid tools type",
			raw:      []byte(`{"tools":"bad"}`),
			wantErr:  "failed to get existing tools: unsupported 'tools' type: String",
			wantSame: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			p := mustPayload(t, tc.raw)
			updated, err := p.disableParallelToolCalls()
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
			} else {
				require.NoError(t, err)
			}

			if tc.wantSame {
				assert.Equal(t, p, updated)
			}
			if tc.wantFlagSet {
				assert.False(t, gjson.GetBytes(updated, "parallel_tool_calls").Bool())
			}
		})
	}
}

func TestWithAppendedInputItems(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		raw       []byte
		items     []responses.ResponseInputItemUnionParam
		wantErr   string
		wantSame  bool
		wantPaths map[string]string
	}{
		{
			name:  "string input becomes user message",
			raw:   []byte(`{"model":"gpt-4o","input":"hello"}`),
			items: []responses.ResponseInputItemUnionParam{responses.ResponseInputItemParamOfFunctionCallOutput("call_123", "done")},
			wantPaths: map[string]string{
				"input.0.role":    "user",
				"input.0.content": "hello",
				"input.1.type":    "function_call_output",
				"input.1.call_id": "call_123",
			},
		},
		{
			name:  "array input is preserved and appended",
			raw:   []byte(`{"model":"gpt-4o","input":[{"role":"user","content":"hello"}]}`),
			items: []responses.ResponseInputItemUnionParam{responses.ResponseInputItemParamOfFunctionCallOutput("call_123", "done")},
			wantPaths: map[string]string{
				"input.0.content": "hello",
				"input.1.call_id": "call_123",
			},
		},
		{
			name:     "unsupported input shape errors during rewrite",
			raw:      []byte(`{"model":"gpt-4o","input":123}`),
			items:    []responses.ResponseInputItemUnionParam{responses.ResponseInputItemParamOfFunctionCallOutput("call_123", "done")},
			wantErr:  "failed to get existing 'input' items: unsupported 'input' type: Number",
			wantSame: true,
		},
		{
			name:  "missing input creates appended input",
			raw:   []byte(`{"model":"gpt-4o"}`),
			items: []responses.ResponseInputItemUnionParam{responses.ResponseInputItemParamOfFunctionCallOutput("call_123", "done")},
			wantPaths: map[string]string{
				"input.0.type":    "function_call_output",
				"input.0.call_id": "call_123",
			},
		},
		{
			name:  "null input creates appended input",
			raw:   []byte(`{"model":"gpt-4o","input":null}`),
			items: []responses.ResponseInputItemUnionParam{responses.ResponseInputItemParamOfFunctionCallOutput("call_123", "done")},
			wantPaths: map[string]string{
				"input.0.type":    "function_call_output",
				"input.0.call_id": "call_123",
			},
		},
		{
			name: "multiple output item types are appended in order",
			raw:  []byte(`{"model":"gpt-4o","input":[{"role":"user","content":"hello"}]}`),
			items: []responses.ResponseInputItemUnionParam{
				responses.ResponseInputItemParamOfCompaction("encrypted-content"),
				responses.ResponseInputItemParamOfOutputMessage([]responses.ResponseOutputMessageContentUnionParam{
					{
						OfOutputText: &responses.ResponseOutputTextParam{
							Annotations: []responses.ResponseOutputTextAnnotationUnionParam{},
							Text:        "assistant text",
						},
					},
				}, "msg_123", responses.ResponseOutputMessageStatusCompleted),
				responses.ResponseInputItemParamOfFileSearchCall("fs_123", []string{"hello"}, "completed"),
				responses.ResponseInputItemParamOfImageGenerationCall("img_123", "base64-image", "completed"),
			},
			wantPaths: map[string]string{
				"input.0.content":        "hello",
				"input.1.type":           "compaction",
				"input.2.type":           "message",
				"input.2.id":             "msg_123",
				"input.2.content.0.type": "output_text",
				"input.2.content.0.text": "assistant text",
				"input.3.type":           "file_search_call",
				"input.3.id":             "fs_123",
				"input.4.type":           "image_generation_call",
				"input.4.id":             "img_123",
			},
		},
		{
			name:     "empty appended items is no op",
			raw:      []byte(`{"model":"gpt-4o","input":"hello"}`),
			wantSame: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			p := mustPayload(t, tc.raw)
			updated, err := p.appendInputItems(tc.items)

			if tc.wantErr != "" {
				require.EqualError(t, err, tc.wantErr)
			} else {
				require.NoError(t, err)
			}

			if tc.wantSame {
				require.Equal(t, p, updated)
			}

			for path, want := range tc.wantPaths {
				require.Equal(t, want, gjson.GetBytes(updated, path).String())
			}
		})
	}
}

func TestChainedRewritesProduceValidJSON(t *testing.T) {
	t.Parallel()

	p := mustPayload(t, []byte(`{"model":"gpt-4o","input":"hello"}`))
	p, err := p.injectTools([]responses.ToolUnionParam{{
		OfFunction: &responses.FunctionToolParam{
			Name:        "tool_a",
			Description: openai.String("tool"),
			Strict:      openai.Bool(false),
			Parameters: map[string]any{
				"type": "object",
			},
		},
	}})
	require.NoError(t, err)
	p, err = p.disableParallelToolCalls()
	require.NoError(t, err)
	p, err = p.appendInputItems([]responses.ResponseInputItemUnionParam{
		responses.ResponseInputItemParamOfFunctionCallOutput("call_123", "done"),
	})
	require.NoError(t, err)

	assert.True(t, json.Valid(p), "chained rewrites should produce valid JSON")
	assert.Equal(t, "tool_a", gjson.GetBytes(p, "tools.0.name").String())
	assert.Equal(t, "call_123", gjson.GetBytes(p, "input.1.call_id").String())
	assert.False(t, gjson.GetBytes(p, "parallel_tool_calls").Bool())
}

func injectedFunctionTool(name string) responses.ToolUnionParam {
	return responses.ToolUnionParam{
		OfFunction: &responses.FunctionToolParam{
			Name:        name,
			Description: openai.String("tool"),
			Strict:      openai.Bool(false),
			Parameters: map[string]any{
				"type": "object",
			},
		},
	}
}

func mustPayload(t *testing.T, raw []byte) ResponsesRequestPayload {
	t.Helper()

	payload, err := NewResponsesRequestPayload(raw)
	require.NoError(t, err)
	return payload
}
