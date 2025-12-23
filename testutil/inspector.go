package testutil

import (
	"encoding/json"
	"testing"

	"github.com/coder/aibridge"
)

// Inspector provides a single place to access data needed for assertions.
//
// It is intentionally small; tests can always drop down to the underlying
// Recorder/MCP/Upstream objects when needed.
type Inspector struct {
	Recorder *RecorderSpy
	MCP      *MCPServer
	Upstream *UpstreamServer
}

func NewInspector(recorder *RecorderSpy, mcpServer *MCPServer, upstream *UpstreamServer) *Inspector {
	return &Inspector{Recorder: recorder, MCP: mcpServer, Upstream: upstream}
}

func (i *Inspector) UpstreamCalls() int {
	if i == nil || i.Upstream == nil {
		return 0
	}
	return i.Upstream.CallCount()
}

// RequireToolCalledOnceWithArgs asserts that:
//   - the bridge recorded exactly one tool usage with the given name
//   - the mock MCP server received exactly one call with matching args
func (i *Inspector) RequireToolCalledOnceWithArgs(t testing.TB, tool string, wantArgs any) {
	t.Helper()
	if i == nil {
		t.Fatalf("inspector is nil")
	}
	if i.Recorder == nil {
		t.Fatalf("inspector Recorder is nil")
	}
	if i.MCP == nil {
		t.Fatalf("inspector MCP is nil")
	}

	// Verify bridge-side record.
	var toolUsages []*aibridge.ToolUsageRecord
	for _, u := range i.Recorder.RecordedToolUsages() {
		if u.Tool == tool {
			toolUsages = append(toolUsages, u)
		}
	}
	if len(toolUsages) != 1 {
		t.Fatalf("tool usages for %q: got %d, want 1", tool, len(toolUsages))
	}

	wantJSON, err := json.Marshal(wantArgs)
	if err != nil {
		t.Fatalf("marshal wantArgs: %v", err)
	}
	gotJSON, err := json.Marshal(toolUsages[0].Args)
	if err != nil {
		t.Fatalf("marshal recorded tool args: %v", err)
	}
	if string(wantJSON) != string(gotJSON) {
		t.Fatalf("recorded tool args mismatch\nwant: %s\ngot:  %s", string(wantJSON), string(gotJSON))
	}

	// Verify MCP-side receipt.
	invocations := i.MCP.CallsByTool(tool)
	if len(invocations) != 1 {
		t.Fatalf("MCP calls for %q: got %d, want 1", tool, len(invocations))
	}
	gotJSON, err = json.Marshal(invocations[0])
	if err != nil {
		t.Fatalf("marshal MCP call args: %v", err)
	}
	if string(wantJSON) != string(gotJSON) {
		t.Fatalf("MCP call args mismatch\nwant: %s\ngot:  %s", string(wantJSON), string(gotJSON))
	}
}
