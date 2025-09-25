package mcp_test

import (
	"regexp"
	"testing"

	"cdr.dev/slog"
	"go.uber.org/goleak"

	"github.com/coder/aibridge/mcp"
	"github.com/stretchr/testify/require"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

func TestFilterAllowedTools(t *testing.T) {
	t.Parallel()

	createTools := func(names ...string) map[string]*mcp.Tool {
		tools := make(map[string]*mcp.Tool)
		for i, name := range names {
			id := string(rune('a' + i))
			tools[id] = &mcp.Tool{
				ID:   id,
				Name: name,
			}
		}
		return tools
	}

	mustCompile := func(pattern string) *regexp.Regexp {
		if pattern == "" {
			return nil
		}
		return regexp.MustCompile(pattern)
	}

	tests := []struct {
		name      string
		tools     map[string]*mcp.Tool
		allowlist string
		denylist  string
		expected  []string
	}{
		{
			name:      "empty tools returns empty",
			tools:     map[string]*mcp.Tool{},
			allowlist: ".*",
			denylist:  "",
			expected:  []string{},
		},
		{
			name:      "nil allow and deny lists returns all tools",
			tools:     createTools("tool1", "tool2", "tool3"),
			allowlist: "",
			denylist:  "",
			expected:  []string{"tool1", "tool2", "tool3"},
		},
		{
			name:      "allowlist only - match all",
			tools:     createTools("tool1", "tool2", "tool3"),
			allowlist: ".*",
			denylist:  "",
			expected:  []string{"tool1", "tool2", "tool3"},
		},
		{
			name:      "allowlist only - match specific",
			tools:     createTools("tool1", "tool2", "tool3"),
			allowlist: "tool[12]",
			denylist:  "",
			expected:  []string{"tool1", "tool2"},
		},
		{
			name:      "allowlist only - match none",
			tools:     createTools("tool1", "tool2", "tool3"),
			allowlist: "nonexistent",
			denylist:  "",
			expected:  []string{},
		},
		{
			name:      "denylist only - deny all",
			tools:     createTools("tool1", "tool2", "tool3"),
			allowlist: "",
			denylist:  ".*",
			expected:  []string{},
		},
		{
			name:      "denylist only - deny specific",
			tools:     createTools("tool1", "tool2", "tool3"),
			allowlist: "",
			denylist:  "tool2",
			expected:  []string{"tool1", "tool3"},
		},
		{
			name:      "denylist only - deny none",
			tools:     createTools("tool1", "tool2", "tool3"),
			allowlist: "",
			denylist:  "nonexistent",
			expected:  []string{"tool1", "tool2", "tool3"},
		},
		{
			name:      "both lists - no conflict",
			tools:     createTools("tool1", "tool2", "tool3", "tool4"),
			allowlist: "tool[124]",
			denylist:  "tool3",
			expected:  []string{"tool1", "tool2", "tool4"},
		},
		{
			name:      "both lists - denylist supersedes allowlist",
			tools:     createTools("tool1", "tool2", "tool3"),
			allowlist: "tool.*",
			denylist:  "tool2",
			expected:  []string{"tool1", "tool3"},
		},
		{
			name:      "both lists - complete conflict (denylist wins)",
			tools:     createTools("tool1", "tool2", "tool3"),
			allowlist: ".*",
			denylist:  ".*",
			expected:  []string{},
		},
		{
			name:      "both lists - partial overlap conflict",
			tools:     createTools("read_file", "write_file", "delete_file", "list_files"),
			allowlist: ".*_file",
			denylist:  "delete.*",
			expected:  []string{"read_file", "write_file", "list_files"},
		},
		{
			name:      "regex patterns - word boundaries",
			tools:     createTools("test", "testing", "pretest", "test123"),
			allowlist: "^test$",
			denylist:  "",
			expected:  []string{"test"},
		},
		{
			name:      "regex patterns - alternation in allowlist",
			tools:     createTools("read", "write", "execute", "delete"),
			allowlist: "read|write",
			denylist:  "",
			expected:  []string{"read", "write"},
		},
		{
			name:      "regex patterns - alternation in denylist",
			tools:     createTools("read", "write", "execute", "delete"),
			allowlist: "",
			denylist:  "execute|delete",
			expected:  []string{"read", "write"},
		},
		{
			name:      "complex regex - character classes",
			tools:     createTools("tool1", "tool2", "toolA", "toolB", "tool_special"),
			allowlist: "tool[A-Z]",
			denylist:  "",
			expected:  []string{"toolA", "toolB"},
		},
		{
			name:      "case sensitivity",
			tools:     createTools("Tool", "tool", "TOOL"),
			allowlist: "^tool$",
			denylist:  "",
			expected:  []string{"tool"},
		},
		{
			name:      "special characters in tool names",
			tools:     createTools("tool.test", "tool-test", "tool_test", "tool$test"),
			allowlist: `tool\.test`,
			denylist:  "",
			expected:  []string{"tool.test"},
		},
		{
			name:      "empty string tool name",
			tools:     createTools("", "tool1", "tool2"),
			allowlist: "tool.*",
			denylist:  "",
			expected:  []string{"tool1", "tool2"},
		},
		{
			name:      "unicode in tool names",
			tools:     createTools("工具1", "工具2", "tool3"),
			allowlist: "工具.*",
			denylist:  "",
			expected:  []string{"工具1", "工具2"},
		},
		{
			name:      "whitespace in tool names",
			tools:     createTools("tool 1", "tool  2", "tool\t3", "tool4"),
			allowlist: `tool\s+\d`,
			denylist:  "",
			expected:  []string{"tool 1", "tool  2", "tool\t3"},
		},
		{
			name:      "with both lists unmatched items are denied",
			tools:     createTools("foo1", "bar1", "other1", "other2"),
			allowlist: "^foo",
			denylist:  "^bar",
			expected:  []string{"foo1"}, // Only items matching allowlist (and not denylist).
		},
		{
			name:      "complex overlap - denylist pattern subset of allowlist",
			tools:     createTools("api_read", "api_write", "api_read_sensitive", "api_write_sensitive"),
			allowlist: "^api_.*",
			denylist:  ".*_sensitive$",
			expected:  []string{"api_read", "api_write"},
		},
		{
			name:      "nil tools map",
			tools:     nil,
			allowlist: ".*",
			denylist:  ".*",
			expected:  []string{},
		},
		{
			// Tool IDs are a composite of a prefix, their server name, and their tool name.
			name: "tools with same name different IDs",
			tools: map[string]*mcp.Tool{
				"id1": {ID: "id1", Name: "duplicate"},
				"id2": {ID: "id2", Name: "duplicate"},
				"id3": {ID: "id3", Name: "unique"},
			},
			allowlist: "duplicate",
			denylist:  "",
			expected:  []string{"duplicate", "duplicate"},
		},
		{
			name:      "greedy vs non-greedy matching",
			tools:     createTools("start_middle_end", "start_end", "middle"),
			allowlist: "start.*end",
			denylist:  "",
			expected:  []string{"start_middle_end", "start_end"},
		},
		{
			name:      "anchored patterns",
			tools:     createTools("prefix_tool", "tool_suffix", "prefix_tool_suffix"),
			allowlist: "^prefix_",
			denylist:  "_suffix$",
			expected:  []string{"prefix_tool"},
		},
		{
			name:      "invalid regex chars in tool names treated literally",
			tools:     createTools("tool[1]", "tool(2)", "tool{3}", "tool*4"),
			allowlist: `tool\[1\]`,
			denylist:  "",
			expected:  []string{"tool[1]"},
		},
		{
			name:      "effective filtering - use denylist to exclude non-matching",
			tools:     createTools("api_read", "api_write", "db_read", "db_write", "file_read"),
			allowlist: "",
			denylist:  "^(db_|file_)",
			expected:  []string{"api_read", "api_write"},
		},
		{
			name:      "allowlist with explicit denylist for complement",
			tools:     createTools("tool1", "tool2", "tool3", "tool4"),
			allowlist: "tool[12]",
			denylist:  "tool[34]",
			expected:  []string{"tool1", "tool2"},
		},
		{
			name:      "allowlist only filters correctly",
			tools:     createTools("allowed", "notallowed"),
			allowlist: "^allowed$",
			denylist:  "",
			expected:  []string{"allowed"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var resultNames []string
			result := mcp.FilterAllowedTools(slog.Make(), tt.tools, mustCompile(tt.allowlist), mustCompile(tt.denylist))
			for _, tool := range result {
				resultNames = append(resultNames, tool.Name)
			}

			require.ElementsMatch(t, tt.expected, resultNames)
		})
	}
}
