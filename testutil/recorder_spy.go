package testutil

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/coder/aibridge"
)

var _ aibridge.Recorder = (*RecorderSpy)(nil)

// RecorderSpy is a thread-safe in-memory implementation of [aibridge.Recorder]
// intended for tests.
//
// It provides query helpers so tests can assert on recorded interceptions,
// token/prompt usage, and tool usage.
type RecorderSpy struct {
	mu sync.Mutex

	interceptions    []*aibridge.InterceptionRecord
	tokenUsages      []*aibridge.TokenUsageRecord
	userPrompts      []*aibridge.PromptUsageRecord
	toolUsages       []*aibridge.ToolUsageRecord
	interceptionsEnd map[string]time.Time
}

func (m *RecorderSpy) RecordInterception(ctx context.Context, req *aibridge.InterceptionRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.interceptions = append(m.interceptions, req)
	return nil
}

func (m *RecorderSpy) RecordInterceptionEnded(ctx context.Context, req *aibridge.InterceptionRecordEnded) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.interceptionsEnd == nil {
		m.interceptionsEnd = make(map[string]time.Time)
	}
	if !slices.ContainsFunc(m.interceptions, func(intc *aibridge.InterceptionRecord) bool { return intc.ID == req.ID }) {
		return fmt.Errorf("interception id not found: %q", req.ID)
	}
	m.interceptionsEnd[req.ID] = req.EndedAt
	return nil
}

func (m *RecorderSpy) RecordPromptUsage(ctx context.Context, req *aibridge.PromptUsageRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.userPrompts = append(m.userPrompts, req)
	return nil
}

func (m *RecorderSpy) RecordTokenUsage(ctx context.Context, req *aibridge.TokenUsageRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tokenUsages = append(m.tokenUsages, req)
	return nil
}

func (m *RecorderSpy) RecordToolUsage(ctx context.Context, req *aibridge.ToolUsageRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.toolUsages = append(m.toolUsages, req)
	return nil
}

// RecordedTokenUsages returns a shallow clone of recorded token usages.
func (m *RecorderSpy) RecordedTokenUsages() []*aibridge.TokenUsageRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.tokenUsages)
}

// RecordedPromptUsages returns a shallow clone of recorded prompt usages.
func (m *RecorderSpy) RecordedPromptUsages() []*aibridge.PromptUsageRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.userPrompts)
}

// RecordedToolUsages returns a shallow clone of recorded tool usages.
func (m *RecorderSpy) RecordedToolUsages() []*aibridge.ToolUsageRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.toolUsages)
}

// RecordedInterceptions returns a shallow clone of recorded interceptions.
func (m *RecorderSpy) RecordedInterceptions() []*aibridge.InterceptionRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.interceptions)
}

// RequireAllInterceptionsEnded fails the test if any recorded interception did
// not receive a corresponding RecordInterceptionEnded call.
func (m *RecorderSpy) RequireAllInterceptionsEnded(t testing.TB) {
	t.Helper()

	m.mu.Lock()
	defer m.mu.Unlock()

	gotEnded := 0
	if m.interceptionsEnd != nil {
		gotEnded = len(m.interceptionsEnd)
	}

	if len(m.interceptions) != gotEnded {
		t.Fatalf("got %d interception ended calls, want %d", gotEnded, len(m.interceptions))
	}
	for _, intc := range m.interceptions {
		if m.interceptionsEnd == nil {
			t.Fatalf("interception with id %q has not been ended", intc.ID)
		}
		if _, ok := m.interceptionsEnd[intc.ID]; !ok {
			t.Fatalf("interception with id %q has not been ended", intc.ID)
		}
	}
}

// TotalInputTokens sums input tokens from token usage records.
func TotalInputTokens(in []*aibridge.TokenUsageRecord) int64 {
	var total int64
	for _, el := range in {
		total += el.Input
	}
	return total
}

// TotalOutputTokens sums output tokens from token usage records.
func TotalOutputTokens(in []*aibridge.TokenUsageRecord) int64 {
	var total int64
	for _, el := range in {
		total += el.Output
	}
	return total
}
