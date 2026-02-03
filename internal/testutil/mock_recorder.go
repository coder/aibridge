package testutil

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/coder/aibridge/recorder"
	"github.com/stretchr/testify/require"
)

// MockRecorder is a test implementation of aibridge.Recorder that
// captures all recording calls for test assertions.
type MockRecorder struct {
	mu sync.Mutex

	interceptions    []*recorder.InterceptionRecord
	tokenUsages      []*recorder.TokenUsageRecord
	userPrompts      []*recorder.PromptUsageRecord
	toolUsages       []*recorder.ToolUsageRecord
	interceptionsEnd map[string]time.Time
}

func (m *MockRecorder) RecordInterception(ctx context.Context, req *recorder.InterceptionRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.interceptions = append(m.interceptions, req)
	return nil
}

func (m *MockRecorder) RecordInterceptionEnded(ctx context.Context, req *recorder.InterceptionRecordEnded) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.interceptionsEnd == nil {
		m.interceptionsEnd = make(map[string]time.Time)
	}
	if !slices.ContainsFunc(m.interceptions, func(intc *recorder.InterceptionRecord) bool { return intc.ID == req.ID }) {
		return fmt.Errorf("id not found")
	}
	m.interceptionsEnd[req.ID] = req.EndedAt
	return nil
}

func (m *MockRecorder) RecordPromptUsage(ctx context.Context, req *recorder.PromptUsageRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.userPrompts = append(m.userPrompts, req)
	return nil
}

func (m *MockRecorder) RecordTokenUsage(ctx context.Context, req *recorder.TokenUsageRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tokenUsages = append(m.tokenUsages, req)
	return nil
}

func (m *MockRecorder) RecordToolUsage(ctx context.Context, req *recorder.ToolUsageRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.toolUsages = append(m.toolUsages, req)
	return nil
}

// RecordedTokenUsages returns a copy of recorded token usages in a thread-safe manner.
// Note: This is a shallow clone - the slice is copied but the pointers reference the
// same underlying records. This is sufficient for our test assertions which only read
// the data and don't modify the records.
func (m *MockRecorder) RecordedTokenUsages() []*recorder.TokenUsageRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.tokenUsages)
}

// RecordedPromptUsages returns a copy of recorded prompt usages in a thread-safe manner.
// Note: This is a shallow clone (see RecordedTokenUsages for details).
func (m *MockRecorder) RecordedPromptUsages() []*recorder.PromptUsageRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.userPrompts)
}

// RecordedToolUsages returns a copy of recorded tool usages in a thread-safe manner.
// Note: This is a shallow clone (see RecordedTokenUsages for details).
func (m *MockRecorder) RecordedToolUsages() []*recorder.ToolUsageRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.toolUsages)
}

// RecordedInterceptions returns a copy of recorded interceptions in a thread-safe manner.
// Note: This is a shallow clone (see RecordedTokenUsages for details).
func (m *MockRecorder) RecordedInterceptions() []*recorder.InterceptionRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.interceptions)
}

// ToolUsages returns the raw toolUsages slice for direct field access in tests.
// Use RecordedToolUsages() for thread-safe access when assertions don't need direct field access.
func (m *MockRecorder) ToolUsages() []*recorder.ToolUsageRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.toolUsages
}

// VerifyAllInterceptionsEnded verifies all recorded interceptions have been marked as completed.
func (m *MockRecorder) VerifyAllInterceptionsEnded(t *testing.T) {
	t.Helper()

	m.mu.Lock()
	defer m.mu.Unlock()
	require.Equalf(t, len(m.interceptions), len(m.interceptionsEnd), "got %v interception ended calls, want: %v", len(m.interceptionsEnd), len(m.interceptions))
	for _, intc := range m.interceptions {
		require.Containsf(t, m.interceptionsEnd, intc.ID, "interception with id: %v has not been ended", intc.ID)
	}
}
