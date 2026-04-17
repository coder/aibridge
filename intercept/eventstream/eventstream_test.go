package eventstream_test

import (
	"bufio"
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"cdr.dev/slog/v3"
	"cdr.dev/slog/v3/sloggers/slogtest"
	"github.com/stretchr/testify/require"

	"github.com/coder/aibridge/intercept/eventstream"
	"github.com/coder/quartz"
)

// captureSink collects log entries for assertions in tests.
type captureSink struct {
	mu      sync.Mutex
	entries []slog.SinkEntry
}

func (s *captureSink) LogEntry(_ context.Context, e slog.SinkEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = append(s.entries, e)
}

func (*captureSink) Sync() {}

func (s *captureSink) warns() []slog.SinkEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []slog.SinkEntry
	for _, e := range s.entries {
		if e.Level == slog.LevelWarn {
			out = append(out, e)
		}
	}
	return out
}

// clockAdvancingFlusher wraps httptest.ResponseRecorder and advances the mock
// clock on each Flush call, simulating a slow client without real sleeping.
type clockAdvancingFlusher struct {
	*httptest.ResponseRecorder
	clk     *quartz.Mock
	advance time.Duration
}

func (f *clockAdvancingFlusher) Flush() {
	f.clk.Advance(f.advance)
	f.ResponseRecorder.Flush()
}

// Hijack satisfies the FullResponseWriter lint rule.
func (*clockAdvancingFlusher) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return nil, nil, nil
}

func TestEventStream_LogsWarning_WhenFlushIsSlow(t *testing.T) {
	t.Parallel()

	sink := &captureSink{}
	logger := slogtest.Make(t, nil).AppendSinks(sink).Leveled(slog.LevelWarn)
	ctx := context.Background()
	clk := quartz.NewMock(t)

	stream := eventstream.NewEventStream(ctx, logger, nil, clk)

	w := &clockAdvancingFlusher{
		ResponseRecorder: httptest.NewRecorder(),
		clk:              clk,
		advance:          600 * time.Millisecond, // exceeds slowFlushThreshold (500ms)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "/", nil)
	require.NoError(t, err)

	done := make(chan struct{})
	go func() {
		defer close(done)
		stream.Start(w, req)
	}()

	stream.InitiateStream(w)
	require.NoError(t, stream.SendRaw(ctx, []byte("data: hello\n\n")))
	require.NoError(t, stream.Shutdown(ctx))
	<-done

	warns := sink.warns()
	require.Len(t, warns, 1)
	require.Equal(t, "slow client detected", warns[0].Message)
}

func TestEventStream_NoWarning_WhenFlushIsFast(t *testing.T) {
	t.Parallel()

	sink := &captureSink{}
	logger := slogtest.Make(t, nil).AppendSinks(sink).Leveled(slog.LevelWarn)
	ctx := context.Background()
	clk := quartz.NewMock(t)

	stream := eventstream.NewEventStream(ctx, logger, nil, clk)

	// No clock advance — flush duration stays at 0, below threshold.
	w := &clockAdvancingFlusher{
		ResponseRecorder: httptest.NewRecorder(),
		clk:              clk,
		advance:          0,
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "/", nil)
	require.NoError(t, err)

	done := make(chan struct{})
	go func() {
		defer close(done)
		stream.Start(w, req)
	}()

	stream.InitiateStream(w)
	require.NoError(t, stream.SendRaw(ctx, []byte("data: hello\n\n")))
	require.NoError(t, stream.Shutdown(ctx))
	<-done

	require.Empty(t, sink.warns())
}
