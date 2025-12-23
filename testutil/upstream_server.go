package testutil

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// UpstreamRequest captures a single request received by an [UpstreamServer].
//
// It is intended for test assertions (e.g. validating headers, paths, and
// request bodies).
type UpstreamRequest struct {
	Call   int
	Method string
	Path   string
	Header http.Header
	Body   []byte
}

// UpstreamServer is an httptest.Server that mimics an upstream LLM provider.
//
// It is driven by an [LLMFixture]. It chooses between the streaming and
// non-streaming fixture responses based on the request body ("stream": true)
// and the request path (AWS Bedrock uses a streaming-specific path).
//
// For injected-tool tests, it can serve a second response ("*/tool-call") on
// the 2nd request.
type UpstreamServer struct {
	*httptest.Server

	fixture LLMFixture

	callCount atomic.Uint32

	requestsMu sync.Mutex
	requests   []UpstreamRequest

	// statusCode is only applied for non-streaming responses, matching the current
	// test behavior (streaming responses default to 200 once the stream begins).
	statusCode atomic.Int32

	responseMutator func(call int, resp []byte) []byte
}

type UpstreamOption func(*UpstreamServer)

func WithUpstreamNonStreamingStatusCode(code int) UpstreamOption {
	return func(s *UpstreamServer) {
		s.statusCode.Store(int32(code))
	}
}

func WithUpstreamResponseMutator(fn func(call int, resp []byte) []byte) UpstreamOption {
	return func(s *UpstreamServer) {
		s.responseMutator = fn
	}
}

func NewUpstreamServer(t testing.TB, ctx context.Context, fixture LLMFixture, opts ...UpstreamOption) *UpstreamServer {
	t.Helper()
	if ctx == nil {
		ctx = context.Background()
	}

	s := &UpstreamServer{fixture: fixture}
	for _, opt := range opts {
		if opt != nil {
			opt(s)
		}
	}

	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := int(s.callCount.Add(1))

		body, err := io.ReadAll(r.Body)
		_ = r.Body.Close()
		if err != nil {
			t.Fatalf("read upstream request body: %v", err)
		}

		s.recordRequest(call, r, body)

		type msg struct {
			Stream bool `json:"stream"`
		}
		var reqMsg msg
		if err := json.Unmarshal(body, &reqMsg); err != nil {
			t.Fatalf("unmarshal upstream request body: %v", err)
		}

		// AWS Bedrock uses a streaming-specific path suffix.
		isStreaming := reqMsg.Stream || strings.HasSuffix(r.URL.Path, "invoke-with-response-stream")

		if !isStreaming {
			respBody, err := s.fixture.Response(call, false)
			if err != nil {
				t.Fatalf("select upstream response: %v", err)
			}
			if s.responseMutator != nil {
				respBody = s.responseMutator(call, respBody)
			}

			code := int(s.statusCode.Load())
			if code == 0 {
				code = http.StatusOK
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(code)
			_, _ = w.Write(respBody)
			return
		}

		// Streaming response.
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("Access-Control-Allow-Origin", "*")

		respBody, err := s.fixture.Response(call, true)
		if err != nil {
			t.Fatalf("select upstream response: %v", err)
		}
		if s.responseMutator != nil {
			respBody = s.responseMutator(call, respBody)
		}

		scanner := bufio.NewScanner(bytes.NewReader(respBody))
		// Allow large SSE lines; fixtures may exceed the default 64KiB scanner limit.
		scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}

		for scanner.Scan() {
			line := scanner.Text()
			fmt.Fprintf(w, "%s\n", line)
			flusher.Flush()
		}

		if err := scanner.Err(); err != nil {
			http.Error(w, fmt.Sprintf("error reading fixture: %v", err), http.StatusInternalServerError)
			return
		}
	})

	srv := httptest.NewUnstartedServer(h)
	srv.Config.BaseContext = func(_ net.Listener) context.Context {
		return ctx
	}
	srv.Start()
	t.Cleanup(srv.Close)

	s.Server = srv
	return s
}

func (s *UpstreamServer) recordRequest(call int, r *http.Request, body []byte) {
	if s == nil {
		return
	}
	if r == nil {
		return
	}

	bodyCopy := make([]byte, len(body))
	copy(bodyCopy, body)

	s.requestsMu.Lock()
	s.requests = append(s.requests, UpstreamRequest{
		Call:   call,
		Method: r.Method,
		Path:   r.URL.Path,
		Header: r.Header.Clone(),
		Body:   bodyCopy,
	})
	s.requestsMu.Unlock()
}

// Requests returns a snapshot of requests received by this server.
func (s *UpstreamServer) Requests() []UpstreamRequest {
	if s == nil {
		return nil
	}

	s.requestsMu.Lock()
	defer s.requestsMu.Unlock()

	out := make([]UpstreamRequest, len(s.requests))
	for i, req := range s.requests {
		bodyCopy := make([]byte, len(req.Body))
		copy(bodyCopy, req.Body)

		out[i] = UpstreamRequest{
			Call:   req.Call,
			Method: req.Method,
			Path:   req.Path,
			Header: req.Header.Clone(),
			Body:   bodyCopy,
		}
	}
	return out
}

// LastRequest returns the most recently received request, if any.
func (s *UpstreamServer) LastRequest() (UpstreamRequest, bool) {
	if s == nil {
		return UpstreamRequest{}, false
	}

	s.requestsMu.Lock()
	defer s.requestsMu.Unlock()
	if len(s.requests) == 0 {
		return UpstreamRequest{}, false
	}
	req := s.requests[len(s.requests)-1]

	bodyCopy := make([]byte, len(req.Body))
	copy(bodyCopy, req.Body)

	return UpstreamRequest{
		Call:   req.Call,
		Method: req.Method,
		Path:   req.Path,
		Header: req.Header.Clone(),
		Body:   bodyCopy,
	}, true
}

func (s *UpstreamServer) CallCount() int {
	return int(s.callCount.Load())
}

func (s *UpstreamServer) RequireCallCountEventually(t testing.TB, want int) {
	t.Helper()

	deadline := time.Now().Add(10 * time.Second)
	for {
		if s.CallCount() == want {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("upstream call count: got %d, want %d", s.CallCount(), want)
		}
		time.Sleep(50 * time.Millisecond)
	}
}
