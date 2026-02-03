package apidump

import (
	"bytes"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cdr.dev/slog/v3"
	"cdr.dev/slog/v3/sloggers/slogtest"
	"github.com/coder/quartz"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

func TestMiddleware_StreamingResponse(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
	clk := quartz.NewMock(t)
	interceptionID := uuid.New()

	middleware := NewMiddleware(tmpDir, "openai", "gpt-4", interceptionID, logger, clk)
	require.NotNil(t, middleware)

	req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader([]byte(`{}`)))
	require.NoError(t, err)

	// Simulate a streaming response with multiple chunks
	chunks := []string{
		"data: {\"chunk\": 1}\n\n",
		"data: {\"chunk\": 2}\n\n",
		"data: {\"chunk\": 3}\n\n",
		"data: [DONE]\n\n",
	}

	// Create a pipe to simulate streaming
	pr, pw := io.Pipe()
	go func() {
		for _, chunk := range chunks {
			pw.Write([]byte(chunk))
		}
		pw.Close()
	}()

	resp, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Proto:      "HTTP/1.1",
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       pr,
		}, nil
	})
	require.NoError(t, err)

	// Read response in small chunks to simulate streaming consumption
	var receivedData bytes.Buffer
	buf := make([]byte, 16)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			receivedData.Write(buf[:n])
		}
		if err == io.EOF {
			break
		}
		require.NoError(t, err)
	}
	require.NoError(t, resp.Body.Close())

	// Verify we received all the data
	expectedData := strings.Join(chunks, "")
	require.Equal(t, expectedData, receivedData.String())

	// Verify the dump file was created and contains all the streamed data
	modelDir := filepath.Join(tmpDir, "openai", "gpt-4")
	respDumpPath := findDumpFile(t, modelDir, SuffixResponse)
	respContent, err := os.ReadFile(respDumpPath)
	require.NoError(t, err)

	content := string(respContent)
	require.Contains(t, content, "HTTP/1.1 200 OK")
	require.Contains(t, content, "Content-Type: text/event-stream")
	// All chunks should be in the dump
	for _, chunk := range chunks {
		require.Contains(t, content, chunk)
	}
}

func TestMiddleware_PreservesResponseBody(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
	clk := quartz.NewMock(t)
	interceptionID := uuid.New()

	middleware := NewMiddleware(tmpDir, "openai", "gpt-4", interceptionID, logger, clk)
	require.NotNil(t, middleware)

	req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader([]byte(`{}`)))
	require.NoError(t, err)

	originalRespBody := `{"choices": [{"message": {"content": "hi"}}]}`
	resp, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Proto:      "HTTP/1.1",
			Header:     http.Header{},
			Body:       io.NopCloser(bytes.NewReader([]byte(originalRespBody))),
		}, nil
	})
	require.NoError(t, err)

	// Verify the response body is still readable after middleware
	capturedBody, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, originalRespBody, string(capturedBody))
}
