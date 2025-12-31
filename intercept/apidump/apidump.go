package apidump

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/uuid"
)

// MiddlewareNext is the function to call the next middleware or the actual request.
type MiddlewareNext = func(*http.Request) (*http.Response, error)

// Middleware is an HTTP middleware function compatible with SDK WithMiddleware options.
type Middleware = func(*http.Request, MiddlewareNext) (*http.Response, error)

// DumpPath returns the path to a request/response dump file for a given interception.
// kind should be "req" or "resp".
func DumpPath(baseDir, provider, model string, interceptionID uuid.UUID, kind string) string {
	safeModel := strings.ReplaceAll(model, "/", "-")
	return filepath.Join(baseDir, provider, safeModel, fmt.Sprintf("%s.%s.txt", interceptionID, kind))
}

// NewMiddleware returns a middleware function that dumps requests and responses to files.
// Files are written to the path returned by DumpPath.
// If baseDir is empty, returns nil (no middleware).
func NewMiddleware(baseDir, provider, model string, interceptionID uuid.UUID) Middleware {
	if baseDir == "" {
		return nil
	}

	d := &dumper{
		baseDir:        baseDir,
		provider:       provider,
		model:          model,
		interceptionID: interceptionID,
	}

	return func(req *http.Request, next MiddlewareNext) (*http.Response, error) {
		if err := d.dumpRequest(req); err != nil {
			fmt.Fprintf(os.Stderr, "apidump: failed to dump request: %v\n", err)
		}

		resp, err := next(req)
		if err != nil {
			return resp, err
		}

		if err := d.dumpResponse(resp); err != nil {
			fmt.Fprintf(os.Stderr, "apidump: failed to dump response: %v\n", err)
		}

		return resp, nil
	}
}

type dumper struct {
	baseDir        string
	provider       string
	model          string
	interceptionID uuid.UUID
}

func (d *dumper) dumpRequest(req *http.Request) error {
	dumpPath := DumpPath(d.baseDir, d.provider, d.model, d.interceptionID, "req")
	if err := os.MkdirAll(filepath.Dir(dumpPath), 0o755); err != nil {
		return fmt.Errorf("create dump dir: %w", err)
	}

	// Read and restore body
	var bodyBytes []byte
	if req.Body != nil {
		var err error
		bodyBytes, err = io.ReadAll(req.Body)
		if err != nil {
			return fmt.Errorf("read request body: %w", err)
		}
		req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
	}

	// Build raw HTTP request format
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "%s %s %s\r\n", req.Method, req.URL.RequestURI(), req.Proto)
	fmt.Fprintf(&buf, "Host: %s\r\n", req.Host)
	for key, values := range req.Header {
		for _, value := range values {
			fmt.Fprintf(&buf, "%s: %s\r\n", key, value)
		}
	}
	fmt.Fprintf(&buf, "\r\n")
	buf.Write(prettyPrintJSON(bodyBytes))

	return os.WriteFile(dumpPath, buf.Bytes(), 0o644)
}

func (d *dumper) dumpResponse(resp *http.Response) error {
	dumpPath := DumpPath(d.baseDir, d.provider, d.model, d.interceptionID, "resp")

	// Read and restore body
	var bodyBytes []byte
	if resp.Body != nil {
		var err error
		bodyBytes, err = io.ReadAll(resp.Body)
		if err != nil {
			return fmt.Errorf("read response body: %w", err)
		}
		resp.Body = io.NopCloser(bytes.NewReader(bodyBytes))
	}

	// Build raw HTTP response format
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "%s %s\r\n", resp.Proto, resp.Status)
	for key, values := range resp.Header {
		for _, value := range values {
			fmt.Fprintf(&buf, "%s: %s\r\n", key, value)
		}
	}
	fmt.Fprintf(&buf, "\r\n")
	buf.Write(prettyPrintJSON(bodyBytes))

	return os.WriteFile(dumpPath, buf.Bytes(), 0o644)
}

// prettyPrintJSON returns indented JSON if body is valid JSON, otherwise returns body as-is.
func prettyPrintJSON(body []byte) []byte {
	if len(body) == 0 {
		return body
	}
	var parsed any
	if err := json.Unmarshal(body, &parsed); err != nil {
		return body
	}
	pretty, err := json.MarshalIndent(parsed, "", "  ")
	if err != nil {
		return body
	}
	return pretty
}
