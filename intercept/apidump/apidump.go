package apidump

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"cdr.dev/slog/v3"

	"github.com/coder/quartz"
	"github.com/google/uuid"
	"github.com/tidwall/pretty"
)

const (
	// SuffixRequest is the file suffix for request dump files.
	SuffixRequest = ".req.txt"
	// SuffixResponse is the file suffix for response dump files.
	SuffixResponse = ".resp.txt"
)

// MiddlewareNext is the function to call the next middleware or the actual request.
type MiddlewareNext = func(*http.Request) (*http.Response, error)

// Middleware is an HTTP middleware function compatible with SDK WithMiddleware options.
type Middleware = func(*http.Request, MiddlewareNext) (*http.Response, error)

// NewMiddleware returns a middleware function that dumps requests and responses to files.
// Files are written to the path returned by DumpPath.
// If baseDir is empty, returns nil (no middleware).
func NewMiddleware(baseDir, provider, model string, interceptionID uuid.UUID, logger slog.Logger, clk quartz.Clock) Middleware {
	if baseDir == "" {
		return nil
	}

	d := &dumper{
		baseDir:        baseDir,
		provider:       provider,
		model:          model,
		interceptionID: interceptionID,
		clk:            clk,
	}

	return func(req *http.Request, next MiddlewareNext) (*http.Response, error) {
		if err := d.dumpRequest(req); err != nil {
			logger.Named("apidump").Warn(context.Background(), "failed to dump request", slog.Error(err))
		}

		resp, err := next(req)
		if err != nil {
			return resp, err
		}

		if err := d.dumpResponse(resp); err != nil {
			logger.Named("apidump").Warn(context.Background(), "failed to dump response", slog.Error(err))
		}

		return resp, nil
	}
}

type dumper struct {
	baseDir        string
	provider       string
	model          string
	interceptionID uuid.UUID
	clk            quartz.Clock
}

func (d *dumper) dumpRequest(req *http.Request) error {
	dumpPath := d.path(SuffixRequest)
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

	prettyBody := prettyPrintJSON(bodyBytes)

	// Build raw HTTP request format
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "%s %s %s\r\n", req.Method, req.URL.RequestURI(), req.Proto)
	fmt.Fprintf(&buf, "Host: %s\r\n", req.Host)
	fmt.Fprintf(&buf, "Content-Length: %d\r\n", len(prettyBody))
	for key, values := range req.Header {
		_, sensitive := sensitiveRequestHeaders[key]
		for _, value := range values {
			if sensitive {
				value = redactHeaderValue(value)
			}
			fmt.Fprintf(&buf, "%s: %s\r\n", key, value)
		}
	}
	fmt.Fprintf(&buf, "\r\n")
	buf.Write(prettyBody)

	return os.WriteFile(dumpPath, buf.Bytes(), 0o644)
}

func (d *dumper) dumpResponse(resp *http.Response) error {
	dumpPath := d.path(SuffixResponse)

	// Build raw HTTP response headers
	var headerBuf bytes.Buffer
	fmt.Fprintf(&headerBuf, "%s %s\r\n", resp.Proto, resp.Status)
	for key, values := range resp.Header {
		_, sensitive := sensitiveResponseHeaders[key]
		for _, value := range values {
			if sensitive {
				value = redactHeaderValue(value)
			}
			fmt.Fprintf(&headerBuf, "%s: %s\r\n", key, value)
		}
	}
	fmt.Fprintf(&headerBuf, "\r\n")

	// Wrap the response body to capture it as it streams
	if resp.Body != nil {
		resp.Body = &streamingBodyDumper{
			body:       resp.Body,
			dumpPath:   dumpPath,
			headerData: headerBuf.Bytes(),
		}
	} else {
		// No body, just write headers
		return os.WriteFile(dumpPath, headerBuf.Bytes(), 0o644)
	}

	return nil
}

// path returns the path to a request/response dump file for a given interception.
// suffix should be SuffixRequest or SuffixResponse.
func (d *dumper) path(suffix string) string {
	safeModel := strings.ReplaceAll(d.model, "/", "-")
	return filepath.Join(d.baseDir, d.provider, safeModel, fmt.Sprintf("%d-%s%s", d.clk.Now().UTC().UnixMilli(), d.interceptionID, suffix))
}

// prettyPrintJSON returns indented JSON if body is valid JSON, otherwise returns body as-is.
// Unlike json.MarshalIndent, this preserves the original key order from the input,
// which makes the dumps easier to read and compare with the original requests.
func prettyPrintJSON(body []byte) []byte {
	if len(body) == 0 {
		return body
	}
	result := pretty.Pretty(body)
	// pretty.Pretty returns a truncated/modified result for invalid JSON,
	// so check if the result is valid JSON; if not, return the original.
	if !json.Valid(result) {
		return body
	}
	// Trim trailing newline added by pretty.Pretty.
	return bytes.TrimSuffix(result, []byte("\n"))
}
