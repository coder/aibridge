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
	"slices"
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
		logger:         logger,
	}

	return func(req *http.Request, next MiddlewareNext) (*http.Response, error) {
		if err := d.dumpRequest(req); err != nil {
			logger.Named("apidump").Warn(context.Background(), "failed to dump request", slog.Error(err))
		}

		// TODO: https://github.com/coder/aibridge/issues/129
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
	logger         slog.Logger
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
	d.writeRedactedHeaders(&buf, req.Header, sensitiveRequestHeaders, map[string]string{
		"Content-Length": fmt.Sprintf("%d", len(prettyBody)),
	})

	fmt.Fprintf(&buf, "\r\n")
	buf.Write(prettyBody)

	return os.WriteFile(dumpPath, buf.Bytes(), 0o644)
}

func (d *dumper) dumpResponse(resp *http.Response) error {
	dumpPath := d.path(SuffixResponse)

	// Build raw HTTP response headers
	var headerBuf bytes.Buffer
	fmt.Fprintf(&headerBuf, "%s %s\r\n", resp.Proto, resp.Status)
	d.writeRedactedHeaders(&headerBuf, resp.Header, sensitiveResponseHeaders, nil)
	fmt.Fprintf(&headerBuf, "\r\n")

	// Wrap the response body to capture it as it streams
	if resp.Body != nil {
		resp.Body = &streamingBodyDumper{
			body:       resp.Body,
			dumpPath:   dumpPath,
			headerData: headerBuf.Bytes(),
			logger: func(err error) {
				d.logger.Named("apidump").Warn(context.Background(), "failed to initialize response dump", slog.Error(err))
			},
		}
	} else {
		// No body, just write headers
		return os.WriteFile(dumpPath, headerBuf.Bytes(), 0o644)
	}

	return nil
}

// writeRedactedHeaders writes HTTP headers in wire format (Key: Value\r\n) to w,
// redacting sensitive values and applying any overrides. Headers are sorted by key
// for deterministic output.
// `sensitive` and `overrides` must both supply keys in canoncialized form.
// See [textproto.MIMEHeader].
func (d *dumper) writeRedactedHeaders(w io.Writer, headers http.Header, sensitive map[string]struct{}, overrides map[string]string) {
	// Collect all header keys including overrides.
	headerKeys := make([]string, 0, len(headers)+len(overrides))
	seen := make(map[string]struct{}, len(headers)+len(overrides))
	for key := range headers {
		headerKeys = append(headerKeys, key)
		seen[key] = struct{}{}
	}
	// Add override keys that don't exist in headers.
	for key := range overrides {
		if _, ok := seen[key]; !ok {
			headerKeys = append(headerKeys, key)
		}
	}
	slices.Sort(headerKeys)

	for _, key := range headerKeys {
		_, isSensitive := sensitive[key]
		values := headers[key]
		// If no values exist but we have an override, use that.
		if len(values) == 0 {
			if override, ok := overrides[key]; ok {
				fmt.Fprintf(w, "%s: %s\r\n", key, override)
			}
			continue
		}
		for _, value := range values {
			if override, ok := overrides[key]; ok {
				value = override
			}

			if isSensitive {
				value = redactHeaderValue(value)
			}
			fmt.Fprintf(w, "%s: %s\r\n", key, value)
		}
	}
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
