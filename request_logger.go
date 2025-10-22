package aibridge

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"os"
	"path/filepath"
	"strings"

	"cdr.dev/slog"
)

// sanitizeModelName makes a model name safe for use as a directory name.
// Replaces filesystem-unsafe characters with underscores.
func sanitizeModelName(model string) string {
	replacer := strings.NewReplacer(
		"/", "_",
		"\\", "_",
		":", "_",
		"*", "_",
		"?", "_",
		"\"", "_",
		"<", "_",
		">", "_",
		"|", "_",
	)
	return replacer.Replace(model)
}

// logUpstreamRequest logs an HTTP request with the given ID and model name.
// The prefix format is: [req] [id] [model]
func logUpstreamRequest(logger *log.Logger, id, model string, req *http.Request) {
	if logger == nil {
		return
	}

	if reqDump, err := httputil.DumpRequest(req, true); err == nil {
		logger.Printf("[req] [%s] [%s] %s", id, model, reqDump)
	}
}

// logUpstreamResponse logs an HTTP response with the given ID and model name.
// The prefix format is: [res] [id] [model]
func logUpstreamResponse(logger *log.Logger, id, model string, resp *http.Response) {
	if logger == nil {
		return
	}

	if respDump, err := httputil.DumpResponse(resp, true); err == nil {
		logger.Printf("[res] [%s] [%s] %s", id, model, respDump)
	}
}

// logUpstreamError logs an error that occurred during request/response processing.
// The prefix format is: [res] [id] [model] Error:
func logUpstreamError(logger *log.Logger, id, model string, err error) {
	if logger == nil {
		return
	}

	logger.Printf("[res] [%s] [%s] Error: %v", id, model, err)
}

// createLoggingMiddleware creates a middleware function that logs requests and responses.
// Logs are written to $TMPDIR/$provider/$model/$id.req.log and $TMPDIR/$provider/$model/$id.res.log
// Returns nil if logging setup fails, logging errors via the provided logger.
func createLoggingMiddleware(logger slog.Logger, provider, id, model string) func(*http.Request, func(*http.Request) (*http.Response, error)) (*http.Response, error) {
	ctx := context.Background()
	safeModel := sanitizeModelName(model)
	logDir := filepath.Join(os.TempDir(), provider, safeModel)

	// Create the directory structure if it doesn't exist
	if err := os.MkdirAll(logDir, 0755); err != nil {
		logger.Warn(ctx, "failed to create log directory", slog.Error(err), slog.F("dir", logDir))
		return nil
	}

	reqLogPath := filepath.Join(logDir, fmt.Sprintf("%s.req.log", id))
	resLogPath := filepath.Join(logDir, fmt.Sprintf("%s.res.log", id))

	reqLogFile, err := os.OpenFile(reqLogPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		logger.Warn(ctx, "failed to open request log file", slog.Error(err), slog.F("path", reqLogPath))
		return nil
	}

	resLogFile, err := os.OpenFile(resLogPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		reqLogFile.Close()
		logger.Warn(ctx, "failed to open response log file", slog.Error(err), slog.F("path", resLogPath))
		return nil
	}

	reqLogger := log.New(reqLogFile, "", log.LstdFlags)
	resLogger := log.New(resLogFile, "", log.LstdFlags)

	return func(req *http.Request, next func(*http.Request) (*http.Response, error)) (*http.Response, error) {
		logUpstreamRequest(reqLogger, id, model, req)

		resp, err := next(req)
		if err != nil {
			logUpstreamError(resLogger, id, model, err)
			return resp, err
		}

		logUpstreamResponse(resLogger, id, model, resp)

		return resp, err
	}
}
