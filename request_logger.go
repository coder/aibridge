package aibridge

import (
	"log"
	"net/http"
	"net/http/httputil"
	"os"
)

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
// Returns nil if logging setup fails.
func createLoggingMiddleware(provider, id, model string) func(*http.Request, func(*http.Request) (*http.Response, error)) (*http.Response, error) {
	reqLogFile, err := os.OpenFile("/tmp/"+provider+"-req.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil
	}

	resLogFile, err := os.OpenFile("/tmp/"+provider+"-res.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		reqLogFile.Close()
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
