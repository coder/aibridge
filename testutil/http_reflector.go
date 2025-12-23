package testutil

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
)

// HTTPReflectorServer is an httptest.Server that responds with a raw HTTP
// response loaded from a fixture.
//
// The fixture bytes must be a complete HTTP response (status line, headers,
// blank line, body) as accepted by http.ReadResponse.
//
// This is useful for simulating upstream error responses.
type HTTPReflectorServer struct {
	*httptest.Server
}

func NewHTTPReflectorServer(t testing.TB, ctx context.Context, rawHTTPResponse []byte) *HTTPReflectorServer {
	t.Helper()
	if ctx == nil {
		ctx = context.Background()
	}

	s := &HTTPReflectorServer{}

	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mock, err := http.ReadResponse(bufio.NewReader(bytes.NewReader(rawHTTPResponse)), r)
		if err != nil {
			t.Fatalf("read mock response: %v", err)
		}
		defer mock.Body.Close()

		for key, values := range mock.Header {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}

		w.WriteHeader(mock.StatusCode)
		_, err = io.Copy(w, mock.Body)
		if err != nil {
			t.Fatalf("copy mock response body: %v", err)
		}
	}))
	srv.Config.BaseContext = func(_ net.Listener) context.Context {
		return ctx
	}
	srv.Start()
	t.Cleanup(srv.Close)

	s.Server = srv
	return s
}
