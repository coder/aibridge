package testutil

import (
	"fmt"
	"testing"
)

const (
	FixtureRequest              = "request"
	FixtureStreamingResponse    = "streaming"
	FixtureNonStreamingResponse = "non-streaming"

	FixtureStreamingToolResponse    = "streaming/tool-call"
	FixtureNonStreamingToolResponse = "non-streaming/tool-call"

	FixtureResponse = "response"
)

// LLMFixture is a typed view over a TXTAR fixture used for bridged LLM
// interactions (OpenAI/Anthropic-like request + streaming/non-streaming
// responses).
//
// It knows how to:
//   - derive a request body for streaming/non-streaming modes
//   - select the correct upstream response for the Nth call
//
// It does NOT attempt to interpret or validate the JSON/SSE payload contents.
type LLMFixture struct {
	TXTAR TXTARFixture
}

func NewLLMFixture(txtarFixture TXTARFixture) (LLMFixture, error) {
	if len(txtarFixture.Files) == 0 {
		return LLMFixture{}, fmt.Errorf("empty txtar fixture")
	}

	// We require at minimum a request.
	if !txtarFixture.Has(FixtureRequest) {
		return LLMFixture{}, fmt.Errorf("missing %q section", FixtureRequest)
	}

	return LLMFixture{TXTAR: txtarFixture}, nil
}

func MustLLMFixture(t testing.TB, txtarFixture TXTARFixture) LLMFixture {
	t.Helper()
	f, err := NewLLMFixture(txtarFixture)
	if err != nil {
		t.Fatalf("create LLM fixture: %v", err)
	}
	return f
}

// RequestBody returns the request body with the stream flag set according to
// streaming.
func (f LLMFixture) RequestBody(streaming bool) ([]byte, error) {
	body, ok := f.TXTAR.File(FixtureRequest)
	if !ok {
		// NewLLMFixture requires this, but keep it defensive.
		return nil, fmt.Errorf("missing %q section", FixtureRequest)
	}
	return SetJSON(body, "stream", streaming)
}

// MustRequestBody is a convenience helper for tests.
func (f LLMFixture) MustRequestBody(t testing.TB, streaming bool) []byte {
	t.Helper()
	body, err := f.RequestBody(streaming)
	if err != nil {
		t.Fatalf("fixture request body: %v", err)
	}
	return body
}

// HasToolCallResponses reports whether the fixture includes the */tool-call
// response sections.
func (f LLMFixture) HasToolCallResponses() bool {
	return f.TXTAR.Has(FixtureStreamingToolResponse) || f.TXTAR.Has(FixtureNonStreamingToolResponse)
}

// Response returns the upstream response body for the given call number.
//
// call is 1-indexed.
func (f LLMFixture) Response(call int, streaming bool) ([]byte, error) {
	if call < 1 {
		return nil, fmt.Errorf("call must be >= 1, got %d", call)
	}

	if call == 1 || !f.HasToolCallResponses() {
		if streaming {
			if !f.TXTAR.Has(FixtureStreamingResponse) {
				return nil, fmt.Errorf("missing %q section", FixtureStreamingResponse)
			}
			return f.TXTAR.Files[FixtureStreamingResponse], nil
		}

		if !f.TXTAR.Has(FixtureNonStreamingResponse) {
			return nil, fmt.Errorf("missing %q section", FixtureNonStreamingResponse)
		}
		return f.TXTAR.Files[FixtureNonStreamingResponse], nil
	}

	if call != 2 {
		return nil, fmt.Errorf("unexpected call %d; this fixture only supports 1 or 2 calls", call)
	}

	if streaming {
		if !f.TXTAR.Has(FixtureStreamingToolResponse) {
			return nil, fmt.Errorf("missing %q section", FixtureStreamingToolResponse)
		}
		return f.TXTAR.Files[FixtureStreamingToolResponse], nil
	}

	if !f.TXTAR.Has(FixtureNonStreamingToolResponse) {
		return nil, fmt.Errorf("missing %q section", FixtureNonStreamingToolResponse)
	}
	return f.TXTAR.Files[FixtureNonStreamingToolResponse], nil
}
