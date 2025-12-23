package testutil

import (
	"fmt"
	"testing"

	"golang.org/x/tools/txtar"
)

// TXTARFixture is a parsed txtar archive (see golang.org/x/tools/txtar) with a
// convenient map-based API.
//
// Tests should prefer TXTARFixture over ad-hoc txtar.Parse + manual file maps.
// It provides early validation and clearer error messages.
type TXTARFixture struct {
	Comment string
	Files   map[string][]byte
}

func ParseTXTAR(data []byte) (TXTARFixture, error) {
	if len(data) == 0 {
		return TXTARFixture{}, fmt.Errorf("empty txtar input")
	}

	arc := txtar.Parse(data)

	files := make(map[string][]byte, len(arc.Files))
	for _, f := range arc.Files {
		if f.Name == "" {
			return TXTARFixture{}, fmt.Errorf("txtar contains a file with an empty name")
		}
		if _, exists := files[f.Name]; exists {
			return TXTARFixture{}, fmt.Errorf("txtar contains duplicate file name %q", f.Name)
		}
		files[f.Name] = f.Data
	}

	return TXTARFixture{
		Comment: string(arc.Comment),
		Files:   files,
	}, nil
}

func MustParseTXTAR(t testing.TB, data []byte) TXTARFixture {
	t.Helper()
	f, err := ParseTXTAR(data)
	if err != nil {
		t.Fatalf("parse txtar: %v", err)
	}
	return f
}

func (f TXTARFixture) Has(name string) bool {
	_, ok := f.Files[name]
	return ok
}

func (f TXTARFixture) File(name string) ([]byte, bool) {
	b, ok := f.Files[name]
	return b, ok
}

func (f TXTARFixture) MustFile(t testing.TB, name string) []byte {
	t.Helper()
	b, ok := f.File(name)
	if !ok {
		t.Fatalf("txtar missing section %q; have %v", name, sortedKeys(f.Files))
	}
	return b
}

func (f TXTARFixture) RequireFiles(t testing.TB, names ...string) {
	t.Helper()
	for _, name := range names {
		if !f.Has(name) {
			t.Fatalf("txtar missing required section %q; have %v", name, sortedKeys(f.Files))
		}
	}
}
