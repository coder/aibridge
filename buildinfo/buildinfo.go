package buildinfo

import (
	"runtime/debug"
)

var version string

func init() {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return
	}

	for _, dep := range info.Deps {
		if dep.Path == "github.com/coder/aibridge" {
			version = dep.Version
		}
	}
}

func Version() string {
	if version == "" {
		return "unknown"
	}
	return version
}
