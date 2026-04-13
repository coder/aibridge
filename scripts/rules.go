// Package gorules defines custom ruleguard checks used by golangci-lint.
package gorules

import (
	"github.com/quasilyte/go-ruleguard/dsl"
	"github.com/quasilyte/go-ruleguard/dsl/types"
)

// Use xerrors throughout the project to keep stack information.
//
//nolint:unused,deadcode,varnamelen
func xerrors(m dsl.Matcher) {
	m.Import("errors")
	m.Import("fmt")
	m.Import("golang.org/x/xerrors")

	m.Match("fmt.Errorf($arg)").
		Suggest("xerrors.New($arg)").
		Report("Use xerrors to provide additional stacktrace information!")

	m.Match("fmt.Errorf($arg1, $*args)").
		Suggest("xerrors.Errorf($arg1, $args)").
		Report("Use xerrors to provide additional stacktrace information!")

	m.Match("errors.$_($msg)").
		Where(m["msg"].Type.Is("string")).
		Suggest("xerrors.New($msg)").
		Report("Use xerrors to provide additional stacktrace information!")
}

// useStandardTimeoutsAndDelaysInTests discourages magic timeout values in
// tests so timeout policy stays consistent across the suite.
//
//nolint:unused,deadcode,varnamelen
func useStandardTimeoutsAndDelaysInTests(m dsl.Matcher) {
	m.Import("context")
	m.Import("github.com/stretchr/testify/assert")
	m.Import("github.com/stretchr/testify/require")

	m.Match(`context.WithTimeout($ctx, $duration)`).
		Where(
			m.File().Imports("testing") &&
				!m.File().PkgPath.Matches("testutil$") &&
				!m["duration"].Text.Matches("^testutil\\."),
		).
		At(m["duration"]).
		Report("Do not use magic numbers in test timeouts and delays. Use shared test timeout constants instead.")

	m.Match(`
		$testify.$Eventually($t, func() bool {
			$*_
		}, $timeout, $interval, $*_)
	`).
		Where(
			(m["testify"].Text == "require" || m["testify"].Text == "assert") &&
				(m["Eventually"].Text == "Eventually" || m["Eventually"].Text == "Eventuallyf") &&
				!m["timeout"].Text.Matches("^testutil\\."),
		).
		At(m["timeout"]).
		Report("Do not use magic numbers in test timeouts and delays. Use shared test timeout constants instead.")

	m.Match(`
		$testify.$Eventually($t, func() bool {
			$*_
		}, $timeout, $interval, $*_)
	`).
		Where(
			(m["testify"].Text == "require" || m["testify"].Text == "assert") &&
				(m["Eventually"].Text == "Eventually" || m["Eventually"].Text == "Eventuallyf") &&
				!m["interval"].Text.Matches("^testutil\\."),
		).
		At(m["interval"]).
		Report("Do not use magic numbers in test timeouts and delays. Use shared test timeout constants instead.")
}

// FullResponseWriter ensures wrapper response writers still implement the
// interfaces most handlers expect.
func FullResponseWriter(m dsl.Matcher) {
	m.Match(`
	type $w struct {
		$*_
		http.ResponseWriter
		$*_
	}
	`).
		At(m["w"]).
		Where(m["w"].Filter(notImplementsFullResponseWriter)).
		Report("ResponseWriter \"$w\" must implement http.Flusher and http.Hijacker")
}

func notImplementsFullResponseWriter(ctx *dsl.VarFilterContext) bool {
	flusher := ctx.GetInterface(`net/http.Flusher`)
	hijacker := ctx.GetInterface(`net/http.Hijacker`)
	writer := ctx.GetInterface(`net/http.ResponseWriter`)
	p := types.NewPointer(ctx.Type)
	return !(types.Implements(p, writer) || types.Implements(ctx.Type, writer)) ||
		!(types.Implements(p, flusher) || types.Implements(ctx.Type, flusher)) ||
		!(types.Implements(p, hijacker) || types.Implements(ctx.Type, hijacker))
}

// slogFieldNameSnakeCase keeps structured log keys consistent.
func slogFieldNameSnakeCase(m dsl.Matcher) {
	m.Import("cdr.dev/slog/v3")
	m.Match(`slog.F($name, $value)`).
		Where(m["name"].Const && !m["name"].Text.Matches(`^"[a-z]+(_[a-z]+)*"$`)).
		Report("Field name $name must be snake_case.")
}

// slogMessageFormat enforces sentence style for log messages.
func slogMessageFormat(m dsl.Matcher) {
	m.Import("cdr.dev/slog/v3")
	m.Match(
		`logger.Error($ctx, $message, $*args)`,
		`logger.Warn($ctx, $message, $*args)`,
		`logger.Info($ctx, $message, $*args)`,
		`logger.Debug($ctx, $message, $*args)`,
		`$foo.logger.Error($ctx, $message, $*args)`,
		`$foo.logger.Warn($ctx, $message, $*args)`,
		`$foo.logger.Info($ctx, $message, $*args)`,
		`$foo.logger.Debug($ctx, $message, $*args)`,
		`Logger.Error($ctx, $message, $*args)`,
		`Logger.Warn($ctx, $message, $*args)`,
		`Logger.Info($ctx, $message, $*args)`,
		`Logger.Debug($ctx, $message, $*args)`,
		`$foo.Logger.Error($ctx, $message, $*args)`,
		`$foo.Logger.Warn($ctx, $message, $*args)`,
		`$foo.Logger.Info($ctx, $message, $*args)`,
		`$foo.Logger.Debug($ctx, $message, $*args)`,
	).
		Where(
			m["message"].Text.Matches(`[.!?]"$`) ||
				(m["message"].Text.Matches(`^"[A-Z]{1}`) &&
					!m["message"].Text.Matches(`^"Prometheus`) &&
					!m["message"].Text.Matches(`^"X11`) &&
					!m["message"].Text.Matches(`^"CSP`) &&
					!m["message"].Text.Matches(`^"OIDC`)),
		).
		Report(`Message $message must start with lowercase, and does not end with a special characters.`)
}

// slogMessageLength enforces minimum length for important log messages.
func slogMessageLength(m dsl.Matcher) {
	m.Import("cdr.dev/slog/v3")
	m.Match(
		`logger.Error($ctx, $message, $*args)`,
		`logger.Warn($ctx, $message, $*args)`,
		`logger.Info($ctx, $message, $*args)`,
		`$foo.logger.Error($ctx, $message, $*args)`,
		`$foo.logger.Warn($ctx, $message, $*args)`,
		`$foo.logger.Info($ctx, $message, $*args)`,
		`Logger.Error($ctx, $message, $*args)`,
		`Logger.Warn($ctx, $message, $*args)`,
		`Logger.Info($ctx, $message, $*args)`,
		`$foo.Logger.Error($ctx, $message, $*args)`,
		`$foo.Logger.Warn($ctx, $message, $*args)`,
		`$foo.Logger.Info($ctx, $message, $*args)`,
	).
		Where(
			m["message"].Text.Matches(`^".{0,15}"$`) &&
				!m["message"].Text.Matches(`^"command exit"$`),
		).
		Report(`Message $message is too short, it must be at least 16 characters long.`)
}

// slogError requires error values to use slog.Error rather than slog.F.
func slogError(m dsl.Matcher) {
	m.Import("cdr.dev/slog/v3")
	m.Match(`slog.F($name, $value)`).
		Where(m["name"].Const && m["value"].Type.Is("error") && !m["name"].Text.Matches(`^"internal_error"$`)).
		Report(`Error should be logged using "slog.Error" instead.`)
}
