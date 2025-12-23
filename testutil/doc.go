// Package testutil contains helpers for testing AIBridge.
//
// # Stability
//
// This package is intended for tests and examples within the Coder ecosystem.
// It is not considered a stable public API.
//
// The goal is to make AIBridge tests:
//   - easier to read
//   - easier to extend
//   - less reliant on copy/paste setup
//
// In particular, it provides typed accessors for txtar fixtures and small
// harness structs for standing up mock upstream/MCP/bridge servers.
package testutil
