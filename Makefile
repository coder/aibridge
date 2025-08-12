# Use a single bash shell for each job, and immediately exit on failure
.SHELL := /usr/bin/env bash
.SHELLFLAGS := -ceu
.ONESHELL:

# This doesn't work on directories.
# See https://stackoverflow.com/questions/25752543/make-delete-on-error-for-directory-targets
.DELETE_ON_ERROR:

# Don't print the commands in the file unless you specify VERBOSE. This is
# essentially the same as putting "@" at the start of each line.
ifndef VERBOSE
.SILENT:
endif

test:
	go test -count=1 ./...

test-race:
	CGO_ENABLED=1 go test -count=1 -race ./...

coverage:
	go test -coverprofile=coverage.out ./...
	go tool cover -func=coverage.out | tail -n 1

coverage-html:
	@go test -coverprofile=coverage.out ./...
	@go tool cover -html=coverage.out

clean:
	rm -f coverage.out

fmt: fmt/go
.PHONY: fmt

fmt/go:
ifdef FILE
	# Format single file
	if [[ -f "$(FILE)" ]] && [[ "$(FILE)" == *.go ]] && ! grep -q "DO NOT EDIT" "$(FILE)"; then \
		go run mvdan.cc/gofumpt@v0.8.0 -w -l "$(FILE)"; \
	fi
else
	go mod tidy
	find . -type f -name '*.go' -print0 | \
		xargs -0 grep -E --null -L '^// Code generated .* DO NOT EDIT\.$$' | \
		xargs -0 go run mvdan.cc/gofumpt@v0.8.0 -w -l
endif
.PHONY: fmt/go

mocks: mcp/api.go
	go generate ./mcpmock/