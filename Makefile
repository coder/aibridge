# Use a single bash shell for each job, and immediately exit on failure
SHELL := bash
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

SHELL_SRC_FILES := $(shell find . -type f -name '*.sh' -not -path '*/.git/*')
GOLANGCI_LINT_VERSION ?= 1.64.8
PARALLELTESTCTX_VERSION ?= 0.0.1

lint: lint/shellcheck lint/go
.PHONY: lint

lint/go:
	go run github.com/golangci/golangci-lint/cmd/golangci-lint@v$(GOLANGCI_LINT_VERSION) run
	go run github.com/coder/paralleltestctx/cmd/paralleltestctx@v$(PARALLELTESTCTX_VERSION) -custom-funcs="testutil.Context" ./...
.PHONY: lint/go

lint/shellcheck:
	if test -n "$(strip $(SHELL_SRC_FILES))"; then \
		echo "--- shellcheck"; \
		shellcheck --external-sources $(SHELL_SRC_FILES); \
	fi
.PHONY: lint/shellcheck

test:
	go test -count=1 ./...

test-race:
	CGO_ENABLED=1 go test -count=1 -race ./...

coverage:
	go test -coverprofile=coverage.out -coverpkg=./... ./...
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
