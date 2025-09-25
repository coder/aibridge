package mcp

import (
	"context"
	"fmt"
	"sync"

	"github.com/coder/aibridge/utils"
	"github.com/mark3labs/mcp-go/mcp"
)

var _ ServerProxier = &ServerProxyManager{}

// ServerProxyManager can act on behalf of multiple [ServerProxier]s.
// It aggregates all server resources (currently just tools) across all MCP servers
// for the purpose of injection into bridged requests and invocation.
type ServerProxyManager struct {
	proxiers map[string]ServerProxier

	// Protects access to the tools map.
	toolsMu sync.RWMutex
	tools   map[string]*Tool
}

func NewServerProxyManager(proxiers map[string]ServerProxier) *ServerProxyManager {
	return &ServerProxyManager{proxiers: proxiers}
}

func (s *ServerProxyManager) addTools(tools []*Tool) {
	s.toolsMu.Lock()
	defer s.toolsMu.Unlock()

	if s.tools == nil {
		s.tools = make(map[string]*Tool, len(tools))
	}

	for _, tool := range tools {
		s.tools[tool.ID] = tool
	}
}

// Init concurrently initializes all of its [ServerProxier]s.
func (s *ServerProxyManager) Init(ctx context.Context) error {
	cg := utils.NewConcurrentGroup()
	for _, proxy := range s.proxiers {
		cg.Go(func() error {
			return proxy.Init(ctx)
		})
	}

	// Wait for all servers to initialize and load their tools.
	err := cg.Wait()

	// Aggregate all proxiers' tools.
	for _, proxy := range s.proxiers {
		s.addTools(proxy.ListTools())
	}

	return err
}

func (s *ServerProxyManager) GetTool(name string) *Tool {
	s.toolsMu.RLock()
	defer s.toolsMu.RUnlock()

	if s.tools == nil {
		return nil
	}

	return s.tools[name]
}

func (s *ServerProxyManager) ListTools() []*Tool {
	s.toolsMu.RLock()
	defer s.toolsMu.RUnlock()

	if s.tools == nil {
		return nil
	}

	var out []*Tool
	for _, tool := range s.tools {
		out = append(out, tool)
	}
	return out
}

// CallTool locates the proxier to which the requested tool is associated and
// delegates the tool call to it.
func (s *ServerProxyManager) CallTool(ctx context.Context, name string, input any) (*mcp.CallToolResult, error) {
	tool := s.GetTool(name)
	if tool == nil {
		return nil, fmt.Errorf("%q tool not known", name)
	}

	proxy, ok := s.proxiers[tool.ServerName]
	if !ok {
		return nil, fmt.Errorf("%q server not known", tool.ServerName)
	}

	return proxy.CallTool(ctx, name, input)
}

// Shutdown concurrently shuts down all known proxiers and waits for them *all* to complete.
func (s *ServerProxyManager) Shutdown(ctx context.Context) error {
	cg := utils.NewConcurrentGroup()
	for _, proxy := range s.proxiers {
		cg.Go(func() error {
			return proxy.Shutdown(ctx)
		})
	}
	return cg.Wait()
}
