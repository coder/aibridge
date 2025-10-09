# AI Bridge

AI Bridge is a smart proxy for AI. It acts as a man-in-the-middle between your users' coding agents / IDEs and providers like OpenAI and Anthropic. By intercepting all the AI traffic between these clients and the upstream APIs, Bridge can record user prompts, token usage, and tool invocations. AI Bridge is bundled as part of [Coder](https://github.com/coder/coder)

It solves 3 key problems:

1. Centralized authn/z management: no more issuing & managing API tokens for OpenAI/Anthropic usage. Users use their Coder session or API tokens to authenticate with coderd (Coder control plane), and coderd securely communicates with the upstream APIs on their behalf. Use a single key for all users.
2. Auditing and attribution: all interactions with AI services, whether autonomous or human-initiated, will be audited and attributed back to a user.
3. Centralized MCP administration: define a set of approved MCP servers and tools which your users may use, and prevent users from using their own.

For furrther details on how to use, please see: https://coder.com/docs/ai-coder/ai-bridge
