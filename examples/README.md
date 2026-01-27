# Agent Skills Examples

This directory contains runnable examples demonstrating the agent-skills framework.

## Structure

```
examples/
├── README.md           # This file
└── simple/
    ├── README.md       # Simple examples documentation
    └── skills_example.py  # Comprehensive skills example
```

## Quick Start

```bash
cd examples/simple
python skills_example.py
```

## Examples Overview

### Simple Examples

The `simple/` directory contains a comprehensive example that demonstrates:

| Example | Description |
|---------|-------------|
| **Skill Creation** | Create skills programmatically via the SkillsManager API |
| **SKILL.md Format** | Parse and generate Claude Code compatible SKILL.md files |
| **Skill Discovery** | Discover and search skills from directories |
| **Skill Execution** | Execute skills in sandboxes with arguments |
| **Skill Versioning** | Version management for skills |
| **Skills as Code** | Simple code file-based skills with SimpleSkillsManager |
| **MCP Server** | Expose skills via MCP protocol |

See [simple/README.md](simple/README.md) for details.

## Integration with Agent Codemode

For skills integrated with Agent Codemode (code-first tool composition), see:
- [agent-codemode/examples/skills/](https://github.com/datalayer/agent-codemode/tree/main/examples/skills)

This demonstrates using `AgentSkillsToolset` alongside `CodemodeToolset` for agents with both MCP tools and skills.
