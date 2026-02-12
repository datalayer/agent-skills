<!--
  ~ Copyright (c) 2025-2026 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

[![Datalayer](https://assets.datalayer.tech/datalayer-25.svg)](https://datalayer.io)

[![Become a Sponsor](https://img.shields.io/static/v1?label=Become%20a%20Sponsor&message=%E2%9D%A4&logo=GitHub&style=flat&color=1ABC9C)](https://github.com/sponsors/datalayer)

# 🧰 Agent Skills

[![PyPI - Version](https://img.shields.io/pypi/v/agent-skills)](https://pypi.org/project/agent-skills)

**Reusable Agent Skills**: Create, manage, and execute reusable code-based tool compositions for AI agents.

For more information, see the [Agent Skills community website](https://agentskills.io), the [specification](https://agentskills.io/specification), and the [integration guide](https://agentskills.io/integrate-skills).

## Overview

Agent Skills provides a simple and powerful way for AI agents to build their own toolbox. Skills are Python files that compose MCP tools and other skills to accomplish specific tasks.

Agent Codemode consumes skills from this package. If you are using agent-codemode, import skill utilities from `agent_skills`.

### How It Works

1. **Skills are code files**: Python files in a `skills/` directory with async functions
2. **Agents discover skills**: By listing the skills directory and reading file contents
3. **Agents create skills**: By writing Python files to the skills directory
4. **Agents execute skills**: By importing and calling them in executed code
5. **Agents compose skills**: By importing multiple skills together

This pattern allows agents to evolve their own toolbox over time, saving useful compositions as reusable skills.

## Installation

```bash
pip install agent-skills
```

## Examples

See the runnable examples in [examples/README.md](examples/README.md).

```bash
python examples/skills_example.py
```

For Pydantic AI integration (requires the skills feature branch):

```bash
pip install "pydantic-ai @ git+https://github.com/DougTrajano/pydantic-ai.git@DEV-1099"
```

## Quick Start: Pydantic AI SkillsToolset (Recommended)

The recommended pattern for using agent-skills with Pydantic AI agents is via `AgentSkillsToolset`.

```python
from pydantic_ai import Agent
from agent_skills import AgentSkillsToolset, SandboxExecutor
from code_sandboxes import LocalEvalSandbox

# Create toolset with sandbox execution
sandbox = LocalEvalSandbox()
toolset = AgentSkillsToolset(
    directories=["./skills"],
    executor=SandboxExecutor(sandbox),
)

# Use with pydantic-ai agent
agent = Agent(
    model='openai:gpt-4o',
    toolsets=[toolset],
)

# The agent now has access to:
# - list_skills(): List available skills
# - load_skill(skill_name): Get full skill instructions
# - read_skill_resource(skill_name, resource_name): Read skill resources
# - run_skill_script(skill_name, script_name, args): Execute skill scripts
```

### Programmatic Skills

Define skills in Python code with decorators:

```python
from agent_skills import AgentSkill, AgentSkillsToolset

# Create a skill
skill = AgentSkill(
    name="data-analyzer",
    description="Analyzes datasets and provides insights",
    content="Use this skill to analyze CSV and JSON data files.",
)

# Add a script via decorator
@skill.script
async def analyze(ctx, file_path: str) -> str:
    """Analyze a data file."""
    # Access dependencies via ctx.deps
    data = await ctx.deps.filesystem.read(file_path)
    return f"Analyzed {len(data)} bytes"

# Add a resource
@skill.resource
def get_reference() -> str:
    return "Reference documentation..."

# Use with agent
toolset = AgentSkillsToolset(skills=[skill])
```

### SKILL.md Format

Skills on disk use YAML frontmatter in a `SKILL.md` file:

```markdown
---
name: pdf-extractor
description: Extract text and tables from PDF documents
version: "1.0.0"
allowed-tools: filesystem__read_file filesystem__write_file
denied-tools: network__fetch
tags:
  - pdf
  - extraction
---

# PDF Extractor Skill

Instructions for extracting content from PDF files...

## Usage

1. Use the `extract` script with a PDF path
2. Review the extracted content
```

With optional directories:
- `resources/`: Reference documents, templates, examples
- `references/`: Additional documentation loaded on demand (spec)
- `assets/`: Static resources like templates or data files (spec)
- `scripts/`: Executable Python scripts

Tool access policies in the frontmatter are surfaced in the skill summary and can be used by callers to enforce allow/deny lists.
Optional fields like `license`, `compatibility`, and `metadata` are supported per the Agent Skills specification.

## Quick Start: Skills as Code Files

The primary pattern for agent skills is simple: skills are just Python files.

### Setting Up the Skills Directory

```python
from agent_skills import setup_skills_directory

# Initialize the skills directory
skills = setup_skills_directory("./workspace/skills")
```

### Creating a Skill

Create a Python file in the skills directory:

```python
# skills/analyze_csv.py
#!/usr/bin/env python3
"""Analyze a CSV file and return statistics."""

async def analyze_csv(file_path: str) -> dict:
    """Analyze a CSV file.
    
    Args:
        file_path: Path to the CSV file.
    
    Returns:
        Statistics about the file.
    """
    from generated.mcp.filesystem import read_file
    
    content = await read_file({"path": file_path})
    lines = content.split("\n")
    headers = lines[0].split(",") if lines else []
    
    return {
        "rows": len(lines) - 1,
        "columns": len(headers),
        "headers": headers,
    }


# Optional: CLI support for direct execution
if __name__ == "__main__":
    import asyncio
    import sys
    
    result = asyncio.run(analyze_csv(sys.argv[1]))
    import json
    print(json.dumps(result, indent=2))
```

Or use the API:

```python
skills.create(
    name="analyze_csv",
    code='''
async def analyze_csv(file_path: str) -> dict:
    from generated.mcp.filesystem import read_file
    
    content = await read_file({"path": file_path})
    lines = content.split("\\n")
    headers = lines[0].split(",") if lines else []
    
    return {
        "rows": len(lines) - 1,
        "columns": len(headers),
        "headers": headers,
    }
''',
    description="Analyze a CSV file and return statistics",
)
```

### Using a Skill

In executed code, import and call the skill:

```python
from skills.analyze_csv import analyze_csv

result = await analyze_csv("/data/sales.csv")
print(f"Found {result['rows']} rows with columns: {result['headers']}")
```

### Discovering Skills

```python
from agent_skills import SkillDirectory

skills = SkillDirectory("./workspace/skills")

# List all skills
for skill in skills.list():
    print(f"{skill.name}: {skill.description}")
    print(f"  Functions: {', '.join(skill.functions)}")

# Search for relevant skills
matches = skills.search("data analysis")
for skill in matches:
    print(f"Found: {skill.name}")
```

### Composing Skills

Skills can import and use other skills:

```python
# skills/batch_analyze.py
"""Process and analyze multiple files."""

async def batch_analyze(directory: str) -> list:
    """Analyze all CSV files in a directory.
    
    Args:
        directory: Directory containing CSV files.
    
    Returns:
        List of analysis results.
    """
    from skills.analyze_csv import analyze_csv
    from generated.mcp.filesystem import list_directory
    
    entries = await list_directory({"path": directory})
    results = []
    
    for entry in entries.get("entries", []):
        if entry.endswith(".csv"):
            result = await analyze_csv(f"{directory}/{entry}")
            results.append({"file": entry, **result})
    
    return results
```

## API Reference

### SkillDirectory

The main interface for working with skills as code files.

```python
from agent_skills import SkillDirectory

skills = SkillDirectory("./workspace/skills")
```

#### Methods

- **`list() -> list[SkillFile]`**: List all skills in the directory
- **`get(name: str) -> SkillFile`**: Get a skill by name
- **`search(query: str, limit: int = 10) -> list[SkillFile]`**: Search for skills
- **`create(name, code, description, make_executable) -> SkillFile`**: Create a new skill
- **`delete(name: str) -> bool`**: Delete a skill
- **`add_to_sys_path()`**: Add skills directory to Python path for imports

### SkillFile

Represents a skill file with metadata and callable functions.

```python
skill = skills.get("analyze_csv")
print(skill.name)         # "analyze_csv"
print(skill.description)  # From module docstring
print(skill.functions)    # ["analyze_csv"]

# Load and call the function
func = skill.get_function()
result = await func("/data/file.csv")
```

### setup_skills_directory

Convenience function that creates a SkillDirectory and adds it to sys.path:

```python
from agent_skills import setup_skills_directory

# Call during sandbox initialization
skills = setup_skills_directory("./workspace/skills")

# Now executed code can do:
# from skills.my_skill import my_function
```

## Skill File Format

A skill file is a Python file with:

1. **Module docstring**: Description of what the skill does
2. **Async functions**: The skill's callable functions
3. **Optional CLI support**: For running the skill directly

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Short description of the skill.

Longer description with more details about what the skill does,
what inputs it expects, and what outputs it produces.
"""

async def main_function(param1: str, param2: int = 10) -> dict:
    """Main function description.
    
    Args:
        param1: First parameter description.
        param2: Second parameter with default.
    
    Returns:
        Dictionary with results.
    """
    # Import tools and other skills
    from generated.mcp.filesystem import read_file
    from skills.helper_skill import helper_function
    
    # Do the work
    content = await read_file({"path": param1})
    processed = await helper_function(content)
    
    return {"result": processed, "count": param2}


# CLI support (optional but recommended)
if __name__ == "__main__":
    import asyncio
    import sys
    import json
    
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <param1> [param2]")
        sys.exit(1)
    
    param1 = sys.argv[1]
    param2 = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    result = asyncio.run(main_function(param1, param2))
    print(json.dumps(result, indent=2))
```

## Best Practices

1. **One main function per skill**: Name it the same as the file
2. **Include docstrings**: Document what the skill does and its parameters
3. **Add CLI support**: Makes skills runnable standalone for testing
4. **Keep skills focused**: Do one thing well
5. **Compose skills**: Build complex workflows from simple skills
6. **Use type hints**: For better documentation and IDE support
7. **Handle errors gracefully**: Return meaningful error information

## Advanced: Managed Skills (Optional)

For advanced use cases like versioning, database storage, or skill registries,
you can use the SkillManager and MCP server:

```python
from agent_skills import SkillManager, skills_server, configure_server

# Create a skill manager for database-backed storage
manager = SkillManager("./skills")

# Discover SKILL.md format skills
skills = manager.discover()

# Search with ranking
result = manager.search("data processing", limit=5)

# Configure and run MCP server for agent integration
configure_server(skills_path="./skills")
skills_server.run()
```

See the full documentation for details on the managed skills API.

## Integration with MCP Codemode

Agent Skills works seamlessly with the `agent-codemode` package:

```python
from agent_codemode import CodemodeClient
from agent_skills import setup_skills_directory

# Set up the skills directory
skills = setup_skills_directory("./workspace/skills")

# Create codemode client
client = CodemodeClient()

# Skills are available in executed code
result = await client.execute_code('''
from skills.analyze_csv import analyze_csv

# Call the skill
data = await analyze_csv("/data/sales.csv")
print(f"Analyzed {data['rows']} rows")
''')
```

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.
