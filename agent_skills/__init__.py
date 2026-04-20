# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Agent Skills - Reusable agent skill management.

This package provides two complementary approaches for managing agent skills:

## 1. Pydantic AI SkillsToolset (Recommended for Pydantic AI Agents)

Build on pydantic-ai's SkillsToolset pattern for progressive disclosure of skills.

Skills can be loaded in **three ways** that can be combined:

### Path-based loading

Point ``AgentSkillsToolset`` at one or more directories.  Every sub-directory
that contains a ``SKILL.md`` file is discovered automatically at first use.
Use this when skills live in the same repository or are mounted at a known path:

    from pydantic_ai import Agent
    from agent_skills import AgentSkillsToolset, SandboxExecutor
    from code_sandboxes.eval_sandbox import EvalSandbox

    toolset = AgentSkillsToolset(
        directories=["./skills"],       # scanned recursively for SKILL.md
        executor=SandboxExecutor(EvalSandbox()),
    )

    agent = Agent(model='openai:gpt-4o', toolsets=[toolset])
    # Agent gets: list_skills, load_skill, read_skill_resource, run_skill_script

### Module-based loading

Use ``AgentSkill.from_module()`` to load skills packaged inside an installed
Python library.  Works with both regular packages and namespace packages (no
``__init__.py``).  Pass the results via ``skills=``.  Use this when skills are
distributed as part of a pip-installable package:

    from pydantic_ai import Agent
    from agent_skills import AgentSkill, AgentSkillsToolset, SandboxExecutor
    from code_sandboxes.eval_sandbox import EvalSandbox

    toolset = AgentSkillsToolset(
        skills=[
            AgentSkill.from_module("agent_skills.skills.crawl"),
            AgentSkill.from_module("agent_skills.skills.github"),
            AgentSkill.from_module("agent_skills.skills.pdf"),
        ],
        executor=SandboxExecutor(EvalSandbox()),
    )

    agent = Agent(model='openai:gpt-4o', toolsets=[toolset])

### Entrypoint-based loading

Installed packages that register entrypoints in the ``agent_skills.skills``
group are discovered automatically — no configuration needed.  Use this when
skills are distributed as standalone pip-installable packages:

    # In some-skill-package/pyproject.toml:
    [project.entry-points."agent_skills.skills"]
    my-skill = "some_skill_package.skills.my_skill"

    # Skills are found automatically:
    toolset = AgentSkillsToolset(
        executor=SandboxExecutor(EvalSandbox()),
    )

Disable with ``discover_entrypoints=False`` if needed.

### Combining all three

The three approaches stack freely:

    toolset = AgentSkillsToolset(
        directories=["./skills"],           # local / custom skills
        skills=[
            AgentSkill.from_module("agent_skills.skills.crawl"),
        ],
        # discover_entrypoints=True (default) finds installed packages
        executor=SandboxExecutor(EvalSandbox()),
    )

## 2. Skills as Code Files (Primary Pattern)

Skills are Python files that agents can discover, create, execute, and compose.
This is the recommended approach for building agent skills.

Example:
    from agent_skills import SkillDirectory, setup_skills_directory
    
    # Set up skills directory (call once during initialization)
    skills = setup_skills_directory("./workspace/skills")
    
    # List available skills
    for skill in skills.list():
        print(f"{skill.name}: {skill.description}")
    
    # Get and execute a skill
    skill = skills.get("analyze_csv")
    func = skill.get_function()
    result = await func("/data/file.csv")
    
    # Create a new skill
    skills.create(
        name="process_batch",
        code='''
async def process_batch(input_dir: str) -> dict:
    from skills.analyze_csv import analyze_csv
    # ... compose with other skills
''',
        description="Process files in batch",
    )

For lifecycle management, versioning, and registries,
use the SkillsManager and MCP server.

Example:
    from agent_skills import SkillsManager, Skill
    
    # Create a skill manager
    manager = SkillsManager("./skills")
    
    # Discover and search skills
    skills = manager.discover()
    result = manager.search("data analysis")
    
    # MCP server for agent integration
    from agent_skills import skills_server, configure_server
    configure_server(skills_path="./skills")
    skills_server.run()
"""

# Primary Pattern: Skills as Code Files
from .files import (
    SkillFile,
    SkillDirectory,
    setup_skills_directory,
)

# Helpers for skill composition
from .helpers import (
    wait_for,
    retry,
    run_with_timeout,
    parallel,
    RateLimiter,
)

# Managed skills lifecycle
from .manager import SkillsManager
from .types import (
    Skill,
    SkillContext,
    SkillExecution,
    SkillHooks,
    SkillMetadata,
    SkillSearchResult,
    SkillStatus,
    SkillVersion,
)
from .server import mcp as skills_server, configure as configure_server
from .codegen import generate_skill_file, generate_skill_from_template

# Pydantic AI Integration (SkillsToolset pattern from PR #3780)
from .toolset import (
    AgentSkillsToolset,
    AgentSkill,
    AgentSkillResource,
    AgentSkillScript,
    SandboxExecutor,
    CallableExecutor,
    SkillScriptExecutorProtocol,
    ScriptExecutionResult,
    PYDANTIC_AI_AVAILABLE,
    discover_entrypoint_skills,
    SKILLS_ENTRYPOINT_GROUP,
)

__all__ = [
    # Manager
    "SkillsManager",
    # Models
    "Skill",
    "SkillMetadata",
    "SkillHooks",
    "SkillContext",
    "SkillStatus",
    "SkillExecution",
    "SkillVersion",
    "SkillSearchResult",
    # Code Generation
    "generate_skill_file",
    "generate_skill_from_template",
    # Primary Pattern: Skills as Code Files
    "SkillFile",
    "SkillDirectory",
    "setup_skills_directory",
    # Helpers for skill composition
    "wait_for",
    "retry",
    "run_with_timeout",
    "parallel",
    "RateLimiter",
    # MCP Server (Optional)
    "skills_server",
    "configure_server",
    # Pydantic AI Integration
    "AgentSkillsToolset",
    "AgentSkill",
    "AgentSkillResource",
    "AgentSkillScript",
    "SandboxExecutor",
    "CallableExecutor",
    "SkillScriptExecutorProtocol",
    "ScriptExecutionResult",
    "PYDANTIC_AI_AVAILABLE",
    # Entrypoint Discovery
    "discover_entrypoint_skills",
    "SKILLS_ENTRYPOINT_GROUP",
]
