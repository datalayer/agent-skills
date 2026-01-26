# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Agent Skills - Reusable agent skill management.

This package provides three complementary approaches for managing agent skills:

## 1. Pydantic AI SkillsToolset (Recommended for Pydantic AI Agents)

Build on pydantic-ai's SkillsToolset pattern for progressive disclosure of skills.

Example:
    from pydantic_ai import Agent
    from agent_skills import DatalayerSkillsToolset, SandboxExecutor
    from code_sandboxes import LocalEvalSandbox
    
    # Create toolset with sandbox execution
    sandbox = LocalEvalSandbox()
    toolset = DatalayerSkillsToolset(
        directories=["./skills"],
        executor=SandboxExecutor(sandbox),
    )
    
    # Use with pydantic-ai agent
    agent = Agent(
        model='openai:gpt-4o',
        toolsets=[toolset],
    )
    
    # Agent gets: list_skills, load_skill, read_skill_resource, run_skill_script

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

## 3. Managed Skills (Optional)

For advanced use cases like versioning, database storage, or skill registries,
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

# Simple skill management (backward compatible with agent_codemode)
from .simple import (
    SimpleSkill,
    SimpleSkillsManager,
    SimpleSkillManager,  # Alias for backward compatibility
    SkillManager,  # Alias for backward compatibility
)

# Optional: Managed Skills
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
    DatalayerSkillsToolset,
    DatalayerSkill,
    DatalayerSkillResource,
    DatalayerSkillScript,
    SandboxExecutor,
    LocalPythonExecutor,
    CallableExecutor,
    SkillScriptExecutorProtocol,
    ScriptExecutionResult,
    PYDANTIC_AI_AVAILABLE,
)

__all__ = [
    # Manager
    "SkillsManager",
    # Simple Manager (backward compatible)
    "SimpleSkill",
    "SimpleSkillsManager",
    "SimpleSkillManager",  # Alias for backward compatibility
    "SkillManager",  # Alias for backward compatibility
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
    "DatalayerSkillsToolset",
    "DatalayerSkill",
    "DatalayerSkillResource",
    "DatalayerSkillScript",
    "SandboxExecutor",
    "LocalPythonExecutor",
    "CallableExecutor",
    "SkillScriptExecutorProtocol",
    "ScriptExecutionResult",
    "PYDANTIC_AI_AVAILABLE",
]
