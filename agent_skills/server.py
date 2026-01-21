# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Agent Skills MCP Server.

Provides an MCP server that exposes agent skills as tools,
allowing AI agents to:
- Discover available skills
- Search for relevant skills
- Execute skills with parameters
- Create and manage skills

Based on:
- Claude Code Skills format (SKILL.md)
- Anthropic SDK Skills API
"""

import logging
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from .manager import SkillsManager
from .models import Skill, SkillContext, SkillMetadata, SkillStatus

logger = logging.getLogger(__name__)

# Create the MCP server
mcp = FastMCP("Agent Skills 🎯")

# Global skill manager
_manager: Optional[SkillsManager] = None


def configure(
    skills_path: str = "./skills",
    sandbox_variant: str = "local-eval",
) -> SkillsManager:
    """Configure the Agent Skills MCP server.
    
    Args:
        skills_path: Directory for skill storage.
        sandbox_variant: Sandbox type for execution.
        
    Returns:
        Configured SkillsManager.
    """
    global _manager
    _manager = SkillsManager(skills_path, sandbox_variant)
    logger.info(f"Agent Skills server configured with path: {skills_path}")
    return _manager


def get_manager() -> SkillsManager:
    """Get the skill manager."""
    global _manager
    if _manager is None:
        configure()
    return _manager


# =============================================================================
# Skill Discovery Tools
# =============================================================================

@mcp.tool()
async def search_skills(
    query: str,
    tags: Optional[list[str]] = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search for skills matching a query.
    
    Find relevant skills based on natural language description,
    name, or tags.
    
    Args:
        query: Natural language description of what you need.
               Examples: "data analysis", "file processing"
        tags: Optional list of tags to filter by.
        limit: Maximum number of results (default: 10).
        
    Returns:
        Dictionary with:
        - skills: List of matching skills (name, description, tags)
        - total: Total number of matches
        - query: The search query used
        
    Example:
        # Find data processing skills
        result = search_skills("process CSV files")
    """
    manager = get_manager()
    result = manager.search(query, limit=limit)
    
    skills = result.skills
    if tags:
        skills = [s for s in skills if any(t in s.metadata.tags for t in tags)]
    
    return {
        "skills": [
            {
                "skill_id": s.skill_id,
                "name": s.name,
                "description": s.description,
                "tags": s.metadata.tags,
                "allowed_tools": s.metadata.allowed_tools,
            }
            for s in skills
        ],
        "total": result.total,
        "query": query,
    }


@mcp.tool()
async def list_skills(
    tags: Optional[list[str]] = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List all available skills.
    
    Get an overview of all skills that can be executed.
    
    Args:
        tags: Optional filter by tags.
        limit: Maximum number to return (default: 50).
        
    Returns:
        Dictionary with list of skills.
    """
    manager = get_manager()
    skills = manager.list(tags=tags, limit=limit)
    
    return {
        "skills": [
            {
                "skill_id": s.skill_id,
                "name": s.name,
                "description": s.description,
                "tags": s.metadata.tags,
            }
            for s in skills
        ],
        "total": len(skills),
    }


@mcp.tool()
async def get_skill(skill_id: str) -> dict[str, Any]:
    """Get detailed information about a skill.
    
    Retrieve the full skill definition including instructions
    and code.
    
    Args:
        skill_id: The skill's unique ID or name.
        
    Returns:
        Dictionary with full skill details.
    """
    manager = get_manager()
    
    # Try by ID first
    skill = manager.retrieve(skill_id)
    
    # Try by name if not found
    if not skill:
        skill = manager.get(skill_id)
    
    if not skill:
        return {"error": f"Skill not found: {skill_id}"}
    
    return {
        "skill_id": skill.skill_id,
        "name": skill.name,
        "description": skill.description,
        "content": skill.content,
        "python_code": skill.python_code,
        "tags": skill.metadata.tags,
        "allowed_tools": skill.metadata.allowed_tools,
        "context": skill.metadata.context.value,
        "version": skill.metadata.version,
    }


# =============================================================================
# Skill Execution Tools
# =============================================================================

@mcp.tool()
async def run_skill(
    skill_id: str,
    arguments: Optional[dict[str, Any]] = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Execute a skill.
    
    Run a skill with the provided arguments in an isolated sandbox.
    
    Args:
        skill_id: The skill's unique ID or name.
        arguments: Arguments to pass to the skill (as variables).
        timeout: Maximum execution time in seconds.
        
    Returns:
        Dictionary with:
        - success: Whether execution completed
        - result: The skill's output
        - logs: Captured stdout/stderr
        - execution_time: Time taken
        - error: Error message if failed
        
    Example:
        # Run a data analysis skill
        result = run_skill("analyze_csv", {"file_path": "data.csv"})
    """
    manager = get_manager()
    
    # Find the skill
    skill = manager.retrieve(skill_id)
    if not skill:
        skill = manager.get(skill_id)
    
    if not skill:
        return {
            "success": False,
            "result": None,
            "logs": "",
            "execution_time": 0,
            "error": f"Skill not found: {skill_id}",
        }
    
    # Activate the skill
    if not manager.activate(skill):
        return {
            "success": False,
            "result": None,
            "logs": "",
            "execution_time": 0,
            "error": "Skill activation failed",
        }
    
    # Execute
    execution = await manager.execute(skill, arguments, timeout)
    
    return {
        "success": execution.success,
        "result": execution.result,
        "logs": execution.logs or "",
        "execution_time": execution.execution_time,
        "error": execution.error,
        "tool_calls": execution.tool_calls,
    }


# =============================================================================
# Skill Management Tools
# =============================================================================

@mcp.tool()
async def create_skill(
    name: str,
    description: str,
    content: str,
    python_code: Optional[str] = None,
    allowed_tools: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Create a new skill.
    
    Define a reusable skill with instructions and optional code.
    
    Args:
        name: Unique name for the skill.
        description: What the skill does (1-2 sentences).
        content: Markdown content with detailed instructions.
        python_code: Optional Python implementation.
        allowed_tools: List of tools this skill can use.
        tags: Tags for categorization.
        
    Returns:
        Dictionary with created skill info.
        
    Example:
        create_skill(
            name="analyze_csv",
            description="Analyze a CSV file and generate summary statistics",
            content="# CSV Analysis\\n\\nThis skill processes CSV files...",
            python_code="import pandas as pd\\ndf = pd.read_csv(file_path)...",
            tags=["data", "analysis"]
        )
    """
    manager = get_manager()
    
    try:
        skill = manager.create(
            name=name,
            description=description,
            content=content,
            python_code=python_code,
            allowed_tools=allowed_tools,
            tags=tags,
        )
        
        return {
            "success": True,
            "skill_id": skill.skill_id,
            "name": skill.name,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "skill_id": None,
            "name": name,
            "error": str(e),
        }


@mcp.tool()
async def update_skill(
    skill_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    content: Optional[str] = None,
    python_code: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Update an existing skill.
    
    Modify any of the skill's properties.
    
    Args:
        skill_id: The skill's unique ID.
        Other args: Fields to update (None = keep current).
        
    Returns:
        Dictionary with update status.
    """
    manager = get_manager()
    
    skill = manager.update(
        skill_id=skill_id,
        name=name,
        description=description,
        content=content,
        python_code=python_code,
        tags=tags,
    )
    
    if skill:
        return {
            "success": True,
            "skill_id": skill.skill_id,
            "name": skill.name,
            "error": None,
        }
    else:
        return {
            "success": False,
            "skill_id": skill_id,
            "name": None,
            "error": f"Skill not found: {skill_id}",
        }


@mcp.tool()
async def delete_skill(skill_id: str) -> dict[str, Any]:
    """Delete a skill.
    
    Permanently remove a skill.
    
    Args:
        skill_id: The skill's unique ID.
        
    Returns:
        Dictionary with deletion status.
    """
    manager = get_manager()
    
    success = manager.delete(skill_id)
    
    return {
        "success": success,
        "error": None if success else f"Skill not found: {skill_id}",
    }


# =============================================================================
# Skill Versioning Tools
# =============================================================================

@mcp.tool()
async def create_skill_version(
    skill_id: str,
    version: str,
) -> dict[str, Any]:
    """Create a new version of a skill.
    
    Save the current state of a skill as a named version.
    
    Args:
        skill_id: The skill's unique ID.
        version: Version string (e.g., "1.1.0").
        
    Returns:
        Dictionary with version info.
    """
    manager = get_manager()
    
    version_obj = manager.create_version(skill_id, version)
    
    if version_obj:
        return {
            "success": True,
            "version_id": version_obj.version_id,
            "version": version_obj.version,
            "error": None,
        }
    else:
        return {
            "success": False,
            "version_id": None,
            "version": version,
            "error": f"Skill not found: {skill_id}",
        }


@mcp.tool()
async def list_skill_versions(skill_id: str) -> dict[str, Any]:
    """List all versions of a skill.
    
    Args:
        skill_id: The skill's unique ID.
        
    Returns:
        Dictionary with list of versions.
    """
    manager = get_manager()
    
    versions = manager.list_versions(skill_id)
    
    return {
        "versions": [
            {
                "version_id": v.version_id,
                "version": v.version,
                "created_at": v.created_at.isoformat(),
                "is_current": v.is_current,
            }
            for v in versions
        ],
        "total": len(versions),
    }


# =============================================================================
# Discovery Tool (for importing from SKILL.md files)
# =============================================================================

@mcp.tool()
async def discover_skills(path: Optional[str] = None) -> dict[str, Any]:
    """Discover skills from a directory.
    
    Scan a directory for SKILL.md files and import them.
    
    Args:
        path: Optional path to scan (uses default if not provided).
        
    Returns:
        Dictionary with discovered skills info.
    """
    manager = get_manager()
    
    if path:
        # Temporarily update path
        from pathlib import Path
        original_path = manager.skills_path
        manager.skills_path = Path(path)
        skills = manager.discover()
        manager.skills_path = original_path
    else:
        skills = manager.discover()
    
    return {
        "discovered": len(skills),
        "skills": [
            {
                "skill_id": s.skill_id,
                "name": s.name,
                "description": s.description,
            }
            for s in skills
        ],
    }


def run() -> None:
    """Run the MCP server."""
    configure()
    mcp.run()


if __name__ == "__main__":
    run()

