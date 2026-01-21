# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Simple Skill Manager - Backward compatible skill storage.

This module provides a simpler skill management approach for use cases
that don't need the full SkillsManager with versioning and lifecycle hooks.

Skills are stored as JSON files with accompanying Python files for easy viewing.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SimpleSkill:
    """A simple reusable skill composed of tool calls.

    Skills are saved code patterns that compose multiple tools
    to accomplish a specific task.

    This is a simpler model than the full Skill class, suitable for
    basic skill storage without metadata, versioning, or lifecycle hooks.

    Attributes:
        name: Unique skill name.
        description: Human-readable description.
        code: The Python code implementing the skill.
        tools_used: List of tool names used by this skill.
        tags: Optional tags for categorization.
        parameters: Optional JSON schema for skill parameters.
        created_at: Unix timestamp when created.
        updated_at: Unix timestamp when last updated.
    """

    name: str
    description: str
    code: str
    tools_used: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0

    def __repr__(self) -> str:
        return f"SimpleSkill(name={self.name!r}, tools={len(self.tools_used)})"


class SimpleSkillsManager:
    """Manager for saving and loading simple skills.

    Skills are reusable code patterns that compose MCP tools.
    They can be saved to disk and loaded later for execution.

    This provides a simpler interface than SkillsManager, suitable
    for basic use cases without versioning or lifecycle hooks.

    Example:
        manager = SimpleSkillsManager("./skills")

        # Save a skill
        skill = manager.save_skill(
            name="backup_to_cloud",
            description="Backup files to cloud storage",
            code='''
                files = await bash__ls({"path": source})
                for f in files:
                    await cloud__upload({"file": f, "bucket": bucket})
            ''',
            tools_used=["bash__ls", "cloud__upload"],
        )

        # Load and execute later
        skill = manager.load_skill("backup_to_cloud")
    """

    def __init__(self, skills_path: str = "./skills"):
        """Initialize the skill manager.

        Args:
            skills_path: Directory for storing skills.
        """
        self.skills_path = Path(skills_path)
        self.skills_path.mkdir(parents=True, exist_ok=True)

    def save_skill(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        code: Optional[str] = None,
        tools_used: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        parameters: Optional[dict] = None,
        skill: Optional[SimpleSkill] = None,
    ) -> SimpleSkill:
        """Save a skill to disk.

        Can be called with individual arguments or with a SimpleSkill object.

        Args:
            name: Unique skill name.
            description: Human-readable description.
            code: Python code implementing the skill.
            tools_used: List of tool names used by the skill.
            tags: Optional tags for categorization.
            parameters: Optional JSON schema for skill parameters.
            skill: Optional SimpleSkill object (if provided, other args are ignored).

        Returns:
            The saved SimpleSkill object.
        """
        now = time.time()

        # Handle SimpleSkill object passed directly
        if skill is not None:
            name = skill.name
            description = skill.description
            code = skill.code
            tools_used = skill.tools_used
            tags = skill.tags if hasattr(skill, 'tags') else []
            parameters = skill.parameters if hasattr(skill, 'parameters') else {}

        if name is None or code is None:
            raise ValueError("name and code are required")

        # Check if skill exists to preserve created_at
        existing = self.load_skill(name)
        created_at = existing.created_at if existing else now

        saved_skill = SimpleSkill(
            name=name,
            description=description or "",
            code=code,
            tools_used=tools_used or [],
            tags=tags or [],
            parameters=parameters or {},
            created_at=created_at,
            updated_at=now,
        )

        # Save as JSON
        skill_path = self.skills_path / f"{name}.json"
        skill_data = {
            "name": saved_skill.name,
            "description": saved_skill.description,
            "code": saved_skill.code,
            "tools_used": saved_skill.tools_used,
            "tags": saved_skill.tags,
            "parameters": saved_skill.parameters,
            "created_at": saved_skill.created_at,
            "updated_at": saved_skill.updated_at,
        }
        skill_path.write_text(json.dumps(skill_data, indent=2))

        # Also save the code as a .py file for easy viewing
        code_path = self.skills_path / f"{name}.py"
        code_content = f'''# Skill: {name}
# {description or ""}
# Tools used: {", ".join(tools_used or [])}
# Tags: {", ".join(tags or [])}
# Created: {time.ctime(created_at)}
# Updated: {time.ctime(now)}

{code}
'''
        code_path.write_text(code_content)

        return saved_skill

    def load_skill(self, name: str) -> Optional[SimpleSkill]:
        """Load a skill from disk.

        Args:
            name: Skill name.

        Returns:
            SimpleSkill object or None if not found.
        """
        skill_path = self.skills_path / f"{name}.json"
        if not skill_path.exists():
            return None

        try:
            skill_data = json.loads(skill_path.read_text())
            return SimpleSkill(
                name=skill_data["name"],
                description=skill_data["description"],
                code=skill_data["code"],
                tools_used=skill_data.get("tools_used", []),
                tags=skill_data.get("tags", []),
                parameters=skill_data.get("parameters", {}),
                created_at=skill_data.get("created_at", 0),
                updated_at=skill_data.get("updated_at", 0),
            )
        except Exception:
            return None

    def delete_skill(self, name: str) -> bool:
        """Delete a skill.

        Args:
            name: Skill name.

        Returns:
            True if deleted, False if not found.
        """
        skill_path = self.skills_path / f"{name}.json"
        code_path = self.skills_path / f"{name}.py"

        deleted = False
        if skill_path.exists():
            skill_path.unlink()
            deleted = True
        if code_path.exists():
            code_path.unlink()

        return deleted

    def list_skills(self) -> list[SimpleSkill]:
        """List all saved skills.

        Returns:
            List of SimpleSkill objects.
        """
        skills = []
        for skill_path in self.skills_path.glob("*.json"):
            skill = self.load_skill(skill_path.stem)
            if skill:
                skills.append(skill)

        # Sort by updated_at descending
        skills.sort(key=lambda s: s.updated_at, reverse=True)
        return skills

    def search_skills(
        self, query: str, limit: int = 10
    ) -> list[SimpleSkill]:
        """Search skills by keyword.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            Matching skills.
        """
        query_lower = query.lower()
        query_words = query_lower.split()

        scored_skills = []
        for skill in self.list_skills():
            skill_text = f"{skill.name} {skill.description} {' '.join(skill.tools_used)} {' '.join(skill.tags)}".lower()
            score = sum(1 for word in query_words if word in skill_text)
            if score > 0:
                scored_skills.append((score, skill))

        scored_skills.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored_skills[:limit]]

    def get_skill_code(self, name: str) -> Optional[str]:
        """Get just the code for a skill.

        Args:
            name: Skill name.

        Returns:
            Skill code or None if not found.
        """
        skill = self.load_skill(name)
        return skill.code if skill else None

    def __repr__(self) -> str:
        return f"SimpleSkillsManager(path={self.skills_path!r})"


# Backward compatibility aliases
SimpleSkillManager = SimpleSkillsManager
SkillManager = SimpleSkillsManager
SkillsManagerSimple = SimpleSkillsManager
