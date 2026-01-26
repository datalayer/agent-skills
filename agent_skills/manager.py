# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Skill Manager - Lifecycle management for agent skills.

Handles:
- Skill discovery (from directories, registries)
- Skill activation (based on context)
- Skill execution (using code sandboxes)
- Skill versioning
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .models import (
    Skill,
    SkillContext,
    SkillExecution,
    SkillMetadata,
    SkillSearchResult,
    SkillStatus,
    SkillVersion,
)

logger = logging.getLogger(__name__)


class SkillsManager:
    """Manager for agent skills lifecycle.
    
    Provides functionality similar to Anthropic's Skills API
    while also supporting Claude Code's SKILL.md format.
    
    Example:
        manager = SkillsManager("./skills")
        
        # Discover skills
        skills = manager.discover()
        
        # Find relevant skills
        matches = manager.search("data analysis")
        
        # Activate and execute
        skill = manager.get("analyze_csv")
        result = await manager.execute(skill, {"file": "data.csv"})
    """
    
    def __init__(
        self,
        skills_path: str = "./skills",
        sandbox_variant: str = "local-eval",
    ):
        """Initialize the skill manager.
        
        Args:
            skills_path: Directory for skill storage.
            sandbox_variant: Sandbox type for execution.
        """
        self.skills_path = Path(skills_path)
        self.sandbox_variant = sandbox_variant
        
        # In-memory skill cache
        self._skills: dict[str, Skill] = {}
        self._versions: dict[str, list[SkillVersion]] = {}
        
        # Ensure skills directory exists
        self.skills_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SkillsManager initialized with path: {skills_path}")
    
    # =========================================================================
    # Skill CRUD Operations (Anthropic API style)
    # =========================================================================
    
    def create(
        self,
        name: str,
        description: str,
        content: str,
        python_code: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        context: SkillContext = SkillContext.FORK,
    ) -> Skill:
        """Create a new skill.
        
        Args:
            name: Skill name.
            description: Human-readable description.
            content: Markdown content with instructions.
            python_code: Optional Python implementation.
            allowed_tools: Tools this skill can use.
            tags: Tags for categorization.
            context: Execution context.
            
        Returns:
            Created Skill object.
        """
        skill_id = str(uuid.uuid4())
        
        metadata = SkillMetadata(
            name=name,
            description=description,
            allowed_tools=allowed_tools or [],
            tags=tags or [],
            context=context,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        skill = Skill(
            metadata=metadata,
            content=content,
            python_code=python_code,
            skill_id=skill_id,
        )
        
        # Save to cache and disk
        self._skills[skill_id] = skill
        self._save_skill_to_disk(skill)
        
        logger.info(f"Created skill: {name} ({skill_id})")
        return skill
    
    def retrieve(self, skill_id: str) -> Optional[Skill]:
        """Retrieve a skill by ID.
        
        Args:
            skill_id: The skill's unique ID.
            
        Returns:
            Skill if found, None otherwise.
        """
        if skill_id in self._skills:
            return self._skills[skill_id]
        
        # Try loading from disk
        skill = self._load_skill_from_disk(skill_id)
        if skill:
            self._skills[skill_id] = skill
        
        return skill
    
    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name.
        
        Args:
            name: The skill name.
            
        Returns:
            Skill if found, None otherwise.
        """
        for skill in self._skills.values():
            if skill.name == name:
                return skill
        
        # Try discovering from disk
        self.discover()
        
        for skill in self._skills.values():
            if skill.name == name:
                return skill
        
        return None
    
    def update(
        self,
        skill_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        content: Optional[str] = None,
        python_code: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
    ) -> Optional[Skill]:
        """Update an existing skill.
        
        Args:
            skill_id: The skill's unique ID.
            Other args: Fields to update.
            
        Returns:
            Updated Skill if found, None otherwise.
        """
        skill = self.retrieve(skill_id)
        if not skill:
            return None
        
        if name:
            skill.metadata.name = name
        if description:
            skill.metadata.description = description
        if content:
            skill.content = content
        if python_code:
            skill.python_code = python_code
        if allowed_tools is not None:
            skill.metadata.allowed_tools = allowed_tools
        if tags is not None:
            skill.metadata.tags = tags
        
        skill.metadata.updated_at = datetime.now()
        
        self._save_skill_to_disk(skill)
        
        logger.info(f"Updated skill: {skill.name} ({skill_id})")
        return skill
    
    def delete(self, skill_id: str) -> bool:
        """Delete a skill.
        
        Args:
            skill_id: The skill's unique ID.
            
        Returns:
            True if deleted, False if not found.
        """
        if skill_id not in self._skills:
            return False
        
        skill = self._skills.pop(skill_id)
        
        # Delete from disk
        skill_file = self.skills_path / f"{skill_id}.skill.md"
        if skill_file.exists():
            skill_file.unlink()
        
        logger.info(f"Deleted skill: {skill.name} ({skill_id})")
        return True
    
    def list(
        self,
        tags: Optional[list[str]] = None,
        status: Optional[SkillStatus] = None,
        limit: int = 100,
    ) -> list[Skill]:
        """List all skills with optional filters.
        
        Args:
            tags: Filter by tags.
            status: Filter by status.
            limit: Maximum number to return.
            
        Returns:
            List of matching skills.
        """
        self.discover()  # Ensure we have latest
        
        skills = list(self._skills.values())
        
        if tags:
            skills = [s for s in skills if any(t in s.metadata.tags for t in tags)]
        
        if status:
            skills = [s for s in skills if s.status == status]
        
        return skills[:limit]
    
    # =========================================================================
    # Skill Discovery
    # =========================================================================
    
    def discover(self) -> list[Skill]:
        """Discover skills from the skills directory.
        
        Scans for:
        - SKILL.md files (Claude Code format)
        - *.skill.md files (our format)
        - *.skill.json files (JSON format)
        
        Returns:
            List of discovered skills.
        """
        discovered = []
        
        if not self.skills_path.exists():
            return discovered
        
        # Look for SKILL.md files
        for skill_file in self.skills_path.rglob("SKILL.md"):
            try:
                skill = self._load_skill_md(skill_file)
                if skill.skill_id not in self._skills:
                    self._skills[skill.skill_id or skill.name] = skill
                    discovered.append(skill)
            except Exception as e:
                logger.warning(f"Failed to load skill from {skill_file}: {e}")
        
        # Look for *.skill.md files
        for skill_file in self.skills_path.rglob("*.skill.md"):
            try:
                skill = self._load_skill_md(skill_file)
                skill_id = skill.skill_id or skill_file.stem.replace(".skill", "")
                if skill_id not in self._skills:
                    self._skills[skill_id] = skill
                    discovered.append(skill)
            except Exception as e:
                logger.warning(f"Failed to load skill from {skill_file}: {e}")
        
        # Look for *.skill.json files
        for skill_file in self.skills_path.rglob("*.skill.json"):
            try:
                skill = self._load_skill_json(skill_file)
                skill_id = skill.skill_id or skill_file.stem.replace(".skill", "")
                if skill_id not in self._skills:
                    self._skills[skill_id] = skill
                    discovered.append(skill)
            except Exception as e:
                logger.warning(f"Failed to load skill from {skill_file}: {e}")
        
        logger.info(f"Discovered {len(discovered)} new skills")
        return discovered
    
    # =========================================================================
    # Skill Search and Activation
    # =========================================================================
    
    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> SkillSearchResult:
        """Search for skills matching a query.
        
        Uses simple text matching on name, description, and tags.
        
        Args:
            query: Search query.
            limit: Maximum results.
            
        Returns:
            Search results with matching skills.
        """
        self.discover()
        
        query_lower = query.lower()
        matches = []
        
        for skill in self._skills.values():
            score = 0
            
            # Check name
            if query_lower in skill.name.lower():
                score += 3
            
            # Check description
            if query_lower in skill.description.lower():
                score += 2
            
            # Check tags
            for tag in skill.metadata.tags:
                if query_lower in tag.lower():
                    score += 1
            
            # Check content
            if query_lower in skill.content.lower():
                score += 0.5
            
            if score > 0:
                matches.append((score, skill))
        
        # Sort by score
        matches.sort(key=lambda x: x[0], reverse=True)
        
        skills = [s for _, s in matches[:limit]]
        
        return SkillSearchResult(
            skills=skills,
            total=len(matches),
            query=query,
        )
    
    def activate(self, skill: Skill) -> bool:
        """Activate a skill (mark as ready for use).
        
        This is a lifecycle hook - can run before_invoke hooks.
        
        Args:
            skill: Skill to activate.
            
        Returns:
            True if activation succeeded.
        """
        if skill.metadata.hooks and skill.metadata.hooks.before_invoke:
            try:
                import subprocess
                subprocess.run(
                    skill.metadata.hooks.before_invoke,
                    shell=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Activation hook failed: {e}")
                return False
        
        logger.info(f"Activated skill: {skill.name}")
        return True
    
    # =========================================================================
    # Skill Execution
    # =========================================================================
    
    async def execute(
        self,
        skill: Skill,
        arguments: Optional[dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> SkillExecution:
        """Execute a skill.
        
        Runs the skill's code in the configured sandbox with access
        to the allowed tools.
        
        Args:
            skill: Skill to execute.
            arguments: Arguments to pass to the skill.
            timeout: Execution timeout in seconds.
            
        Returns:
            SkillExecution with results.
        """
        from code_sandboxes import Sandbox
        
        start_time = datetime.now()
        skill_id = skill.skill_id or skill.name
        
        # Get the code to execute
        code = skill.python_code
        if not code:
            return SkillExecution(
                skill_id=skill_id,
                success=False,
                error="Skill has no Python code to execute",
            )
        
        try:
            # Create sandbox based on context
            sandbox_variant = self.sandbox_variant
            if skill.metadata.context == SkillContext.SANDBOX:
                sandbox_variant = "datalayer-runtime"
            
            with Sandbox.create(variant=sandbox_variant) as sandbox:
                # Inject arguments
                if arguments:
                    for name, value in arguments.items():
                        sandbox.set_variable(name, value)
                
                # Execute the code
                execution = sandbox.run_code(code, timeout=timeout)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Run after hook if defined
                if skill.metadata.hooks and skill.metadata.hooks.after_invoke:
                    try:
                        import subprocess
                        subprocess.run(
                            skill.metadata.hooks.after_invoke,
                            shell=True,
                            check=True,
                        )
                    except Exception as e:
                        logger.warning(f"After-invoke hook failed: {e}")
                
                return SkillExecution(
                    skill_id=skill_id,
                    success=execution.success,
                    result=execution.results,
                    error=str(execution.code_error) if execution.code_error else (execution.execution_error if not execution.execution_ok else None),
                    execution_time=execution.duration if execution.duration else execution_time,
                    logs=execution.stdout,
                )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Run error hook if defined
            if skill.metadata.hooks and skill.metadata.hooks.on_error:
                try:
                    import subprocess
                    os.environ["SKILL_ERROR"] = str(e)
                    subprocess.run(
                        skill.metadata.hooks.on_error,
                        shell=True,
                        check=True,
                    )
                except Exception:
                    pass
            
            logger.exception(f"Skill execution failed: {skill.name}")
            return SkillExecution(
                skill_id=skill_id,
                success=False,
                error=str(e),
                execution_time=execution_time,
            )
    
    # =========================================================================
    # Skill Versioning
    # =========================================================================
    
    def create_version(self, skill_id: str, version: str) -> Optional[SkillVersion]:
        """Create a new version of a skill.
        
        Args:
            skill_id: The skill's ID.
            version: Version string (e.g., "1.1.0").
            
        Returns:
            SkillVersion if created, None if skill not found.
        """
        skill = self.retrieve(skill_id)
        if not skill:
            return None
        
        version_id = str(uuid.uuid4())
        
        skill_version = SkillVersion(
            version_id=version_id,
            skill_id=skill_id,
            version=version,
            content=skill.to_skill_md(),
            created_at=datetime.now(),
            is_current=True,
        )
        
        # Mark previous versions as not current
        if skill_id in self._versions:
            for v in self._versions[skill_id]:
                v.is_current = False
        else:
            self._versions[skill_id] = []
        
        self._versions[skill_id].append(skill_version)
        
        # Save version to disk
        version_file = self.skills_path / "versions" / f"{skill_id}_{version}.skill.md"
        version_file.parent.mkdir(parents=True, exist_ok=True)
        version_file.write_text(skill_version.content)
        
        logger.info(f"Created version {version} for skill {skill_id}")
        return skill_version
    
    def list_versions(self, skill_id: str) -> list[SkillVersion]:
        """List all versions of a skill.
        
        Args:
            skill_id: The skill's ID.
            
        Returns:
            List of versions.
        """
        return self._versions.get(skill_id, [])
    
    # =========================================================================
    # Private Helpers
    # =========================================================================
    
    def _load_skill_md(self, path: Path) -> Skill:
        """Load a skill from a SKILL.md file."""
        content = path.read_text()
        skill = Skill.from_skill_md(content)
        skill.skill_id = skill.skill_id or str(uuid.uuid4())
        return skill
    
    def _load_skill_json(self, path: Path) -> Skill:
        """Load a skill from a JSON file."""
        data = json.loads(path.read_text())
        
        metadata = SkillMetadata(
            name=data.get("name", "Unnamed"),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            allowed_tools=data.get("allowed_tools", []),
            tags=data.get("tags", []),
        )
        
        return Skill(
            metadata=metadata,
            content=data.get("content", ""),
            python_code=data.get("python_code"),
            skill_id=data.get("skill_id"),
        )
    
    def _save_skill_to_disk(self, skill: Skill) -> None:
        """Save a skill to disk as SKILL.md."""
        skill_id = skill.skill_id or skill.name
        skill_file = self.skills_path / f"{skill_id}.skill.md"
        skill_file.write_text(skill.to_skill_md())
    
    def _load_skill_from_disk(self, skill_id: str) -> Optional[Skill]:
        """Load a skill from disk by ID."""
        skill_file = self.skills_path / f"{skill_id}.skill.md"
        if skill_file.exists():
            return self._load_skill_md(skill_file)
        return None
