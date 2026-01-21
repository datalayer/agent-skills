# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Data models for Agent Skills.

Based on Anthropic's Claude Code Skills format with YAML frontmatter
and the Anthropic SDK Skills API structure.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from datetime import datetime


class SkillStatus(Enum):
    """Status of a skill."""
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class SkillContext(Enum):
    """Execution context for skills."""
    FORK = "fork"          # Run in isolated fork (default)
    INLINE = "inline"      # Run in current context
    SANDBOX = "sandbox"    # Run in code sandbox


@dataclass
class SkillHooks:
    """Lifecycle hooks for skills.
    
    Based on Claude Code Skills hook system.
    """
    before_invoke: Optional[str] = None   # Shell command to run before
    after_invoke: Optional[str] = None    # Shell command to run after
    on_error: Optional[str] = None        # Shell command on error


@dataclass
class SkillMetadata:
    """Metadata for a skill.
    
    Based on SKILL.md YAML frontmatter format from Claude Code.
    """
    name: str
    description: str
    version: str = "1.0.0"
    
    # Tool access control
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)
    
    # Model configuration
    model: Optional[str] = None
    
    # Execution context
    context: SkillContext = SkillContext.FORK
    
    # Visibility
    user_invocable: bool = True
    
    # Lifecycle hooks
    hooks: Optional[SkillHooks] = None
    
    # Tags for categorization
    tags: list[str] = field(default_factory=list)
    
    # Author info
    author: Optional[str] = None

    # Optional spec fields
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Skill:
    """A reusable agent skill.
    
    Skills are code-based tool compositions that can be:
    - Discovered by agents
    - Activated when relevant
    - Executed with parameters
    - Saved and versioned
    
    Based on:
    - Claude Code SKILL.md format
    - Anthropic SDK Skills API
    """
    metadata: SkillMetadata
    content: str  # Markdown content with instructions
    
    # Optional code implementations
    python_code: Optional[str] = None
    typescript_code: Optional[str] = None
    
    # Status
    status: SkillStatus = SkillStatus.ACTIVE
    
    # Unique ID (for API compatibility)
    skill_id: Optional[str] = None
    
    @property
    def name(self) -> str:
        return self.metadata.name
    
    @property
    def description(self) -> str:
        return self.metadata.description
    
    @classmethod
    def from_skill_md(cls, content: str) -> "Skill":
        """Parse a SKILL.md file into a Skill object.
        
        Args:
            content: Full content of a SKILL.md file.
            
        Returns:
            Parsed Skill object.
        """
        import re
        
        # Extract YAML frontmatter
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.DOTALL)
        
        if not frontmatter_match:
            raise ValueError("Invalid SKILL.md format: missing YAML frontmatter")
        
        yaml_content = frontmatter_match.group(1)
        markdown_content = frontmatter_match.group(2)
        
        # Parse YAML (simple parser for common fields)
        metadata_dict = cls._parse_yaml(yaml_content)
        
        # Build metadata
        hooks = None
        if "hooks" in metadata_dict:
            hooks_dict = metadata_dict["hooks"]
            hooks = SkillHooks(
                before_invoke=hooks_dict.get("before-invoke"),
                after_invoke=hooks_dict.get("after-invoke"),
                on_error=hooks_dict.get("on-error"),
            )
        
        context = SkillContext.FORK
        if "context" in metadata_dict:
            context = SkillContext(metadata_dict["context"])
        
        name = metadata_dict.get("name", "")
        description = metadata_dict.get("description", "")

        if not name or not description:
            raise ValueError("Invalid SKILL.md format: name and description are required")

        allowed_tools = metadata_dict.get("allowed-tools", [])
        if isinstance(allowed_tools, str):
            allowed_tools = [t for t in allowed_tools.split(" ") if t]

        denied_tools = metadata_dict.get("denied-tools", [])
        if isinstance(denied_tools, str):
            denied_tools = [t for t in denied_tools.split(" ") if t]

        metadata = SkillMetadata(
            name=name,
            description=description,
            version=metadata_dict.get("version", "1.0.0"),
            allowed_tools=allowed_tools,
            denied_tools=denied_tools,
            model=metadata_dict.get("model"),
            context=context,
            user_invocable=metadata_dict.get("user-invocable", True),
            hooks=hooks,
            tags=metadata_dict.get("tags", []),
            author=metadata_dict.get("author"),
            license=metadata_dict.get("license"),
            compatibility=metadata_dict.get("compatibility"),
            metadata=metadata_dict.get("metadata", {}),
        )
        
        # Extract code blocks from markdown
        python_code = cls._extract_code_block(markdown_content, "python")
        typescript_code = cls._extract_code_block(markdown_content, "typescript")
        
        return cls(
            metadata=metadata,
            content=markdown_content,
            python_code=python_code,
            typescript_code=typescript_code,
        )
    
    def to_skill_md(self) -> str:
        """Convert skill to SKILL.md format.
        
        Returns:
            SKILL.md formatted string.
        """
        lines = ["---"]
        lines.append(f"name: {self.metadata.name}")
        lines.append(f"description: {self.metadata.description}")
        lines.append(f"version: {self.metadata.version}")
        
        if self.metadata.allowed_tools:
            lines.append(f"allowed-tools: {' '.join(self.metadata.allowed_tools)}")
        
        if self.metadata.denied_tools:
            lines.append(f"denied-tools: {' '.join(self.metadata.denied_tools)}")
        
        if self.metadata.model:
            lines.append(f"model: {self.metadata.model}")
        
        lines.append(f"context: {self.metadata.context.value}")
        lines.append(f"user-invocable: {str(self.metadata.user_invocable).lower()}")
        
        if self.metadata.hooks:
            lines.append("hooks:")
            if self.metadata.hooks.before_invoke:
                lines.append(f"  before-invoke: {self.metadata.hooks.before_invoke}")
            if self.metadata.hooks.after_invoke:
                lines.append(f"  after-invoke: {self.metadata.hooks.after_invoke}")
            if self.metadata.hooks.on_error:
                lines.append(f"  on-error: {self.metadata.hooks.on_error}")
        
        if self.metadata.tags:
            lines.append("tags:")
            for tag in self.metadata.tags:
                lines.append(f"  - {tag}")
        
        if self.metadata.author:
            lines.append(f"author: {self.metadata.author}")

        if self.metadata.license:
            lines.append(f"license: {self.metadata.license}")

        if self.metadata.compatibility:
            lines.append(f"compatibility: {self.metadata.compatibility}")

        if self.metadata.metadata:
            lines.append("metadata:")
            for key, value in self.metadata.metadata.items():
                lines.append(f"  {key}: {value}")
        
        lines.append("---")
        lines.append("")
        lines.append(self.content)
        
        return "\n".join(lines)
    
    @staticmethod
    def _parse_yaml(content: str) -> dict[str, Any]:
        """Simple YAML parser for skill frontmatter."""
        result: dict[str, Any] = {}
        current_key = None
        current_list: list[str] = []
        current_dict: dict[str, str] = {}
        in_list = False
        in_dict = False
        
        for line in content.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            
            # Check for list item
            if stripped.startswith("- "):
                if in_list and current_key:
                    current_list.append(stripped[2:].strip())
                continue
            
            # Check for dict item (indented key: value)
            if line.startswith("  ") and ":" in stripped and not stripped.endswith(":"):
                if in_dict and current_key:
                    key, value = stripped.split(":", 1)
                    current_dict[key.strip()] = value.strip()
                continue
            
            # Save previous list/dict
            if in_list and current_key:
                result[current_key] = current_list
                current_list = []
                in_list = False
            if in_dict and current_key:
                result[current_key] = current_dict
                current_dict = {}
                in_dict = False
            
            # Parse key: value
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = value.strip()
                
                if value == "":
                    # Start of list or dict
                    current_key = key
                    in_list = True
                    in_dict = True
                elif value.lower() in ("true", "false"):
                    result[key] = value.lower() == "true"
                else:
                    result[key] = value
        
        # Save final list/dict
        if in_list and current_key and current_list:
            result[current_key] = current_list
        if in_dict and current_key and current_dict:
            result[current_key] = current_dict
        
        return result
    
    @staticmethod
    def _extract_code_block(content: str, language: str) -> Optional[str]:
        """Extract a code block for a specific language."""
        import re
        pattern = rf'```{language}\s*\n(.*?)\n```'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1) if match else None


@dataclass
class SkillVersion:
    """A version of a skill.
    
    Supports skill versioning as in Anthropic SDK.
    """
    version_id: str
    skill_id: str
    version: str
    content: str
    created_at: datetime
    is_current: bool = False


@dataclass
class SkillExecution:
    """Result of executing a skill."""
    skill_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    logs: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SkillSearchResult:
    """Result of searching for skills."""
    skills: list[Skill]
    total: int
    query: str
