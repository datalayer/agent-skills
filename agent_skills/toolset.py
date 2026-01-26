# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Datalayer Skills Toolset - Integration with pydantic-ai SkillsToolset.

This module provides extensions to pydantic-ai's SkillsToolset with:
- SandboxExecutor: Execute skill scripts in isolated code-sandboxes
- DatalayerSkillsToolset: Extended toolset with Datalayer-specific features

Example:
    from pydantic_ai import Agent
    from agent_skills import DatalayerSkillsToolset, SandboxExecutor
    from code_sandboxes import LocalEvalSandbox
    
    # Create executor with sandbox
    sandbox = LocalEvalSandbox()
    executor = SandboxExecutor(sandbox)
    
    # Create toolset
    skills_toolset = DatalayerSkillsToolset(
        directories=["./skills"],
        executor=executor,
    )
    
    # Use with pydantic-ai agent
    agent = Agent(
        model='openai:gpt-4o',
        toolsets=[skills_toolset],
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from code_sandboxes import ExecutionResult

if TYPE_CHECKING:
    from code_sandboxes import LocalEvalSandbox
    from pydantic_ai._run_context import RunContext

logger = logging.getLogger(__name__)


# =============================================================================
# Executor Protocols (aligned with pydantic-ai PR #3780)
# =============================================================================


@runtime_checkable
class SkillScriptExecutorProtocol(Protocol):
    """Protocol for skill script execution.
    
    This aligns with pydantic-ai's SkillScriptExecutor pattern from PR #3780,
    allowing pluggable execution environments.
    """
    
    async def execute(
        self,
        skill_name: str,
        script_name: str,
        script_path: Path,
        args: list[str],
        timeout: int | None = None,
    ) -> str:
        """Execute a skill script.
        
        Args:
            skill_name: Name of the skill.
            script_name: Name of the script within the skill.
            script_path: Path to the script file.
            args: Command-line arguments.
            timeout: Execution timeout in seconds.
            
        Returns:
            Script output as string.
        """
        ...


# =============================================================================
# Sandbox Executor
# =============================================================================


@dataclass
class SandboxExecutor:
    """Execute skill scripts in an isolated code sandbox.
    
    Uses code-sandboxes (LocalEvalSandbox or remote) to execute
    skill scripts safely with proper isolation.
    
    Example:
        from code_sandboxes import LocalEvalSandbox
        from agent_skills import SandboxExecutor
        
        sandbox = LocalEvalSandbox()
        executor = SandboxExecutor(sandbox)
        
        result = await executor.execute(
            skill_name="pdf-extractor",
            script_name="extract",
            script_path=Path("./skills/pdf-extractor/scripts/extract.py"),
            args=["--input", "document.pdf"],
        )
    """
    
    sandbox: Any  # LocalEvalSandbox or compatible
    default_timeout: int = 30
    
    async def execute(
        self,
        skill_name: str,
        script_name: str,
        script_path: Path,
        args: list[str],
        timeout: int | None = None,
    ) -> str:
        """Execute a skill script in the sandbox.
        
        Args:
            skill_name: Name of the skill.
            script_name: Name of the script.
            script_path: Path to the script file.
            args: Command-line arguments.
            timeout: Execution timeout (uses default if None).
            
        Returns:
            Script output as string.
            
        Raises:
            TimeoutError: If execution exceeds timeout.
            RuntimeError: If execution fails.
        """
        timeout = timeout or self.default_timeout
        
        logger.debug(
            f"Executing skill script: {skill_name}/{script_name} "
            f"with args={args}, timeout={timeout}s"
        )
        
        # Read the script content
        script_content = script_path.read_text()
        
        # Build execution code that mimics CLI invocation
        execution_code = self._build_execution_code(
            script_content=script_content,
            script_name=script_name,
            args=args,
        )
        
        # Get identity environment variables from request context
        identity_env: dict[str, str] | None = None
        try:
            from agent_runtimes.context.identities import get_identity_env
            identity_env = get_identity_env()
            if identity_env:
                logger.debug(f"SandboxExecutor: Using identity env vars: {list(identity_env.keys())}")
        except ImportError:
            # agent_runtimes not installed, skip identity context
            pass
        
        try:
            # Execute in sandbox - prefer run_code if available (supports envs)
            if hasattr(self.sandbox, 'run_code'):
                # Use run_code which supports envs parameter
                result: ExecutionResult = self.sandbox.run_code(execution_code, envs=identity_env)
                
                # Check for execution failure (infrastructure error)
                if not result.execution_ok:
                    raise RuntimeError(f"Sandbox execution failed: {result.execution_error or 'Unknown error'}")
                
                # Check for code error (user code exception)
                if result.code_error:
                    # Return error details so the agent can handle/fix it
                    return str(result.code_error)

                # Extract output from ExecutionResult object
                if result.logs and result.logs.stdout:
                    return result.stdout
                elif result.results:
                    return '\n'.join(str(r.data) for r in result.results)
                else:
                    return ""
            elif asyncio.iscoroutinefunction(getattr(self.sandbox, 'execute', None)):
                result = await asyncio.wait_for(
                    self.sandbox.execute(execution_code),
                    timeout=timeout,
                )
            else:
                # Sync sandbox - run in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, self.sandbox.execute, execution_code),
                    timeout=timeout,
                )
            
            # Extract output
            if hasattr(result, 'stdout'):
                return result.stdout or ""
            elif hasattr(result, 'result'):
                return str(result.result)
            else:
                return str(result)
                
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Skill script {skill_name}/{script_name} timed out after {timeout}s"
            )
        except Exception as e:
            raise RuntimeError(
                f"Skill script {skill_name}/{script_name} failed: {e}"
            ) from e
    
    def _build_execution_code(
        self,
        script_content: str,
        script_name: str,
        args: list[str],
    ) -> str:
        """Build executable code that runs the script with arguments.
        
        This wraps the script to:
        1. Set sys.argv with the provided arguments
        2. Execute the script
        3. Capture and return output
        """
        # Escape the script content for embedding
        escaped_script = script_content.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        args_json = json.dumps(args)
        
        return f'''
import sys
import json
import io
from contextlib import redirect_stdout, redirect_stderr

# Set up sys.argv as if running from CLI
sys.argv = ["{script_name}.py"] + {args_json}

# Capture output
_stdout = io.StringIO()
_stderr = io.StringIO()

_script = """
{escaped_script}
"""

try:
    with redirect_stdout(_stdout), redirect_stderr(_stderr):
        exec(compile(_script, "{script_name}.py", "exec"))
    
    # Return captured output
    output = _stdout.getvalue()
    if not output:
        output = _stderr.getvalue()
    print(output)
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    raise
'''


# =============================================================================
# Local Python Executor (fallback)
# =============================================================================


@dataclass
class LocalPythonExecutor:
    """Execute skill scripts using local Python subprocess.
    
    This is a simpler executor that runs scripts via subprocess.
    Use SandboxExecutor for better isolation.
    
    Args:
        python_executable: Path to Python executable (defaults to "python").
        default_timeout: Default execution timeout in seconds.
        env: Environment variables to pass to the subprocess.
              These are merged with the current environment.
        use_identity_context: Whether to include identity tokens from the
              request context as environment variables (default: True).
    """
    
    python_executable: str | Path | None = None
    default_timeout: int = 30
    env: dict[str, str] | None = None
    use_identity_context: bool = True
    
    async def execute(
        self,
        skill_name: str,
        script_name: str,
        script_path: Path,
        args: list[str],
        timeout: int | None = None,
    ) -> str:
        """Execute a skill script using subprocess."""
        import os
        import subprocess
        
        timeout = timeout or self.default_timeout
        python = self.python_executable or "python"
        
        cmd = [str(python), str(script_path)] + args
        
        # Merge environment variables with current environment
        exec_env = os.environ.copy()
        if self.env:
            exec_env.update(self.env)
        
        # Also merge identity environment variables from request context
        if self.use_identity_context:
            try:
                from agent_runtimes.context.identities import get_identity_env
                identity_env = get_identity_env()
                if identity_env:
                    exec_env.update(identity_env)
                    logger.debug(f"Added identity env vars: {list(identity_env.keys())}")
            except ImportError:
                # agent_runtimes not installed, skip identity context
                pass
        
        logger.debug(f"Executing: {' '.join(cmd)}")
        if self.env:
            logger.debug(f"With env vars: {list(self.env.keys())}")
        
        try:
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=exec_env,
                ),
                timeout=timeout,
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise RuntimeError(
                    f"Script failed with code {result.returncode}: {stderr.decode()}"
                )
            
            return stdout.decode()
            
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Skill script {skill_name}/{script_name} timed out after {timeout}s"
            )


# =============================================================================
# Callable Executor (for programmatic skills)
# =============================================================================


@dataclass
class CallableExecutor:
    """Execute skill scripts that are Python callables.
    
    For programmatic skills created with decorators:
    
        skill = Skill(name="my-skill", ...)
        
        @skill.script
        async def process(ctx: RunContext, data: str) -> str:
            return f"Processed: {data}"
    """
    
    default_timeout: int = 30
    
    async def execute_callable(
        self,
        func: Callable,
        ctx: Any,
        args: list[str],
        timeout: int | None = None,
    ) -> str:
        """Execute a callable skill script.
        
        Args:
            func: The async callable to execute.
            ctx: RunContext for dependency injection.
            args: Arguments to pass (parsed as needed).
            timeout: Execution timeout.
            
        Returns:
            Result as string.
        """
        timeout = timeout or self.default_timeout
        
        # Determine if function takes context
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        try:
            if asyncio.iscoroutinefunction(func):
                if params and params[0] in ('ctx', 'context', 'run_context'):
                    result = await asyncio.wait_for(
                        func(ctx, *args),
                        timeout=timeout,
                    )
                else:
                    result = await asyncio.wait_for(
                        func(*args),
                        timeout=timeout,
                    )
            else:
                # Sync function
                loop = asyncio.get_event_loop()
                if params and params[0] in ('ctx', 'context', 'run_context'):
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, func, ctx, *args),
                        timeout=timeout,
                    )
                else:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, func, *args),
                        timeout=timeout,
                    )
            
            if isinstance(result, str):
                return result
            return json.dumps(result)
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Callable script timed out after {timeout}s")


# =============================================================================
# Datalayer Skills Toolset
# =============================================================================


@dataclass
class DatalayerSkillResource:
    """A resource file associated with a skill.
    
    Aligned with pydantic-ai's SkillResource from PR #3780.
    """
    name: str
    content: str | None = None
    path: Path | None = None
    
    async def read(self) -> str:
        """Read the resource content."""
        if self.content is not None:
            return self.content
        if self.path is not None:
            return self.path.read_text()
        return ""


@dataclass
class DatalayerSkillScript:
    """A script that can be executed as part of a skill.
    
    Aligned with pydantic-ai's SkillScript from PR #3780.
    """
    name: str
    path: Path | None = None
    callable: Callable | None = None
    description: str = ""
    
    def is_callable(self) -> bool:
        """Check if this is a callable (programmatic) script."""
        return self.callable is not None


@dataclass
class DatalayerSkill:
    """A skill definition compatible with pydantic-ai's Skill.
    
    This extends the pydantic-ai Skill model with Datalayer-specific features
    while maintaining full compatibility.
    
    Example:
        # From filesystem (SKILL.md)
        skill = DatalayerSkill.from_skill_md(path)
        
        # Programmatic
        skill = DatalayerSkill(
            name="my-skill",
            description="Does something useful",
            content="Instructions for the skill...",
        )
        
        @skill.script
        async def run(ctx, input: str) -> str:
            return f"Result: {input}"
    """
    name: str
    description: str
    content: str = ""
    path: Path | None = None
    resources: list[DatalayerSkillResource] = field(default_factory=list)
    scripts: list[DatalayerSkillScript] = field(default_factory=list)
    
    # Datalayer-specific fields
    tags: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    
    def resource(self, func: Callable) -> Callable:
        """Decorator to add a callable resource to the skill.
        
        Example:
            @skill.resource
            def get_context() -> str:
                return "Dynamic context..."
        """
        name = func.__name__
        # For callable resources, we store them differently
        # This creates a resource that will be evaluated on read
        self.resources.append(DatalayerSkillResource(
            name=name,
            content=None,  # Will be populated dynamically
            path=None,
        ))
        return func
    
    def script(self, func: Callable) -> Callable:
        """Decorator to add a callable script to the skill.
        
        Example:
            @skill.script
            async def process(ctx: RunContext, data: str) -> str:
                result = await ctx.deps.db.query(data)
                return str(result)
        """
        name = func.__name__
        self.scripts.append(DatalayerSkillScript(
            name=name,
            path=None,
            callable=func,
            description=func.__doc__ or "",
        ))
        return func
    
    @classmethod
    def from_skill_md(cls, skill_path: Path) -> "DatalayerSkill":
        """Load a skill from a SKILL.md file.
        
        Parses YAML frontmatter and discovers resources/scripts.
        
        Args:
            skill_path: Path to SKILL.md file or skill directory.
            
        Returns:
            Loaded skill.
        """
        import yaml
        
        if skill_path.is_dir():
            skill_md = skill_path / "SKILL.md"
        else:
            skill_md = skill_path
            skill_path = skill_md.parent
        
        if not skill_md.exists():
            raise FileNotFoundError(f"SKILL.md not found: {skill_md}")
        
        content = skill_md.read_text()
        
        # Parse YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                yaml_content = parts[1]
                markdown_content = parts[2].strip()
                try:
                    metadata = yaml.safe_load(yaml_content) or {}
                except yaml.YAMLError:
                    metadata = {}
            else:
                metadata = {}
                markdown_content = content
        else:
            metadata = {}
            markdown_content = content
        
        # Extract metadata
        name = metadata.get("name")
        description = metadata.get("description")

        if not name or not description:
            raise ValueError("Invalid SKILL.md format: name and description are required")
        version = metadata.get("version", "1.0.0")
        tags = metadata.get("tags", [])
        author = metadata.get("author")
        license_name = metadata.get("license")
        compatibility = metadata.get("compatibility")
        metadata_map = metadata.get("metadata", {})

        allowed_tools = metadata.get("allowed-tools", [])
        if isinstance(allowed_tools, str):
            allowed_tools = [t for t in allowed_tools.split(" ") if t]

        denied_tools = metadata.get("denied-tools", [])
        if isinstance(denied_tools, str):
            denied_tools = [t for t in denied_tools.split(" ") if t]
        
        # Discover resources
        resources = []
        for resource_dir_name in ["resources", "references", "assets"]:
            resources_dir = skill_path / resource_dir_name
            if resources_dir.exists():
                for res_file in resources_dir.iterdir():
                    if res_file.is_file():
                        resources.append(DatalayerSkillResource(
                            name=f"{resource_dir_name}/{res_file.name}",
                            path=res_file,
                        ))
        
        # Also check for common resource files in skill root
        for res_name in ["REFERENCE.md", "FORMS.md", "TEMPLATES.md"]:
            res_path = skill_path / res_name
            if res_path.exists():
                resources.append(DatalayerSkillResource(
                    name=res_name,
                    path=res_path,
                ))
        
        # Discover scripts
        scripts = []
        scripts_dir = skill_path / "scripts"
        if scripts_dir.exists():
            for script_file in scripts_dir.glob("*.py"):
                scripts.append(DatalayerSkillScript(
                    name=script_file.stem,
                    path=script_file,
                    description=f"Script: {script_file.name}",
                ))
        
        return cls(
            name=name,
            description=description,
            content=markdown_content,
            path=skill_path,
            resources=resources,
            scripts=scripts,
            tags=tags,
            version=version,
            author=author,
            allowed_tools=allowed_tools,
            denied_tools=denied_tools,
            license=license_name,
            compatibility=compatibility,
            metadata=metadata_map or {},
        )
    
    def get_skills_header(self) -> str:
        """Get a brief header for system prompt injection.
        
        Used for progressive disclosure - only the header is shown initially.
        """
        location = str(self.path) if self.path else "programmatic"
        safe_name = self.name.replace('"', "'")
        safe_description = self.description.replace('"', "'")
        safe_location = location.replace('"', "'")
        return (
            f"<skill name=\"{safe_name}\" "
            f"description=\"{safe_description}\" "
            f"location=\"{safe_location}\" />"
        )
    
    def get_full_content(self) -> str:
        """Get full skill content for load_skill().
        
        Returns the complete SKILL.md content plus resource/script listings.
        """
        lines = [
            f"# Skill: {self.name}",
            f"**Description:** {self.description}",
            f"**Path:** {self.path or 'programmatic'}",
        ]

        if self.allowed_tools:
            lines.append(f"**Allowed Tools:** {', '.join(self.allowed_tools)}")
        if self.denied_tools:
            lines.append(f"**Denied Tools:** {', '.join(self.denied_tools)}")
        if self.license:
            lines.append(f"**License:** {self.license}")
        if self.compatibility:
            lines.append(f"**Compatibility:** {self.compatibility}")
        if self.metadata:
            lines.append(f"**Metadata:** {self.metadata}")
        
        if self.resources:
            lines.append("**Available Resources:**")
            for res in self.resources:
                lines.append(f"- {res.name}")
        else:
            lines.append("**Available Resources:** None")
        
        if self.scripts:
            lines.append("**Available Scripts:**")
            for script in self.scripts:
                lines.append(f"- {script.name}")
        else:
            lines.append("**Available Scripts:** None")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(self.content)
        
        return "\n".join(lines)


# =============================================================================
# Skills Toolset for Pydantic-AI
# =============================================================================


try:
    from pydantic_ai.toolsets import AbstractToolset
    from pydantic_ai.toolsets.abstract import ToolsetTool
    from pydantic_ai.tools import ToolDefinition
    from pydantic_ai._run_context import RunContext
    from pydantic_core import SchemaValidator, core_schema
    
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    AbstractToolset = object  # Fallback for type hints


if PYDANTIC_AI_AVAILABLE:
    
    # Schema validator for any args
    SKILL_ARGS_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())
    
    @dataclass
    class DatalayerSkillsToolset(AbstractToolset):
        """Skills toolset for pydantic-ai with Datalayer extensions.
        
        Provides the standard skills tools:
        - list_skills(): List available skills
        - load_skill(skill_name): Load full skill content
        - read_skill_resource(skill_name, resource_name): Read a resource
        - run_skill_script(skill_name, script_name, args): Execute a script
        
        With Datalayer-specific features:
        - SandboxExecutor for isolated script execution
        - Support for programmatic skills (decorators)
        - Integration with code-sandboxes
        
        Example:
            from agent_skills import DatalayerSkillsToolset, SandboxExecutor
            from code_sandboxes import LocalEvalSandbox
            from pydantic_ai import Agent
            
            # With sandbox execution
            sandbox = LocalEvalSandbox()
            toolset = DatalayerSkillsToolset(
                directories=["./skills"],
                executor=SandboxExecutor(sandbox),
            )
            
            agent = Agent(
                model='openai:gpt-4o',
                toolsets=[toolset],
            )
        """
        
        directories: list[str | Path] = field(default_factory=list)
        skills: list[DatalayerSkill] = field(default_factory=list)
        executor: SandboxExecutor | LocalPythonExecutor | None = None
        script_timeout: int = 30
        _id: str | None = None
        
        # Internal state
        _discovered_skills: dict[str, DatalayerSkill] = field(
            default_factory=dict, repr=False
        )
        _initialized: bool = field(default=False, repr=False)
        _skills_call_count: int = field(default=0, repr=False)
        
        def __post_init__(self):
            # Add programmatic skills
            for skill in self.skills:
                self._discovered_skills[skill.name] = skill
        
        @property
        def id(self) -> str | None:
            return self._id
        
        @property
        def label(self) -> str:
            return "Datalayer Skills Toolset"
        
        async def _ensure_initialized(self) -> None:
            """Discover skills from directories if not already done."""
            if self._initialized:
                return
            
            for directory in self.directories:
                dir_path = Path(directory)
                if not dir_path.exists():
                    logger.warning(f"Skills directory not found: {dir_path}")
                    continue
                
                # Look for SKILL.md files
                for skill_md in dir_path.rglob("SKILL.md"):
                    try:
                        skill = DatalayerSkill.from_skill_md(skill_md)
                        if skill.name not in self._discovered_skills:
                            self._discovered_skills[skill.name] = skill
                            logger.debug(f"Discovered skill: {skill.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load skill from {skill_md}: {e}")
            
            self._initialized = True
            logger.info(f"Discovered {len(self._discovered_skills)} skills")
        
        async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]:
            """Get the tools provided by this toolset."""
            await self._ensure_initialized()
            
            tools = {}
            
            # list_skills tool
            tools["list_skills"] = ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name="list_skills",
                    description="List all available skills with their names and descriptions.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
                max_retries=0,
                args_validator=SKILL_ARGS_VALIDATOR,
            )
            
            # load_skill tool
            tools["load_skill"] = ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name="load_skill",
                    description="Load the full content and instructions for a skill.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "skill_name": {
                                "type": "string",
                                "description": "Name of the skill to load",
                            },
                        },
                        "required": ["skill_name"],
                    },
                ),
                max_retries=1,
                args_validator=SKILL_ARGS_VALIDATOR,
            )
            
            # read_skill_resource tool
            tools["read_skill_resource"] = ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name="read_skill_resource",
                    description="Read a resource file from a skill.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "skill_name": {
                                "type": "string",
                                "description": "Name of the skill",
                            },
                            "resource_name": {
                                "type": "string",
                                "description": "Name of the resource to read",
                            },
                        },
                        "required": ["skill_name", "resource_name"],
                    },
                ),
                max_retries=1,
                args_validator=SKILL_ARGS_VALIDATOR,
            )
            
            # run_skill_script tool
            tools["run_skill_script"] = ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name="run_skill_script",
                    description="Execute a script from a skill with arguments.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "skill_name": {
                                "type": "string",
                                "description": "Name of the skill",
                            },
                            "script_name": {
                                "type": "string",
                                "description": "Name of the script to run",
                            },
                            "args": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Arguments to pass to the script",
                                "default": [],
                            },
                        },
                        "required": ["skill_name", "script_name"],
                    },
                ),
                max_retries=1,
                args_validator=SKILL_ARGS_VALIDATOR,
            )
            
            return tools
        
        async def call_tool(
            self,
            name: str,
            tool_args: dict[str, Any],
            ctx: RunContext,
            tool: ToolsetTool,
        ) -> Any:
            """Call a tool by name."""
            await self._ensure_initialized()
            self._skills_call_count += 1
            
            if name == "list_skills":
                return self._list_skills()
            elif name == "load_skill":
                return self._load_skill(tool_args.get("skill_name", ""))
            elif name == "read_skill_resource":
                return await self._read_skill_resource(
                    tool_args.get("skill_name", ""),
                    tool_args.get("resource_name", ""),
                )
            elif name == "run_skill_script":
                return await self._run_skill_script(
                    tool_args.get("skill_name", ""),
                    tool_args.get("script_name", ""),
                    tool_args.get("args", []),
                    ctx,
                )
            else:
                raise ValueError(f"Unknown tool: {name}")
        
        def get_call_counts(self) -> dict[str, int]:
            """Return counts for skills tool calls."""
            return {
                "skills_tool_calls": self._skills_call_count,
            }
        
        def _list_skills(self) -> str:
            """List all available skills."""
            if not self._discovered_skills:
                return "No skills available."
            
            lines = ["Available skills:"]
            for skill in self._discovered_skills.values():
                lines.append(skill.get_skills_header())
            
            return "\n".join(lines)
        
        def _load_skill(self, skill_name: str) -> str:
            """Load full skill content."""
            skill = self._discovered_skills.get(skill_name)
            if not skill:
                return f"Skill not found: {skill_name}"
            
            return skill.get_full_content()
        
        async def _read_skill_resource(
            self,
            skill_name: str,
            resource_name: str,
        ) -> str:
            """Read a skill resource."""
            skill = self._discovered_skills.get(skill_name)
            if not skill:
                return f"Skill not found: {skill_name}"
            
            for resource in skill.resources:
                if resource.name == resource_name:
                    return await resource.read()
            
            return f"Resource not found: {resource_name}"
        
        async def _run_skill_script(
            self,
            skill_name: str,
            script_name: str,
            args: list[str],
            ctx: RunContext,
        ) -> str:
            """Run a skill script."""
            skill = self._discovered_skills.get(skill_name)
            if not skill:
                return f"Skill not found: {skill_name}"
            
            # Find the script
            script = None
            for s in skill.scripts:
                if s.name == script_name:
                    script = s
                    break
            
            if not script:
                return f"Script not found: {script_name}"
            
            # Execute
            try:
                if script.is_callable():
                    # Programmatic script
                    callable_executor = CallableExecutor(
                        default_timeout=self.script_timeout
                    )
                    return await callable_executor.execute_callable(
                        script.callable,
                        ctx,
                        args,
                        self.script_timeout,
                    )
                elif self.executor and script.path:
                    # File-based script with executor
                    return await self.executor.execute(
                        skill_name=skill_name,
                        script_name=script_name,
                        script_path=script.path,
                        args=args,
                        timeout=self.script_timeout,
                    )
                else:
                    return f"No executor configured for script: {script_name}"
            except TimeoutError as e:
                return f"Script timed out: {e}"
            except Exception as e:
                return f"Script failed: {e}"
        
        async def get_instructions(self, ctx: RunContext | None = None) -> str:
            """Get instructions for system prompt injection.
            
            This implements the get_instructions() method proposed in PR #3780
            for automatic system prompt integration.
            """
            if not self._discovered_skills:
                return ""
            
            lines = [
                "<skills>",
                "You have access to skills that extend your capabilities.",
                "Skills are modular packages with instructions, resources, and scripts.",
                "",
                "<available_skills>",
            ]
            
            for skill in self._discovered_skills.values():
                lines.append(skill.get_skills_header())
            
            lines.extend([
                "</available_skills>",
                "",
                "<usage>",
                "1. Use load_skill(skill_name) to read full instructions",
                "2. Use read_skill_resource(skill_name, resource) for docs",
                "3. Use run_skill_script(skill_name, script, args) to execute",
                "</usage>",
                "</skills>",
            ])
            
            return "\n".join(lines)


else:
    # Fallback when pydantic-ai is not available
    class DatalayerSkillsToolset:  # type: ignore
        """Placeholder when pydantic-ai is not installed."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "pydantic-ai with skills support is required. "
                "Install with: pip install pydantic-ai @ git+https://github.com/DougTrajano/pydantic-ai.git@DEV-1099"
            )
