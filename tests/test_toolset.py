# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Tests for pydantic-ai SkillsToolset integration."""

import pytest
from pathlib import Path

from code_sandboxes import CodeError, ExecutionResult, Logs, OutputMessage

from agent_skills import (
    DatalayerSkillsToolset,
    DatalayerSkill,
    DatalayerSkillResource,
    DatalayerSkillScript,
    SandboxExecutor,
    LocalPythonExecutor,
    CallableExecutor,
    PYDANTIC_AI_AVAILABLE,
)


# =============================================================================
# Tests for DatalayerSkill
# =============================================================================


class TestDatalayerSkill:
    """Tests for DatalayerSkill class."""
    
    def test_create_skill(self):
        """Test creating a skill programmatically."""
        skill = DatalayerSkill(
            name="test-skill",
            description="A test skill",
            content="Instructions for the skill...",
        )
        
        assert skill.name == "test-skill"
        assert skill.description == "A test skill"
        assert "Instructions" in skill.content
    
    def test_skill_decorator_script(self):
        """Test adding a script via decorator."""
        skill = DatalayerSkill(
            name="decorated-skill",
            description="Skill with decorated script",
        )
        
        @skill.script
        async def process(data: str) -> str:
            """Process the data."""
            return f"Processed: {data}"
        
        assert len(skill.scripts) == 1
        assert skill.scripts[0].name == "process"
        assert skill.scripts[0].is_callable()
    
    def test_skill_decorator_resource(self):
        """Test adding a resource via decorator."""
        skill = DatalayerSkill(
            name="resource-skill",
            description="Skill with resource",
        )
        
        @skill.resource
        def get_reference() -> str:
            return "Reference content..."
        
        assert len(skill.resources) == 1
        assert skill.resources[0].name == "get_reference"
    
    def test_skill_get_skills_header(self):
        """Test getting skill header for system prompt."""
        skill = DatalayerSkill(
            name="my-skill",
            description="Does something useful",
        )
        
        header = skill.get_skills_header()
        
        assert "my-skill" in header
        assert "Does something useful" in header
    
    def test_skill_get_full_content(self):
        """Test getting full skill content."""
        skill = DatalayerSkill(
            name="full-skill",
            description="A complete skill",
            content="Detailed instructions here.",
            resources=[DatalayerSkillResource(name="ref.md")],
            scripts=[DatalayerSkillScript(name="run")],
            allowed_tools=["filesystem__read_file"],
            denied_tools=["network__fetch"],
        )
        
        content = skill.get_full_content()
        
        assert "# Skill: full-skill" in content
        assert "Available Resources:" in content
        assert "ref.md" in content
        assert "Available Scripts:" in content
        assert "run" in content
        assert "Allowed Tools:" in content
        assert "Denied Tools:" in content
        assert "Detailed instructions here." in content


class TestDatalayerSkillFromFile:
    """Tests for loading skills from SKILL.md files."""
    
    @pytest.fixture
    def skill_dir(self, tmp_path: Path) -> Path:
        """Create a skill directory with SKILL.md."""
        skill_path = tmp_path / "test-skill"
        skill_path.mkdir()
        
        skill_md = skill_path / "SKILL.md"
        skill_md.write_text("""---
name: test-skill
description: A test skill from file
version: "2.0.0"
allowed-tools: filesystem__read_file bash__execute
denied-tools: network__fetch
license: Apache-2.0
compatibility: Requires git and access to the internet
metadata:
    author: TestOrg
    version: "2.0"
tags:
  - testing
  - example
author: TestUser
---

# Test Skill

This is the skill content with instructions.

## Usage

Use this skill for testing purposes.
""")
        
        # Create resources
        resources_dir = skill_path / "resources"
        resources_dir.mkdir()
        (resources_dir / "reference.md").write_text("Reference content")
        
        # Create scripts
        scripts_dir = skill_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "process.py").write_text("""
def process(data):
    return f"Processed: {data}"
""")
        
        return skill_path
    
    def test_load_skill_from_skill_md(self, skill_dir: Path):
        """Test loading a skill from SKILL.md."""
        skill = DatalayerSkill.from_skill_md(skill_dir)
        
        assert skill.name == "test-skill"
        assert skill.description == "A test skill from file"
        assert skill.version == "2.0.0"
        assert "testing" in skill.tags
        assert skill.author == "TestUser"
        assert "filesystem__read_file" in skill.allowed_tools
        assert "network__fetch" in skill.denied_tools
        assert skill.license == "Apache-2.0"
        assert "internet" in skill.compatibility
        assert skill.metadata.get("author") == "TestOrg"
        assert "Test Skill" in skill.content
    
    def test_load_skill_discovers_resources(self, skill_dir: Path):
        """Test that resources are discovered."""
        skill = DatalayerSkill.from_skill_md(skill_dir)
        
        assert len(skill.resources) >= 1
        resource_names = [r.name for r in skill.resources]
        assert "resources/reference.md" in resource_names
    
    def test_load_skill_discovers_scripts(self, skill_dir: Path):
        """Test that scripts are discovered."""
        skill = DatalayerSkill.from_skill_md(skill_dir)
        
        assert len(skill.scripts) >= 1
        script_names = [s.name for s in skill.scripts]
        assert "process" in script_names


# =============================================================================
# Tests for Executors
# =============================================================================


class TestCallableExecutor:
    """Tests for CallableExecutor."""
    
    @pytest.mark.asyncio
    async def test_execute_async_callable(self):
        """Test executing an async callable."""
        executor = CallableExecutor(default_timeout=10)
        
        async def my_func(arg1: str) -> str:
            return f"Result: {arg1}"
        
        result = await executor.execute_callable(
            func=my_func,
            ctx=None,
            args=["test-input"],
        )
        
        assert result == "Result: test-input"
    
    @pytest.mark.asyncio
    async def test_execute_sync_callable(self):
        """Test executing a sync callable."""
        executor = CallableExecutor(default_timeout=10)
        
        def my_func(arg1: str) -> str:
            return f"Sync: {arg1}"
        
        result = await executor.execute_callable(
            func=my_func,
            ctx=None,
            args=["value"],
        )
        
        assert result == "Sync: value"


class TestLocalPythonExecutor:
    """Tests for LocalPythonExecutor."""
    
    @pytest.fixture
    def script_file(self, tmp_path: Path) -> Path:
        """Create a simple test script."""
        script = tmp_path / "test_script.py"
        script.write_text("""
import sys
print(f"Hello from script! Args: {sys.argv[1:]}")
""")
        return script
    
    @pytest.mark.asyncio
    async def test_execute_script(self, script_file: Path):
        """Test executing a Python script."""
        executor = LocalPythonExecutor(default_timeout=10)
        
        result = await executor.execute(
            skill_name="test",
            script_name="test_script",
            script_path=script_file,
            args=["arg1", "arg2"],
        )

        assert result["success"] is True
        assert "Hello from script" in result["output"]
        assert "arg1" in result["output"]


# =============================================================================
# Tests for DatalayerSkillsToolset (requires pydantic-ai)
# =============================================================================


@pytest.mark.skipif(
    not PYDANTIC_AI_AVAILABLE,
    reason="pydantic-ai with skills support not installed"
)
class TestDatalayerSkillsToolset:
    """Tests for DatalayerSkillsToolset."""
    
    @pytest.fixture
    def skills_directory(self, tmp_path: Path) -> Path:
        """Create a skills directory with sample skills."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        
        # Create first skill
        skill1_dir = skills_dir / "skill-one"
        skill1_dir.mkdir()
        (skill1_dir / "SKILL.md").write_text("""---
name: skill-one
description: First test skill
---

# Skill One

Instructions for skill one.
""")
        
        # Create second skill
        skill2_dir = skills_dir / "skill-two"
        skill2_dir.mkdir()
        (skill2_dir / "SKILL.md").write_text("""---
name: skill-two
description: Second test skill
---

# Skill Two

Instructions for skill two.
""")
        
        return skills_dir
    
    @pytest.mark.asyncio
    async def test_toolset_initialization(self, skills_directory: Path):
        """Test toolset initializes and discovers skills."""
        toolset = DatalayerSkillsToolset(
            directories=[str(skills_directory)],
        )
        
        await toolset._ensure_initialized()
        
        assert len(toolset._discovered_skills) == 2
        assert "skill-one" in toolset._discovered_skills
        assert "skill-two" in toolset._discovered_skills
    
    @pytest.mark.asyncio
    async def test_list_skills(self, skills_directory: Path):
        """Test list_skills tool."""
        toolset = DatalayerSkillsToolset(
            directories=[str(skills_directory)],
        )
        
        await toolset._ensure_initialized()
        result = toolset._list_skills()
        
        assert "Available skills:" in result
        assert "<skill" in result
        assert "skill-one" in result
        assert "skill-two" in result
    
    @pytest.mark.asyncio
    async def test_load_skill(self, skills_directory: Path):
        """Test load_skill tool."""
        toolset = DatalayerSkillsToolset(
            directories=[str(skills_directory)],
        )
        
        await toolset._ensure_initialized()
        result = toolset._load_skill("skill-one")
        
        assert "Skill: skill-one" in result
        assert "First test skill" in result
        assert "Skill One" in result
    
    @pytest.mark.asyncio
    async def test_load_skill_not_found(self, skills_directory: Path):
        """Test load_skill with non-existent skill."""
        toolset = DatalayerSkillsToolset(
            directories=[str(skills_directory)],
        )
        
        await toolset._ensure_initialized()
        result = toolset._load_skill("nonexistent")
        
        assert "not found" in result.lower()
    
    @pytest.mark.asyncio
    async def test_toolset_with_programmatic_skills(self):
        """Test toolset with programmatically defined skills."""
        skill = DatalayerSkill(
            name="programmatic-skill",
            description="A skill defined in code",
            content="Use this skill for testing.",
        )
        
        @skill.script
        async def run(data: str) -> str:
            return f"Executed: {data}"
        
        toolset = DatalayerSkillsToolset(
            skills=[skill],
        )
        
        await toolset._ensure_initialized()
        
        assert "programmatic-skill" in toolset._discovered_skills
        result = toolset._list_skills()
        assert "programmatic-skill" in result
        assert "location=\"programmatic\"" in result
    
    @pytest.mark.asyncio
    async def test_get_instructions(self):
        """Test get_instructions for system prompt."""
        skill = DatalayerSkill(
            name="test-skill",
            description="A test skill",
        )
        
        toolset = DatalayerSkillsToolset(skills=[skill])
        toolset._discovered_skills["test-skill"] = skill
        
        instructions = await toolset.get_instructions()
        
        assert "<skills>" in instructions
        assert "<available_skills>" in instructions
        assert "test-skill" in instructions
        assert "location=\"programmatic\"" in instructions
        assert "load_skill" in instructions




# =============================================================================


    

    class TestSandboxExecutor:
        """Tests for SandboxExecutor using ExecutionResult fields."""

        class DummySandbox:
            def __init__(self, result: ExecutionResult):
                self.result = result
                self.last_code: str | None = None
                self.last_envs: dict[str, str] | None = None

            def run_code(self, code: str, envs: dict[str, str] | None = None) -> ExecutionResult:
                self.last_code = code
                self.last_envs = envs
                return self.result

        @pytest.mark.asyncio
        async def test_executor_success_uses_stdout(self, tmp_path: Path):
            script_path = tmp_path / "script.py"
            script_path.write_text("print('hello')")

            result = ExecutionResult(
                logs=Logs(stdout=[OutputMessage(line="ok", timestamp=0.0, error=False)])
            )
            sandbox = self.DummySandbox(result)
            executor = SandboxExecutor(sandbox)

            output = await executor.execute(
                skill_name="test-skill",
                script_name="run",
                script_path=script_path,
                args=["--flag"],
            )

            assert output["success"] is True
            assert output["output"] == "ok"

        @pytest.mark.asyncio
        async def test_executor_returns_code_error(self, tmp_path: Path):
            script_path = tmp_path / "script.py"
            script_path.write_text("raise ValueError('bad')")

            result = ExecutionResult(
                code_error=CodeError(name="ValueError", value="bad", traceback=""),
            )
            executor = SandboxExecutor(self.DummySandbox(result))

            output = await executor.execute(
                skill_name="test-skill",
                script_name="run",
                script_path=script_path,
                args=[],
            )

            assert output["success"] is False
            assert output["code_error"] is not None
            assert output["code_error"]["name"] == "ValueError"
            assert output["code_error"]["value"] == "bad"

        @pytest.mark.asyncio
        async def test_executor_raises_on_execution_failure(self, tmp_path: Path):
            script_path = tmp_path / "script.py"
            script_path.write_text("print('x')")

            result = ExecutionResult(
                execution_ok=False,
                execution_error="sandbox unavailable",
            )
            executor = SandboxExecutor(self.DummySandbox(result))

            output = await executor.execute(
                skill_name="test-skill",
                script_name="run",
                script_path=script_path,
                args=[],
            )

            assert output["success"] is False
            assert output["execution_ok"] is False
            assert output["execution_error"] == "sandbox unavailable"
# Integration Tests
# =============================================================================


@pytest.mark.skipif(
    not PYDANTIC_AI_AVAILABLE,
    reason="pydantic-ai with skills support not installed"
)
class TestPydanticAIIntegration:
    """Integration tests with pydantic-ai Agent."""
    
    def test_toolset_import(self):
        """Test that toolset can be imported with pydantic-ai."""
        from pydantic_ai import Agent
        from agent_skills import DatalayerSkillsToolset
        
        # Should not raise
        toolset = DatalayerSkillsToolset()
        assert toolset is not None
    
    @pytest.mark.asyncio
    async def test_toolset_provides_tools(self, tmp_path: Path):
        """Test that toolset provides the expected tools."""
        from pydantic_ai._run_context import RunContext
        
        # Create a skill directory
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: Test skill
---
Content here.
""")
        
        toolset = DatalayerSkillsToolset(
            directories=[str(tmp_path / "skills")],
        )
        
        # Mock minimal RunContext
        class MockCtx:
            deps = None
        
        tools = await toolset.get_tools(MockCtx())  # type: ignore
        
        tool_names = set(tools.keys())
        assert "list_skills" in tool_names
        assert "load_skill" in tool_names
        assert "read_skill_resource" in tool_names
        assert "run_skill_script" in tool_names
