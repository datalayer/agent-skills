# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Unit tests for agent-skills package."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_skills.files import SkillFile, SkillDirectory, setup_skills_directory
from agent_skills.codegen import generate_skill_file, generate_skill_from_template
from agent_skills.types import (
    Skill,
    SkillContext,
    SkillHooks,
    SkillMetadata,
    SkillSearchResult,
    SkillStatus,
)


# =============================================================================
# SkillFile Tests
# =============================================================================

class TestSkillFile:
    """Tests for SkillFile class."""

    def test_from_file_basic(self, tmp_path: Path):
        """Test creating SkillFile from a simple Python file."""
        skill_code = '''"""Test skill for processing data."""

async def process_data(input_path: str) -> dict:
    """Process data from a file."""
    return {"result": "success"}
'''
        skill_file = tmp_path / "process_data.py"
        skill_file.write_text(skill_code)

        skill = SkillFile.from_file(skill_file, tmp_path)

        assert skill.name == "process_data"
        assert skill.path == skill_file
        assert "Test skill" in skill.description
        assert "process_data" in skill.functions

    def test_from_file_multiple_functions(self, tmp_path: Path):
        """Test extracting multiple async functions."""
        skill_code = '''"""Multi-function skill."""

async def func_a():
    pass

async def func_b():
    pass

def sync_func():
    pass

async def _private():
    pass
'''
        skill_file = tmp_path / "multi.py"
        skill_file.write_text(skill_code)

        skill = SkillFile.from_file(skill_file, tmp_path)

        assert "func_a" in skill.functions
        assert "func_b" in skill.functions
        assert "sync_func" not in skill.functions  # Not async
        assert "_private" not in skill.functions  # Private

    def test_from_file_no_docstring(self, tmp_path: Path):
        """Test file without docstring."""
        skill_code = '''
async def no_doc():
    pass
'''
        skill_file = tmp_path / "no_doc.py"
        skill_file.write_text(skill_code)

        skill = SkillFile.from_file(skill_file, tmp_path)

        assert skill.description == ""
        assert skill.functions == ["no_doc"]

    def test_from_file_syntax_error(self, tmp_path: Path):
        """Test handling of syntax errors."""
        skill_file = tmp_path / "invalid.py"
        skill_file.write_text("this is not valid python {{{{")

        skill = SkillFile.from_file(skill_file, tmp_path)

        assert skill.name == "invalid"
        assert skill.functions == []
        assert skill.description == ""

    def test_module_name_calculation(self, tmp_path: Path):
        """Test module name calculation for nested skills."""
        # Create nested structure
        subdir = tmp_path / "subpackage"
        subdir.mkdir()
        skill_file = subdir / "nested_skill.py"
        skill_file.write_text('async def nested(): pass')

        skill = SkillFile.from_file(skill_file, tmp_path)

        assert "skills" in skill.module_name
        assert "nested_skill" in skill.module_name

    def test_load_module(self, tmp_path: Path):
        """Test loading skill as a module."""
        skill_code = '''
TEST_VALUE = 42

async def get_value():
    return TEST_VALUE
'''
        skill_file = tmp_path / "loadable.py"
        skill_file.write_text(skill_code)

        skill = SkillFile.from_file(skill_file, tmp_path)
        module = skill.load_module()

        assert hasattr(module, "TEST_VALUE")
        assert module.TEST_VALUE == 42
        assert hasattr(module, "get_value")

    def test_get_function_by_name(self, tmp_path: Path):
        """Test getting a specific function by name."""
        skill_code = '''
async def func_a():
    return "a"

async def func_b():
    return "b"
'''
        skill_file = tmp_path / "multi_func.py"
        skill_file.write_text(skill_code)

        skill = SkillFile.from_file(skill_file, tmp_path)
        func = skill.get_function("func_b")

        assert func is not None
        assert asyncio.iscoroutinefunction(func)

    def test_get_function_default(self, tmp_path: Path):
        """Test getting default function (matching skill name)."""
        skill_code = '''
async def my_skill():
    return "default"
'''
        skill_file = tmp_path / "my_skill.py"
        skill_file.write_text(skill_code)

        skill = SkillFile.from_file(skill_file, tmp_path)
        func = skill.get_function()

        assert func is not None


# =============================================================================
# SkillDirectory Tests
# =============================================================================

class TestSkillDirectory:
    """Tests for SkillDirectory class."""

    def test_init_creates_directory(self, tmp_path: Path):
        """Test that initialization creates the directory."""
        skills_path = tmp_path / "new_skills"
        assert not skills_path.exists()

        skills = SkillDirectory(str(skills_path))

        assert skills_path.exists()
        assert (skills_path / "__init__.py").exists()
        assert (skills_path / "README.md").exists()

    def test_list_empty(self, tmp_path: Path):
        """Test listing empty skills directory."""
        skills = SkillDirectory(str(tmp_path / "empty_skills"))
        result = skills.list()
        assert result == []

    def test_list_skills(self, tmp_path: Path):
        """Test listing skills."""
        skills = SkillDirectory(str(tmp_path))

        # Create some skill files
        (tmp_path / "skill_a.py").write_text('async def skill_a(): pass')
        (tmp_path / "skill_b.py").write_text('async def skill_b(): pass')

        result = skills.list()

        assert len(result) == 2
        names = [s.name for s in result]
        assert "skill_a" in names
        assert "skill_b" in names

    def test_list_ignores_private_files(self, tmp_path: Path):
        """Test that private files are ignored."""
        skills = SkillDirectory(str(tmp_path))

        (tmp_path / "public_skill.py").write_text('async def public(): pass')
        (tmp_path / "_private.py").write_text('async def private(): pass')

        result = skills.list()

        assert len(result) == 1
        assert result[0].name == "public_skill"

    def test_get_existing_skill(self, tmp_path: Path):
        """Test getting an existing skill."""
        skills = SkillDirectory(str(tmp_path))
        (tmp_path / "my_skill.py").write_text('"""Desc"""\nasync def my_skill(): pass')

        skill = skills.get("my_skill")

        assert skill is not None
        assert skill.name == "my_skill"

    def test_get_nonexistent_skill(self, tmp_path: Path):
        """Test getting a non-existent skill."""
        skills = SkillDirectory(str(tmp_path))
        skill = skills.get("does_not_exist")
        assert skill is None

    def test_search_skills(self, tmp_path: Path):
        """Test searching for skills."""
        skills = SkillDirectory(str(tmp_path))

        (tmp_path / "csv_analyzer.py").write_text('"""Analyze CSV files"""\nasync def analyze(): pass')
        (tmp_path / "json_parser.py").write_text('"""Parse JSON"""\nasync def parse(): pass')
        (tmp_path / "data_processor.py").write_text('"""Process data"""\nasync def process(): pass')

        results = skills.search("CSV")

        assert len(results) >= 1
        assert any(s.name == "csv_analyzer" for s in results)

    def test_create_skill(self, tmp_path: Path):
        """Test creating a new skill."""
        skills = SkillDirectory(str(tmp_path))

        skill = skills.create(
            name="new_skill",
            code='async def new_skill(x: str) -> str:\n    return x.upper()',
            description="Convert text to uppercase",
        )

        assert skill.name == "new_skill"
        assert skill.path.exists()
        assert "new_skill" in skill.functions

        # Verify file content
        content = skill.path.read_text()
        assert "Convert text to uppercase" in content
        assert "async def new_skill" in content

    def test_delete_skill(self, tmp_path: Path):
        """Test deleting a skill."""
        skills = SkillDirectory(str(tmp_path))
        (tmp_path / "to_delete.py").write_text('async def to_delete(): pass')

        assert (tmp_path / "to_delete.py").exists()
        result = skills.delete("to_delete")

        assert result is True
        assert not (tmp_path / "to_delete.py").exists()

    def test_delete_nonexistent_skill(self, tmp_path: Path):
        """Test deleting a non-existent skill."""
        skills = SkillDirectory(str(tmp_path))
        result = skills.delete("does_not_exist")
        assert result is False

    def test_add_to_sys_path(self, tmp_path: Path):
        """Test adding skills to sys.path."""
        import sys
        skills = SkillDirectory(str(tmp_path / "test_skills"))
        
        original_path = sys.path.copy()
        skills.add_to_sys_path()

        try:
            assert str(tmp_path) in sys.path
        finally:
            sys.path = original_path


# =============================================================================
# setup_skills_directory Tests
# =============================================================================

class TestSetupSkillsDirectory:
    """Tests for setup_skills_directory function."""

    def test_setup_creates_directory(self, tmp_path: Path):
        """Test that setup creates and configures the directory."""
        import sys
        original_path = sys.path.copy()

        try:
            skills = setup_skills_directory(str(tmp_path / "setup_test"))

            assert isinstance(skills, SkillDirectory)
            assert skills.path.exists()
        finally:
            sys.path = original_path


# =============================================================================
# Code Generation Tests
# =============================================================================

class TestCodeGeneration:
    """Tests for skill code generation."""

    def test_generate_skill_file_basic(self, tmp_path: Path):
        """Test generating a basic skill file."""
        path = generate_skill_file(
            name="test_skill",
            description="A test skill",
            code="return {'status': 'ok'}",
            output_dir=tmp_path,
        )

        assert path.exists()
        content = path.read_text()
        assert "test_skill" in content
        assert "A test skill" in content
        assert "return {'status': 'ok'}" in content

    def test_generate_skill_file_with_parameters(self, tmp_path: Path):
        """Test generating skill with parameters."""
        path = generate_skill_file(
            name="param_skill",
            description="Skill with parameters",
            code="return input_path",
            parameters=[
                {"name": "input_path", "type": "str", "description": "Input path", "required": True},
                {"name": "limit", "type": "int", "description": "Limit", "required": False, "default": 10},
            ],
            output_dir=tmp_path,
        )

        content = path.read_text()
        assert "input_path: str" in content
        assert "limit: int" in content

    def test_generate_skill_from_template(self, tmp_path: Path):
        """Test generating skill from template."""
        path = generate_skill_from_template(
            name="template_skill",
            template="wait_for_condition",
            output_dir=tmp_path,
        )

        assert path.exists()
        content = path.read_text()
        assert "template_skill" in content or "wait" in content.lower()


# =============================================================================
# Models Tests
# =============================================================================

class TestModels:
    """Tests for data models."""

    def test_skill_status_enum(self):
        """Test SkillStatus enum values."""
        assert SkillStatus.DRAFT.value == "draft"
        assert SkillStatus.ACTIVE.value == "active"
        assert SkillStatus.ARCHIVED.value == "archived"

    def test_skill_context_enum(self):
        """Test SkillContext enum values."""
        assert SkillContext.FORK.value == "fork"
        assert SkillContext.INLINE.value == "inline"
        assert SkillContext.SANDBOX.value == "sandbox"

    def test_skill_hooks(self):
        """Test SkillHooks dataclass."""
        hooks = SkillHooks(
            before_invoke="echo 'before'",
            after_invoke="echo 'after'",
        )
        assert hooks.before_invoke == "echo 'before'"
        assert hooks.after_invoke == "echo 'after'"
        assert hooks.on_error is None

    def test_skill_metadata(self):
        """Test SkillMetadata dataclass."""
        metadata = SkillMetadata(
            name="test_skill",
            description="A test skill",
            version="2.0.0",
            tags=["test", "example"],
            allowed_tools=["filesystem__read"],
        )

        assert metadata.name == "test_skill"
        assert metadata.version == "2.0.0"
        assert "test" in metadata.tags
        assert metadata.context == SkillContext.FORK  # Default

    def test_skill_dataclass(self):
        """Test Skill dataclass."""
        metadata = SkillMetadata(name="full_skill", description="Full skill")
        skill = Skill(
            metadata=metadata,
            content="# Full Skill\n\nInstructions...",
            python_code="print('hello')",
        )

        assert skill.metadata.name == "full_skill"
        assert "Instructions" in skill.content
        assert skill.python_code is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for agent-skills."""

    @pytest.mark.asyncio
    async def test_create_and_execute_skill(self, tmp_path: Path):
        """Test creating and executing a skill end-to-end."""
        skills = SkillDirectory(str(tmp_path))

        # Create a simple skill
        skill = skills.create(
            name="double_value",
            code='''
async def double_value(x: int) -> int:
    return x * 2
''',
            description="Double a value",
        )

        # Get and execute the skill
        loaded_skill = skills.get("double_value")
        assert loaded_skill is not None

        func = loaded_skill.get_function()
        result = await func(21)

        assert result == 42

    @pytest.mark.asyncio
    async def test_skill_composition(self, tmp_path: Path):
        """Test that skills can import other skills."""
        import sys
        skills_path = tmp_path / "compose_skills"
        skills = SkillDirectory(str(skills_path))
        skills.add_to_sys_path()

        # Create first skill
        skills.create(
            name="add_one",
            code='''
async def add_one(x: int) -> int:
    return x + 1
''',
        )

        # Create second skill that imports the first
        skills.create(
            name="add_two",
            code=f'''
import sys
sys.path.insert(0, "{skills_path.parent}")

async def add_two(x: int) -> int:
    from compose_skills.add_one import add_one
    result = await add_one(x)
    result = await add_one(result)
    return result
''',
        )

        # Test composition
        skill = skills.get("add_two")
        func = skill.get_function()
        result = await func(40)

        assert result == 42
