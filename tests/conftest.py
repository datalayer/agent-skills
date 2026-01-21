# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Pytest configuration and fixtures for agent-skills tests."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """Create a temporary skills directory."""
    skills_path = tmp_path / "skills"
    skills_path.mkdir()
    (skills_path / "__init__.py").write_text('"""Test skills."""\n')
    return skills_path


@pytest.fixture
def sample_skill_file(skills_dir: Path) -> Path:
    """Create a sample skill file."""
    skill_code = '''"""Sample skill for testing."""

async def sample_skill(input_value: str) -> dict:
    """Process input value.
    
    Args:
        input_value: The value to process.
    
    Returns:
        Processed result.
    """
    return {
        "input": input_value,
        "output": input_value.upper(),
        "length": len(input_value),
    }


if __name__ == "__main__":
    import asyncio
    import sys
    result = asyncio.run(sample_skill(sys.argv[1]))
    print(result)
'''
    skill_file = skills_dir / "sample_skill.py"
    skill_file.write_text(skill_code)
    return skill_file
