# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The bundled skills load from their modules, each with its SKILL.md and its scripts."""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

from agent_skills.toolset import AgentSkill

#: The framework's own example skill. The catalogue's skills (accounting, crawl, events, github,
#: image, pdf, text-summarizer, video) live in datalayer-skills since 1.4.0.
BUNDLED = {
    "jokes": ["tell_joke"],
}

MOVED = ["accounting", "crawl", "events", "github", "image", "pdf", "text_summarizer", "video"]


@pytest.mark.parametrize("name", sorted(BUNDLED))
def test_the_skill_loads_from_its_module_with_every_script(name: str) -> None:
    skill = AgentSkill.from_module(f"agent_skills.skills.{name}")
    assert skill.name == name
    assert skill.description
    scripts_dir = (
        Path(skill.path).parent / "scripts"
        if Path(skill.path).is_file()
        else Path(skill.path) / "scripts"
    )
    found = sorted(path.stem for path in scripts_dir.glob("*.py"))
    assert found == sorted(BUNDLED[name])
    for path in scripts_dir.glob("*.py"):
        py_compile.compile(str(path), doraise=True)


def test_the_skill_md_names_every_script() -> None:
    for name, scripts in BUNDLED.items():
        text = (
            Path(__file__).parent.parent / "agent_skills" / "skills" / name / "SKILL.md"
        ).read_text()
        for script in scripts:
            assert f"`script_name: {script}`" in text, (name, script)


@pytest.mark.parametrize("name", MOVED)
def test_a_moved_skill_is_gone_with_no_alias(name: str) -> None:
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"agent_skills.skills.{name}")


def test_the_bundled_skill_is_public_and_needs_no_secret() -> None:
    """`metadata.visibility: public` and no secret: what a company may publish to people's
    personal agents."""
    import re

    from agent_skills.types import Skill

    text = (
        Path(__file__).parents[1] / "agent_skills" / "skills" / "jokes" / "SKILL.md"
    ).read_text()
    skill = Skill.from_skill_md(text)
    section = re.search(
        r"^## Required Environment Variables\s*$(.*?)(?=^## |\Z)", text, re.M | re.S
    )
    assert skill.metadata.metadata.get("visibility") == "public"
    assert not (section and re.search(r"`[A-Z][A-Z0-9_]+`", section.group(1)))


def test_no_extra_names_a_moved_skill() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert "skills-" not in pyproject
    assert "agent_skills[" not in pyproject
