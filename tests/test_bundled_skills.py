# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The bundled skills load from their modules, each with its SKILL.md and its scripts."""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

from agent_skills.toolset import AgentSkill

BUNDLED = {
    "image": [
        "image_info",
        "resize_image",
        "crop_image",
        "convert_image",
        "optimize_image",
        "thumbnail_image",
        "snapshot_page",
        "snapshot_video_frame",
    ],
    "video": [
        "video_info",
        "record_page",
        "trim_video",
        "transcode_video",
        "optimize_video",
        "video_thumbnail",
        "video_to_gif",
        "concat_videos",
    ],
}


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
