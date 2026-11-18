# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""A skill says the environment it reads, and is activated only when it has what it needs."""

from pathlib import Path

import pytest

import agent_skills.toolset as toolset_module
from agent_skills import (
    PYDANTIC_AI_AVAILABLE,
    AgentSkill,
    AgentSkillsToolset,
    parse_skill_env,
)


def write_skill(root: Path, name: str, frontmatter: str = "", body: str = "") -> Path:
    path = root / name
    (path / "scripts").mkdir(parents=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: The {name} skill\n{frontmatter}---\n\n# {name}\n\n{body}"
    )
    (path / "scripts" / "run.py").write_text("print('ran')\n")
    return path


ANY_OF = """env:
  optional:
    - PHOTOS_TIMEOUT
  any_of:
    - FIRST_PHOTOS_KEY
    - SECOND_PHOTOS_KEY
"""

REQUIRED = """env:
  required:
    - SERVICE_TOKEN
    - SERVICE_URL
"""

SECTION = "## Required Environment Variables\n\n- `LEGACY_TOKEN` (required)\n"


class TestDeclaration:
    def test_env_block_lists(self):
        declared = parse_skill_env(
            {"required": ["A_URL"], "optional": "B_LIMIT, C_DIR", "any-of": ["D_KEY", "E_KEY"]}
        )
        assert declared == {
            "required": ["A_URL"],
            "optional": ["B_LIMIT", "C_DIR"],
            "any_of": ["D_KEY", "E_KEY"],
        }

    def test_env_block_ignores_what_is_no_variable_name(self):
        assert parse_skill_env({"required": ["OK_NAME", "not a name", "", 3]}) == {
            "required": ["OK_NAME"]
        }
        assert parse_skill_env(None) == {}
        assert parse_skill_env(["A"]) == {}

    def test_all_of_is_required(self):
        assert parse_skill_env({"all_of": ["A", "B"]}) == {"required": ["A", "B"]}

    def test_from_skill_md(self, tmp_path: Path):
        skill = AgentSkill.from_skill_md(write_skill(tmp_path, "photos", ANY_OF))
        assert skill.get_required_env_vars() == []
        assert skill.get_any_of_env_vars() == ["FIRST_PHOTOS_KEY", "SECOND_PHOTOS_KEY"]
        assert skill.get_optional_env_vars() == [
            "FIRST_PHOTOS_KEY",
            "PHOTOS_TIMEOUT",
            "SECOND_PHOTOS_KEY",
        ]

    def test_env_block_wins_over_the_sections(self, tmp_path: Path):
        skill = AgentSkill.from_skill_md(write_skill(tmp_path, "photos", ANY_OF, SECTION))
        assert skill.get_required_env_vars() == []

    def test_sections_without_an_env_block(self, tmp_path: Path):
        body = SECTION + "\n## Optional Environment Variables\n\n- `LEGACY_LIMIT`: a limit.\n"
        skill = AgentSkill.from_skill_md(write_skill(tmp_path, "legacy", body=body))
        assert skill.get_required_env_vars() == ["LEGACY_TOKEN"]
        assert skill.get_optional_env_vars() == ["LEGACY_LIMIT"]
        assert skill.get_any_of_env_vars() == []


class TestActivation:
    def test_nothing_declared_is_always_activated(self, tmp_path: Path):
        skill = AgentSkill.from_skill_md(write_skill(tmp_path, "plain"))
        assert skill.is_activated({})
        assert skill.get_inactive_reason({}) is None

    def test_required_needs_every_one(self, tmp_path: Path):
        skill = AgentSkill.from_skill_md(write_skill(tmp_path, "service", REQUIRED))
        assert not skill.is_activated({})
        assert not skill.is_activated({"SERVICE_TOKEN": "t"})
        assert skill.is_activated({"SERVICE_TOKEN": "t", "SERVICE_URL": "u"})
        assert skill.get_missing_env_vars({"SERVICE_TOKEN": "t"}) == ["SERVICE_URL"]

    def test_an_empty_or_blank_value_is_not_set(self, tmp_path: Path):
        skill = AgentSkill.from_skill_md(write_skill(tmp_path, "service", REQUIRED))
        assert not skill.is_activated({"SERVICE_TOKEN": "", "SERVICE_URL": "u"})
        assert not skill.is_activated({"SERVICE_TOKEN": "  ", "SERVICE_URL": "u"})

    def test_any_of_needs_one(self, tmp_path: Path):
        skill = AgentSkill.from_skill_md(write_skill(tmp_path, "photos", ANY_OF))
        assert not skill.is_activated({})
        # An optional variable that is no key does not activate it.
        assert not skill.is_activated({"PHOTOS_TIMEOUT": "5"})
        assert skill.is_activated({"SECOND_PHOTOS_KEY": "k"})
        assert skill.get_set_env_vars({"SECOND_PHOTOS_KEY": "k"}) == ["SECOND_PHOTOS_KEY"]

    def test_required_and_any_of_together(self, tmp_path: Path):
        both = "env:\n  required: [BASE_URL]\n  any_of: [A_KEY, B_KEY]\n"
        skill = AgentSkill.from_skill_md(write_skill(tmp_path, "both", both))
        assert not skill.is_activated({"A_KEY": "k"})
        assert not skill.is_activated({"BASE_URL": "u"})
        assert skill.is_activated({"BASE_URL": "u", "B_KEY": "k"})

    def test_the_reason_names_variables_and_no_value(self, tmp_path: Path):
        skill = AgentSkill.from_skill_md(write_skill(tmp_path, "photos", ANY_OF))
        assert skill.get_inactive_reason({}) == "none of FIRST_PHOTOS_KEY, SECOND_PHOTOS_KEY set"
        service = AgentSkill.from_skill_md(write_skill(tmp_path, "service", REQUIRED))
        reason = service.get_inactive_reason({"SERVICE_TOKEN": "s3cr3t"})
        assert reason == "SERVICE_URL not set"

    def test_the_process_environment_when_unsaid(self, tmp_path: Path, monkeypatch):
        skill = AgentSkill.from_skill_md(write_skill(tmp_path, "photos", ANY_OF))
        monkeypatch.delenv("FIRST_PHOTOS_KEY", raising=False)
        monkeypatch.delenv("SECOND_PHOTOS_KEY", raising=False)
        assert not skill.is_activated()
        monkeypatch.setenv("FIRST_PHOTOS_KEY", "k")
        assert skill.is_activated()

    def test_summary_and_content_say_which_optional_are_set(self, tmp_path: Path):
        skill = AgentSkill.from_skill_md(write_skill(tmp_path, "photos", ANY_OF))
        env = {"SECOND_PHOTOS_KEY": "the-value"}
        assert "optional_env_set=SECOND_PHOTOS_KEY" in skill.get_discovery_summary(env)
        content = skill.get_full_content(env)
        assert "**Environment set now:** SECOND_PHOTOS_KEY" in content
        assert "**Environment not set:** FIRST_PHOTOS_KEY, PHOTOS_TIMEOUT" in content
        assert "the-value" not in content + skill.get_skills_header(env)


class Ctx:
    deps = None


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return {"success": True, "output": "ran", "stdout": "ran", "stderr": "", "exit_code": 0}


@pytest.mark.skipif(not PYDANTIC_AI_AVAILABLE, reason="pydantic-ai not installed")
class TestToolsetActivation:
    @pytest.fixture
    def directory(self, tmp_path: Path) -> Path:
        write_skill(tmp_path, "plain")
        write_skill(tmp_path, "photos", ANY_OF)
        write_skill(tmp_path, "service", REQUIRED)
        write_skill(tmp_path, "legacy", body=SECTION)
        return tmp_path

    async def toolset(self, directory: Path, env: dict, **kwargs) -> AgentSkillsToolset:
        toolset = AgentSkillsToolset(directories=[directory], env=env, **kwargs)
        await toolset._ensure_initialized()
        return toolset

    @pytest.mark.asyncio
    async def test_a_skill_without_its_environment_is_not_listed(self, directory: Path):
        toolset = await self.toolset(directory, {})
        listed = toolset._list_skills()
        instructions = await toolset.get_instructions()
        for text in (listed, instructions):
            assert 'name="plain"' in text
            for hidden in ("photos", "service", "legacy"):
                assert hidden not in text
        assert toolset.get_inactive_skills() == {
            "photos": "none of FIRST_PHOTOS_KEY, SECOND_PHOTOS_KEY set",
            "service": "SERVICE_TOKEN, SERVICE_URL not set",
            "legacy": "LEGACY_TOKEN not set",
        }

    @pytest.mark.asyncio
    async def test_it_cannot_be_loaded_or_run(self, directory: Path):
        executor = RecordingExecutor()
        toolset = await self.toolset(directory, {}, executor=executor)
        loaded = toolset._load_skill("photos")
        assert loaded == "Skill not found: photos. Available skills: plain"
        result = await toolset._run_skill_script("photos", "run", [], {}, Ctx())
        assert result["success"] is False
        assert "Skill not found: photos" in result["error"]
        assert executor.calls == []

    @pytest.mark.asyncio
    async def test_one_key_activates_it(self, directory: Path):
        toolset = await self.toolset(directory, {"FIRST_PHOTOS_KEY": "k", "LEGACY_TOKEN": "t"})
        listed = toolset._list_skills()
        assert 'name="photos"' in listed and 'name="legacy"' in listed
        assert "service" not in listed
        assert "optional_env_set=FIRST_PHOTOS_KEY" in listed
        assert "**Environment set now:** FIRST_PHOTOS_KEY" in toolset._load_skill("photos")

    @pytest.mark.asyncio
    async def test_the_environment_is_read_at_each_call(self, directory: Path, monkeypatch):
        for name in ("FIRST_PHOTOS_KEY", "SECOND_PHOTOS_KEY"):
            monkeypatch.delenv(name, raising=False)
        toolset = AgentSkillsToolset(directories=[directory])
        await toolset._ensure_initialized()
        assert "photos" not in toolset._list_skills()
        monkeypatch.setenv("SECOND_PHOTOS_KEY", "k")
        assert 'name="photos"' in toolset._list_skills()
        monkeypatch.delenv("SECOND_PHOTOS_KEY")
        assert "photos" not in toolset._list_skills()

    @pytest.mark.asyncio
    async def test_no_tool_when_no_skill_is_activated(self, tmp_path: Path):
        write_skill(tmp_path, "photos", ANY_OF)
        toolset = await self.toolset(tmp_path, {})
        assert await toolset.get_tools(Ctx()) == {}
        assert await toolset.get_instructions() == ""
        assert toolset._list_skills() == "No skills available."
        toolset = await self.toolset(tmp_path, {"FIRST_PHOTOS_KEY": "k"})
        assert set(await toolset.get_tools(Ctx())) == {
            "list_skills",
            "load_skill",
            "read_skill_resource",
            "run_skill_script",
        }

    @pytest.mark.asyncio
    async def test_enforcement_can_be_turned_off(self, directory: Path):
        toolset = await self.toolset(directory, {}, enforce_activation=False)
        listed = toolset._list_skills()
        for name in ("plain", "photos", "service", "legacy"):
            assert f'name="{name}"' in listed
        assert toolset.get_inactive_skills() == {}

    @pytest.mark.asyncio
    async def test_a_request_identity_activates(self, directory: Path, monkeypatch):
        import sys
        import types

        monkeypatch.delenv("LEGACY_TOKEN", raising=False)
        identities = types.ModuleType("agent_runtimes.context.identities")
        identities.get_identity_env = lambda: {"LEGACY_TOKEN": "from-a-sign-in"}
        toolset = AgentSkillsToolset(directories=[directory])
        await toolset._ensure_initialized()
        assert "legacy" not in toolset._list_skills()
        monkeypatch.setitem(sys.modules, "agent_runtimes.context.identities", identities)
        assert 'name="legacy"' in toolset._list_skills()


def test_a_command_line_is_described_by_its_arguments(tmp_path: Path):
    """A script with helpers before its argparse is described by its arguments, not a helper."""
    path = write_skill(tmp_path, "cli")
    (path / "scripts" / "run.py").write_text(
        '''"""Find things.

Usage:
    python run.py QUERY [--limit 5]
"""

import argparse


def setting(name: str, default: str = "") -> str:
    return default


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="What to look for")
    parser.add_argument("--limit", type=int, default=5, help="How many")
    return vars(parser.parse_args())
'''
    )
    description = AgentSkill.from_skill_md(path).scripts[0].description
    assert description.startswith("Find things.")
    assert "query — What to look for" in description
    assert "limit: int — How many" in description
    assert "default" not in description and "Returns" not in description


def test_a_command_line_without_arguments_takes_none(tmp_path: Path):
    path = write_skill(tmp_path, "cli")
    (path / "scripts" / "run.py").write_text(
        '''"""List things."""

import argparse


def setting(name: str, default: str = "") -> str:
    return default


def main() -> dict:
    argparse.ArgumentParser(description="List things").parse_args()
    return {}
'''
    )
    assert AgentSkill.from_skill_md(path).scripts[0].description == "List things."


# ---- Review of the env block and activation --------------------------------------------------

LEGACY_BOTH = (
    "## Required Environment Variables\n\n- `LEGACY_TOKEN` (required)\n\n"
    "## Optional Environment Variables\n\n- `LEGACY_LIMIT`: a limit.\n"
)

DOCUMENTED_SCRIPT = '''"""Run it.

Environment:
    LEGACY_LIMIT
    SCRIPT_TOKEN
"""

print("ran")
'''


class TestReview:
    def test_an_empty_env_block_declares_that_nothing_is_needed(self, tmp_path: Path):
        """`env: {}` is a declaration: the legacy sections and scripts are not read."""
        path = write_skill(tmp_path, "nothing", "env: {}\n", LEGACY_BOTH)
        (path / "scripts" / "run.py").write_text(DOCUMENTED_SCRIPT)
        skill = AgentSkill.from_skill_md(path)
        assert skill.get_required_env_vars() == []
        assert skill.get_optional_env_vars() == []
        assert skill.is_activated({})

    def test_an_env_block_left_blank_declares_nothing_too(self, tmp_path: Path):
        skill = AgentSkill.from_skill_md(write_skill(tmp_path, "blank", "env:\n", SECTION))
        assert skill.get_required_env_vars() == []
        assert skill.is_activated({})

    def test_a_programmatic_skill_may_declare_an_empty_env(self):
        body = "## Required Environment Variables\n\n- `OLD_TOKEN`\n"
        assert AgentSkill(name="a", description="a", content=body).get_required_env_vars() == [
            "OLD_TOKEN"
        ]
        declared = AgentSkill(name="a", description="a", content=body, env={})
        assert declared.get_required_env_vars() == []
        assert declared.is_activated({})

    def test_a_scripts_optional_variable_is_not_required(self, tmp_path: Path):
        """A variable the SKILL.md says is optional is not required because a script names it."""
        body = "## Optional Environment Variables\n\n- `LEGACY_LIMIT`: a limit.\n"
        path = write_skill(tmp_path, "legacy", body=body)
        (path / "scripts" / "run.py").write_text(DOCUMENTED_SCRIPT)
        skill = AgentSkill.from_skill_md(path)
        assert skill.get_required_env_vars() == ["SCRIPT_TOKEN"]
        assert skill.get_optional_env_vars() == ["LEGACY_LIMIT"]
        assert skill.is_activated({"SCRIPT_TOKEN": "t"})

    def test_a_legacy_skill_says_which_optional_variables_are_set(self, tmp_path: Path):
        skill = AgentSkill.from_skill_md(write_skill(tmp_path, "legacy", body=LEGACY_BOTH))
        env = {"LEGACY_TOKEN": "t", "LEGACY_LIMIT": "the-value"}
        assert "optional_env_set=LEGACY_LIMIT" in skill.get_discovery_summary(env)
        assert "optional_env_set=none" in skill.get_discovery_summary({"LEGACY_TOKEN": "t"})
        content = skill.get_full_content(env)
        assert "**Environment set now:** LEGACY_LIMIT" in content
        assert "**Environment not set:** LEGACY_LIMIT" in skill.get_full_content({})
        assert "the-value" not in content + skill.get_skills_header(env)

    def test_a_skill_with_no_optional_variable_says_nothing_of_them(self, tmp_path: Path):
        skill = AgentSkill.from_skill_md(write_skill(tmp_path, "plain"))
        assert "optional_env_set" not in skill.get_discovery_summary({})
        assert "Environment set now" not in skill.get_full_content({})


@pytest.mark.skipif(not PYDANTIC_AI_AVAILABLE, reason="pydantic-ai not installed")
class TestReviewToolset:
    @pytest.fixture
    def directory(self, tmp_path: Path) -> Path:
        write_skill(tmp_path, "plain")
        write_skill(tmp_path, "photos", ANY_OF)
        return tmp_path

    def test_diagnostics_need_no_initializer(self, directory: Path):
        """A freshly built toolset answers for its directories without a private call first."""
        toolset = AgentSkillsToolset(directories=[directory], env={})
        assert toolset.get_inactive_skills() == {
            "photos": "none of FIRST_PHOTOS_KEY, SECOND_PHOTOS_KEY set"
        }
        assert sorted(AgentSkillsToolset(directories=[directory], env={}).get_active_skills()) == [
            "plain"
        ]

    def test_diagnostics_discover_the_entry_points(self, directory: Path, monkeypatch):
        hidden = AgentSkill.from_skill_md(directory / "photos")
        monkeypatch.setattr(toolset_module, "discover_entrypoint_skills", lambda: [hidden])
        toolset = AgentSkillsToolset(discover_entrypoints=True, env={})
        assert list(toolset.get_inactive_skills()) == ["photos"]

    @pytest.mark.asyncio
    async def test_instructions_and_tools_need_no_initializer(self, directory: Path):
        toolset = AgentSkillsToolset(directories=[directory], env={})
        assert 'name="plain"' in await toolset.get_instructions()
        only_hidden = AgentSkillsToolset(directories=[directory / "photos"], env={})
        assert await only_hidden.get_instructions() == ""
        assert await only_hidden.get_tools(Ctx()) == {}

    def test_private_readers_need_no_initializer(self, directory: Path):
        toolset = AgentSkillsToolset(directories=[directory], env={})
        assert 'name="plain"' in toolset._list_skills()
        assert "# Skill: plain" in AgentSkillsToolset(directories=[directory], env={})._load_skill(
            "plain"
        )

    @pytest.mark.asyncio
    async def test_discovery_runs_once(self, directory: Path, monkeypatch):
        calls = []
        real = AgentSkill.from_skill_md
        monkeypatch.setattr(
            AgentSkill, "from_skill_md", lambda path: (calls.append(path), real(path))[1]
        )
        toolset = AgentSkillsToolset(directories=[directory], env={})
        toolset.get_inactive_skills()
        await toolset._ensure_initialized()
        toolset.get_active_skills()
        assert len(calls) == 2
