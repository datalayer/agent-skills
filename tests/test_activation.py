# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""A skill says the environment it reads, and is activated only when it has what it needs."""

from pathlib import Path

import pytest

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
