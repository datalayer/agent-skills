<!--
  ~ Copyright (c) 2023-2024 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

# Changelog

## 1.5.0

18 November 2026.

- **A skill says the environment it reads, and is activated only when it has it.** A `SKILL.md` frontmatter takes an `env` block: `required` (or `all_of`) for the variables the skill cannot work without, `optional` for those it reads when they are set, and `any_of` for those of which one is enough. A skill is activated when every required variable is set and, when it names some, one at least of `any_of`; an empty value is not set. Without an `env` block a skill's `Required Environment Variables` section is read as before, and its `Optional Environment Variables` section now too. New on `AgentSkill`: `env`, `get_optional_env_vars`, `get_any_of_env_vars`, `get_missing_env_vars`, `get_set_env_vars`, `is_activated`, `get_inactive_reason`; `parse_skill_env` and `activation_env` are exported.
- **Behaviour change: `AgentSkillsToolset` keeps a skill that is not activated out.** It is not listed (`list_skills`, the instructions), and `load_skill`, `read_skill_resource` and `run_skill_script` answer that it is not found; a toolset whose skills are all inactive offers no tool. Until now a skill whose required variable was missing was listed and its scripts failed when run. The environment is the process's, with the variables the request's identities stand for, read at each call; `env=` decides on another mapping, `enforce_activation=False` offers every skill as before, and `get_inactive_skills()` says which are left out and why, by variable name. For a skill with optional variables the listing and `load_skill` say which are set now.

## 1.4.0

12 November 2026.

- **The framework alone.** `accounting`, `crawl`, `events`, `github`, `image`, `pdf`, `text_summarizer` and `video` leave this package for [`datalayer-skills`](https://github.com/datalayer/skills) 0.3.0, where they are `datalayer_skills.skills.<name>` under the same skill names, found through its `agent_skills.skills` entry points. **Breaking:** `agent_skills.skills.<name>` no longer imports for those eight, with no alias left behind, and the extras `skills`, `skills-crawl`, `skills-github`, `skills-pdf`, `skills-accounting`, `skills-image` and `skills-video` are gone: install `datalayer-skills[skills]` (or `datalayer-skills[skills-<name>]`) and name `datalayer_skills.skills.<name>`. `jokes` stays, as the example of a module-based skill.

## 1.3.0

7 November 2026.

- **Public skills.** A skill whose `SKILL.md` frontmatter says `metadata.visibility: public` and that needs no secret may be published by a company built on Datalayer to people's personal agents, as its `skill_index` (Personal Agent Protocol 0.3 §9.4). `accounting`, `crawl`, `image`, `jokes`, `pdf`, `text_summarizer` and `video` say so; `github` and `events` need a token and do not.

## 1.2.1

7 November 2026.

- **Accounting and Jokes skills.** The two skills the catalogue named without code: `accounting` — a reconciliation of two ledgers, the variance between two periods, the aging of open invoices and an audit-ready close summary of a trial balance, over CSV exports, with pandas (extra `skills-accounting`, in `skills`); `jokes` — one joke on request, by topic or at random, from a small bundled list. Both load from their modules like the others.

## 1.2.0

7 November 2026.

- **Image and Video skills.** Two bundled skills: `image` — snapshots of web pages (Playwright) and of video frames (ffmpeg), and images read, resized, cropped, converted, optimized for the web and thumbnailed (Pillow); `video` — recordings of web pages (Playwright), and videos read, trimmed, transcoded, optimized for the web, thumbnailed, turned to GIF and joined (ffmpeg, through `imageio-ffmpeg` or the system's). Extras `skills-image` and `skills-video`, in `skills` with the others.
