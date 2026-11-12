<!--
  ~ Copyright (c) 2023-2024 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

# Changelog

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
