---
name: video
description: Video toolkit for recordings of web pages, and videos read, trimmed, transcoded, optimized for the web, thumbnailed, turned to GIF and joined.
license: BSD 3-Clause. LICENSE has complete terms
version: 1.0.0
tags:
  - video
  - recording
  - ffmpeg
  - optimize
author: Datalayer
metadata:
  visibility: public
---

# Video Skill

Recordings of pages, and what is done to a video afterwards: read what it is,
trim it, transcode it, optimize it for the web, take its thumbnail, turn a clip
into a GIF, join several. ffmpeg does the videos (`imageio-ffmpeg` brings a
binary, or the system's is used); a page recording runs a headless Chromium
through Playwright.

## Required Environment Variables

- None required.

## Required Software

- `pip install "agent_skills[skills-video]"` — imageio-ffmpeg and Playwright.
- `python -m playwright install chromium` once, for `record_page` (the runtime image carries it).
- ffmpeg on the system is used when `imageio-ffmpeg` is not installed.

## Invocation Contract

- Call `run_skill_script` with:
  - `skill_name`: `video`
  - `script_name`: one of `video_info`, `record_page`, `trim_video`, `transcode_video`, `optimize_video`, `video_thumbnail`, `video_to_gif`, `concat_videos`
- These scripts are positional-argument based. Use `args`, not `kwargs`.
- Times are in seconds; the output container follows the output file's extension (`.mp4`, `.webm`, `.gif`, `.png`).

## Scripts API

### `script_name: video_info`

- `args`: `["<video>"]`
- Output: JSON — `seconds`, `bytes`, and each stream's `kind`, `codec`, `width`, `height`, `fps`.

### `script_name: record_page`

- `args`: `["<url>", "<output webm>", "<seconds, 10 when unsaid>", "<width, 1280 when unsaid>", "<height, 800 when unsaid>"]`
- Output: a WebM of the page from its load, for that many seconds.

### `script_name: trim_video`

- `args`: `["<input video>", "<output video>", "<start seconds>", "<duration seconds>"]`
- Output: that clip, re-encoded so the cut is exact.

### `script_name: transcode_video`

- `args`: `["<input video>", "<output video>", "<width in pixels, kept when unsaid>", "<crf, 23 when unsaid>"]`
- Output: an `.mp4` (H.264, AAC, fast start) or a `.webm` (VP9, Opus), scaled to the width when one is given.

### `script_name: optimize_video`

- `args`: `["<input video>", "<output mp4>", "<crf, 28 when unsaid>", "<preset, slow when unsaid>"]`
- Output: a smaller H.264 at constant quality; says the bytes before and after.

### `script_name: video_thumbnail`

- `args`: `["<input video>", "<seconds>", "<output png>"]`
- Output: the frame at that time, as a PNG.

### `script_name: video_to_gif`

- `args`: `["<input video>", "<output gif>", "<start seconds>", "<duration seconds>", "<fps, 12 when unsaid>", "<width, 480 when unsaid>"]`
- Output: an optimized GIF of that clip, a palette made from it first.

### `script_name: concat_videos`

- `args`: `["<output video>", "<input video>", "<input video>", ...]`
- Output: the videos one after the other, re-encoded to one codec.
