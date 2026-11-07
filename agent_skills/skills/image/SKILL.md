---
name: image
description: Image toolkit for snapshots of web pages and video frames, and images read, resized, cropped, converted, optimized for the web and thumbnailed.
license: BSD 3-Clause. LICENSE has complete terms
version: 1.0.0
tags:
  - image
  - snapshot
  - screenshot
  - optimize
author: Datalayer
metadata:
  visibility: public
---

# Image Skill

Snapshots of pages and recordings, and what is done to an image afterwards: read
what it is, resize it, crop it, convert it, optimize it for the web, make its
thumbnail. Pillow does the images; a page snapshot runs a headless Chromium
through Playwright; a video frame is taken with ffmpeg.

## Required Environment Variables

- None required.

## Required Software

- `pip install "agent_skills[skills-image]"` — Pillow and Playwright.
- `python -m playwright install chromium` once, for `snapshot_page` (the runtime image carries it).
- ffmpeg, for `snapshot_video_frame`: `imageio-ffmpeg` brings a binary, or the system's.

## Invocation Contract

- Call `run_skill_script` with:
  - `skill_name`: `image`
  - `script_name`: one of `image_info`, `resize_image`, `crop_image`, `convert_image`, `optimize_image`, `thumbnail_image`, `snapshot_page`, `snapshot_video_frame`
- These scripts are positional-argument based. Use `args`, not `kwargs`.
- The output format follows the output file's extension: `.png`, `.jpg`, `.webp`, `.gif`, `.bmp`, `.tiff`.

## Scripts API

### `script_name: image_info`

- `args`: `["<image>"]`
- Output: JSON — `format`, `width`, `height`, `mode`, `bytes`, `exif_keys`.

### `script_name: resize_image`

- `args`: `["<input image>", "<output image>", "<max dimension in pixels>"]`
- Output: the image with its longest side at most that many pixels, the aspect kept.

### `script_name: crop_image`

- `args`: `["<input image>", "<output image>", "<left>", "<top>", "<right>", "<bottom>"]`
- Output: the box cut out, in pixels.

### `script_name: convert_image`

- `args`: `["<input image>", "<output image>", "<quality 1-100, 90 when unsaid>"]`
- Output: the image in the output's format; a JPEG loses its transparency on white.

### `script_name: optimize_image`

- `args`: `["<input image>", "<output image>", "<quality 1-100, 82 when unsaid>"]`
- Output: metadata stripped, the encoder told to optimize, a progressive JPEG or a WebP at that quality; says the bytes before and after.

### `script_name: thumbnail_image`

- `args`: `["<input image>", "<output image>", "<size in pixels>"]`
- Output: a thumbnail that fits in a square of that size.

### `script_name: snapshot_page`

- `args`: `["<url>", "<output png>", "<width, 1280 when unsaid>", "<height, 800 when unsaid>", "<full_page: yes or no>"]`
- Output: a PNG of the page once its network is idle — the viewport, or the whole page.

### `script_name: snapshot_video_frame`

- `args`: `["<input video>", "<seconds>", "<output png>"]`
- Output: the frame at that time, as a PNG.
