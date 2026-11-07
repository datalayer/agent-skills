# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import shutil
import subprocess
import sys


def ffmpeg_binary():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if not found:
            raise RuntimeError(
                "ffmpeg is not installed: pip install imageio-ffmpeg, or install ffmpeg"
            )
        return found


def run(arguments):
    command = [ffmpeg_binary(), "-y", *arguments]
    completed = subprocess.run(command, capture_output=True, text=True)  # noqa: S603
    if completed.returncode != 0:
        print(
            completed.stderr.strip().splitlines()[-1]
            if completed.stderr.strip()
            else "ffmpeg failed"
        )
        sys.exit(completed.returncode)
    return completed


import os
import tempfile

# Joins videos one after the other into one file, re-encoded to one size and codec so that different sources fit.


def concat(output_path, inputs):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        for path in inputs:
            listing.write(f"file '{os.path.abspath(path)}'\n")
        listing_path = listing.name
    try:
        run(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                listing_path,
                "-c:v",
                "libx264",
                "-crf",
                "23",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                output_path,
            ]
        )
    finally:
        os.remove(listing_path)
    print(f"Saved {output_path} ({len(inputs)} videos joined)")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: concat_videos.py [output video] [input video] [input video] ...")
        sys.exit(1)
    concat(sys.argv[1], sys.argv[2:])
