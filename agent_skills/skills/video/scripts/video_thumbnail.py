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


# Takes one frame of a video, at a time in seconds, as a PNG: its thumbnail.


def thumbnail(input_path, seconds, output_path):
    run(["-ss", str(seconds), "-i", input_path, "-frames:v", "1", output_path])
    print(f"Saved {output_path} (frame at {seconds} s of {input_path})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: video_thumbnail.py [input video] [seconds] [output png]")
        sys.exit(1)
    thumbnail(sys.argv[1], float(sys.argv[2]), sys.argv[3])
