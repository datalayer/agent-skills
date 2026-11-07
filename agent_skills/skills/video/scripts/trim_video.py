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


# Trims a video: from a start, for a duration, both in seconds; re-encoded, so the cut is exact.


def trim(input_path, output_path, start, duration):
    run(["-ss", str(start), "-i", input_path, "-t", str(duration), output_path])
    print(f"Saved {output_path} ({duration} s from {start} s of {input_path})")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: trim_video.py [input video] [output video] [start seconds] [duration seconds]"
        )
        sys.exit(1)
    trim(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
