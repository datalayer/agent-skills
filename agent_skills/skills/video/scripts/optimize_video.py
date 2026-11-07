# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import os
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


# Optimizes a video for the web: H.264 at a constant quality, a slower preset for a smaller file, fast start; says the bytes before and after.


def optimize(input_path, output_path, crf=28, preset="slow"):
    before = os.path.getsize(input_path)
    run(
        [
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            output_path,
        ]
    )
    after = os.path.getsize(output_path)
    print(
        f"Saved {output_path}: {before} bytes before, {after} bytes after ({100 - round(after * 100 / max(before, 1))}% smaller)"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print(
            "Usage: optimize_video.py [input video] [output mp4] [crf, 28 when unsaid] [preset, slow when unsaid]"
        )
        sys.exit(1)
    optimize(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]) if len(sys.argv) > 3 else 28,
        sys.argv[4] if len(sys.argv) > 4 else "slow",
    )
