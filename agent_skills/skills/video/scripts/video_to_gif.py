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


# Turns a clip of a video into an optimized GIF: a palette made from the clip first, then the GIF with it.


def to_gif(input_path, output_path, start, duration, fps=12, width=480):
    filters = f"fps={fps},scale={width}:-1:flags=lanczos"
    palette = f"{output_path}.palette.png"
    run(
        [
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            input_path,
            "-vf",
            f"{filters},palettegen",
            palette,
        ]
    )
    run(
        [
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            input_path,
            "-i",
            palette,
            "-lavfi",
            f"{filters} [x]; [x][1:v] paletteuse",
            output_path,
        ]
    )
    try:
        import os

        os.remove(palette)
    except OSError:
        pass
    print(f"Saved {output_path} ({duration} s from {start} s, {fps} fps, width {width})")


if __name__ == "__main__":
    if len(sys.argv) < 5 or len(sys.argv) > 7:
        print(
            "Usage: video_to_gif.py [input video] [output gif] [start seconds] [duration seconds] [fps, 12 when unsaid] [width, 480 when unsaid]"
        )
        sys.exit(1)
    to_gif(
        sys.argv[1],
        sys.argv[2],
        float(sys.argv[3]),
        float(sys.argv[4]),
        int(sys.argv[5]) if len(sys.argv) > 5 else 12,
        int(sys.argv[6]) if len(sys.argv) > 6 else 480,
    )
