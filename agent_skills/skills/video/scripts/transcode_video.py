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


# Transcodes a video to the container its output name says (.mp4 with H.264 and AAC, .webm with VP9 and Opus), scaled to a width when asked.


def transcode(input_path, output_path, width=None, crf=23):
    target = output_path.rsplit(".", 1)[-1].lower()
    arguments = ["-i", input_path]
    if width:
        arguments += ["-vf", f"scale={int(width)}:-2"]
    if target == "webm":
        arguments += ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-c:a", "libopus"]
    else:
        arguments += [
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
        ]
    run([*arguments, output_path])
    print(f"Saved {output_path} ({target}{f', width {width}' if width else ''}, crf {crf})")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print(
            "Usage: transcode_video.py [input video] [output video] [width in pixels, kept when unsaid] [crf, 23 when unsaid]"
        )
        sys.exit(1)
    transcode(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] not in ("", "0", "-") else None,
        int(sys.argv[4]) if len(sys.argv) > 4 else 23,
    )
