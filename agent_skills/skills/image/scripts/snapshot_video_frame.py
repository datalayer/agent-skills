# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import shutil
import subprocess
import sys

# Takes one frame of a video, at a time in seconds, as a PNG — a snapshot of a recording.


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


def snapshot_frame(video_path, seconds, output_path):
    command = [
        ffmpeg_binary(),
        "-y",
        "-ss",
        str(seconds),
        "-i",
        video_path,
        "-frames:v",
        "1",
        output_path,
    ]
    subprocess.run(command, check=True, capture_output=True)  # noqa: S603
    print(f"Saved {output_path} (frame at {seconds} s of {video_path})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: snapshot_video_frame.py [input video] [seconds] [output png]")
        sys.exit(1)
    snapshot_frame(sys.argv[1], float(sys.argv[2]), sys.argv[3])
