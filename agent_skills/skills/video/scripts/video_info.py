# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import json
import os
import re
import shutil
import subprocess
import sys

# Prints what a video is: its duration, its streams (codec, size, frame rate), and its bytes, read off ffmpeg.


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


def info(path):
    completed = subprocess.run([ffmpeg_binary(), "-i", path], capture_output=True, text=True)  # noqa: S603
    text = completed.stderr
    duration = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", text)
    seconds = None
    if duration:
        hours, minutes, secs = duration.groups()
        seconds = int(hours) * 3600 + int(minutes) * 60 + float(secs)
    streams = []
    for line in text.splitlines():
        if "Stream #" in line:
            kind = "video" if "Video:" in line else "audio" if "Audio:" in line else "other"
            size = re.search(r"(\d{2,5})x(\d{2,5})", line)
            fps = re.search(r"([\d.]+) fps", line)
            codec = re.search(r"(?:Video|Audio): ([A-Za-z0-9_-]+)", line)
            streams.append(
                {
                    "kind": kind,
                    "codec": codec.group(1) if codec else None,
                    "width": int(size.group(1)) if size else None,
                    "height": int(size.group(2)) if size else None,
                    "fps": float(fps.group(1)) if fps else None,
                }
            )
    return {"path": path, "seconds": seconds, "bytes": os.path.getsize(path), "streams": streams}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: video_info.py [video]")
        sys.exit(1)
    print(json.dumps(info(sys.argv[1]), indent=2))
