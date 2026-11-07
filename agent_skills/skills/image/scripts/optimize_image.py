# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import os
import sys

from PIL import Image

# Optimizes an image for the web: metadata stripped, the encoder told to optimize, a JPEG or WebP at the quality given; says the bytes before and after.


def optimize(input_path, output_path, quality=82):
    before = os.path.getsize(input_path)
    with Image.open(input_path) as image:
        clean = Image.new(image.mode, image.size)
        clean.putdata(list(image.getdata()))
        target = output_path.rsplit(".", 1)[-1].lower()
        if target in ("jpg", "jpeg"):
            clean.convert("RGB").save(output_path, optimize=True, quality=quality, progressive=True)
        elif target == "webp":
            clean.save(output_path, quality=quality, method=6)
        else:
            clean.save(output_path, optimize=True)
    after = os.path.getsize(output_path)
    print(
        f"Saved {output_path}: {before} bytes before, {after} bytes after ({100 - round(after * 100 / max(before, 1))}% smaller)"
    )


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(
            "Usage: optimize_image.py [input image] [output image] [quality 1-100, 82 when unsaid]"
        )
        sys.exit(1)
    optimize(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) == 4 else 82)
