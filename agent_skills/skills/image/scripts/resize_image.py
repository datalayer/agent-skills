# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import sys

from PIL import Image

# Resizes an image so that its longest side is at most `max_dim` pixels, keeping its aspect.


def resize(input_path, output_path, max_dim):
    with Image.open(input_path) as image:
        width, height = image.size
        if width > max_dim or height > max_dim:
            scale = min(max_dim / width, max_dim / height)
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS
            )
        image.save(output_path)
        print(f"Saved {output_path} (size: {image.size})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: resize_image.py [input image] [output image] [max dimension in pixels]")
        sys.exit(1)
    resize(sys.argv[1], sys.argv[2], int(sys.argv[3]))
