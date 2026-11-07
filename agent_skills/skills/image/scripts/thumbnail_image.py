# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import sys

from PIL import Image

# Makes a thumbnail that fits in a square of `size` pixels, the aspect kept.


def thumbnail(input_path, output_path, size):
    with Image.open(input_path) as image:
        image.thumbnail((size, size), Image.LANCZOS)
        image.save(output_path)
        print(f"Saved {output_path} (size: {image.size})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: thumbnail_image.py [input image] [output image] [size in pixels]")
        sys.exit(1)
    thumbnail(sys.argv[1], sys.argv[2], int(sys.argv[3]))
