# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import sys

from PIL import Image

# Crops an image to a box: left, top, right, bottom, in pixels.


def crop(input_path, output_path, left, top, right, bottom):
    with Image.open(input_path) as image:
        cropped = image.crop((left, top, right, bottom))
        cropped.save(output_path)
        print(f"Saved {output_path} (size: {cropped.size})")


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: crop_image.py [input image] [output image] [left] [top] [right] [bottom]")
        sys.exit(1)
    crop(sys.argv[1], sys.argv[2], *[int(value) for value in sys.argv[3:7]])
