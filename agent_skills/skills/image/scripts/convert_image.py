# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import sys

from PIL import Image

# Converts an image to the format its output name says (.png, .jpg, .webp, .gif, .bmp, .tiff); a JPEG loses its alpha on white.


def convert(input_path, output_path, quality=90):
    with Image.open(input_path) as image:
        target = output_path.rsplit(".", 1)[-1].lower()
        if target in ("jpg", "jpeg") and image.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image.convert("RGBA"), mask=image.convert("RGBA").split()[-1])
            image = background
        image.save(output_path, quality=quality)
        print(f"Saved {output_path} ({target}, quality {quality})")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(
            "Usage: convert_image.py [input image] [output image] [quality 1-100, 90 when unsaid]"
        )
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) == 4 else 90)
