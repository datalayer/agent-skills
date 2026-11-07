# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import json
import os
import sys

from PIL import Image

# Prints what an image is: its format, its size in pixels, its mode, its bytes, and the EXIF keys it carries.


def info(path):
    with Image.open(path) as image:
        exif = image.getexif()
        return {
            "path": path,
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "bytes": os.path.getsize(path),
            "exif_keys": sorted(str(key) for key in exif.keys()),
        }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: image_info.py [image]")
        sys.exit(1)
    print(json.dumps(info(sys.argv[1]), indent=2))
