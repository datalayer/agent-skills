# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import os
import shutil
import sys
import tempfile

# Records a web page for a few seconds as a WebM, with a headless Chromium (Playwright): a video of the page as it loads and moves.


def record(url, output_path, seconds=10, width=1280, height=800):
    from playwright.sync_api import sync_playwright

    with tempfile.TemporaryDirectory() as folder, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=folder,
            record_video_size={"width": width, "height": height},
        )
        page = context.new_page()
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(int(seconds * 1000))
        recorded = page.video.path() if page.video else None
        context.close()
        browser.close()
        if not recorded or not os.path.exists(recorded):
            print("No video was recorded")
            sys.exit(1)
        shutil.move(recorded, output_path)
    print(f"Saved {output_path} ({url}, {seconds} s, {width}x{height})")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 6:
        print(
            "Usage: record_page.py [url] [output webm] [seconds, 10 when unsaid] [width, 1280 when unsaid] [height, 800 when unsaid]"
        )
        sys.exit(1)
    record(
        sys.argv[1],
        sys.argv[2],
        float(sys.argv[3]) if len(sys.argv) > 3 else 10,
        int(sys.argv[4]) if len(sys.argv) > 4 else 1280,
        int(sys.argv[5]) if len(sys.argv) > 5 else 800,
    )
