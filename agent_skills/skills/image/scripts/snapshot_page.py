# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import sys

# Takes a snapshot of a web page as a PNG, with a headless Chromium (Playwright): the viewport, or the whole page.


def snapshot(url, output_path, width=1280, height=800, full_page=False):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(url, wait_until="networkidle")
        page.screenshot(path=output_path, full_page=full_page)
        browser.close()
    print(f"Saved {output_path} ({url}, {width}x{height}{', whole page' if full_page else ''})")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 6:
        print(
            "Usage: snapshot_page.py [url] [output png] [width, 1280 when unsaid] [height, 800 when unsaid] [full_page: yes or no]"
        )
        sys.exit(1)
    snapshot(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]) if len(sys.argv) > 3 else 1280,
        int(sys.argv[4]) if len(sys.argv) > 4 else 800,
        len(sys.argv) > 5 and sys.argv[5].lower() in ("yes", "true", "1", "full"),
    )
