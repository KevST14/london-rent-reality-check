"""Take a screenshot of the running Streamlit app for the README.

Usage:
    # in one terminal
    uv run streamlit run app/streamlit_app.py
    # in another
    uv run python scripts/screenshot.py            # -> docs/app.png
    uv run python scripts/screenshot.py http://localhost:8531 docs/app.png

Needs the browser downloaded once:  uv run playwright install chromium
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8531"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("docs/app.png")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1250}, device_scale_factor=2)
        page.goto(URL, wait_until="networkidle", timeout=60_000)
        # wait for the model output to render, then let plots settle
        page.get_by_text("Predicted fair price").wait_for(timeout=60_000)
        page.get_by_text("Why this price?").wait_for(timeout=60_000)
        time.sleep(4)
        page.screenshot(path=str(OUT), full_page=True)
        browser.close()
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
