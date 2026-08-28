"""Draw the project's data-to-app pipeline as docs/pipeline.png.

Run:  uv run python scripts/pipeline_diagram.py   (or: make diagram)

Kept in code so the diagram stays in sync with the repo and there is no binary to
hand-edit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT = Path("docs/pipeline.png")

BLUE, LILAC, GREEN, YELLOW = "#dbeafe", "#e0e7ff", "#dcfce7", "#fef9c3"

# name -> (x, y, w, h, title, subtitle, colour)
BOXES = {
    "raw": (
        0.01,
        0.42,
        0.17,
        0.30,
        "Raw data",
        "Inside Airbnb\nlistings + calendar\n+ OpenStreetMap POIs",
        BLUE,
    ),
    "data": (
        0.24,
        0.56,
        0.17,
        0.22,
        "data.py",
        "parse prices, dates,\nbathroom text;\nfilter to London",
        LILAC,
    ),
    "geo": (
        0.24,
        0.26,
        0.17,
        0.22,
        "geo.py",
        "project to metres;\ndistance to station,\ncentre, park, food",
        LILAC,
    ),
    "feat": (
        0.47,
        0.38,
        0.19,
        0.32,
        "features.py",
        "scope (price, room type,\nno hotels); amenity flags;\nproperty class; geo join",
        LILAC,
    ),
    "model": (
        0.72,
        0.38,
        0.19,
        0.32,
        "model.py",
        "one-hot encode;\nHistGradientBoosting;\nspatial block CV;\nmetrics",
        LILAC,
    ),
    "nb": (
        0.45,
        0.02,
        0.50,
        0.21,
        "notebooks 01-04 + 03b",
        "EDA, ablations, tuning, SHAP,\nPDP / ICE, mispricing scan, text analysis",
        GREEN,
    ),
    "bundle": (0.70, 0.80, 0.20, 0.13, "models/\nprice_model.joblib", "", YELLOW),
    "app": (
        0.06,
        0.80,
        0.20,
        0.13,
        "app/streamlit_app.py",
        "predict + explain\n(live demo)",
        GREEN,
    ),
}

ARROWS = [
    ("raw", "right", "data", "left"),
    ("raw", "right", "geo", "left"),
    ("data", "right", "feat", "left"),
    ("geo", "right", "feat", "left"),
    ("feat", "right", "model", "left"),
    ("model", "bottom", "nb", "top"),
    ("model", "top", "bundle", "bottom"),
    ("bundle", "left", "app", "right"),
]


def _anchor(box, side):
    x, y, w, h = box[:4]
    return {
        "left": (x, y + h / 2),
        "right": (x + w, y + h / 2),
        "top": (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
    }[side]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.06)
    ax.axis("off")

    ax.text(
        0.5,
        1.045,
        "Rent Reality Check: from raw data to a live app",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )

    for x, y, w, h, title, sub, colour in BOXES.values():
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                linewidth=1.1,
                edgecolor="#334155",
                facecolor=colour,
            )
        )
        ax.text(
            x + w / 2,
            y + h - 0.04,
            title,
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color="#0f172a",
        )
        if sub:
            ax.text(
                x + w / 2, y + h - 0.095, sub, ha="center", va="top", fontsize=8.3, color="#334155"
            )

    for a, aside, b, bside in ARROWS:
        x0, y0 = _anchor(BOXES[a], aside)
        x1, y1 = _anchor(BOXES[b], bside)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.6, shrinkA=2, shrinkB=2),
        )

    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
