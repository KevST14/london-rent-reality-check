"""Project paths. Import these instead of hard-coding paths in notebooks."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA = PROJECT_ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"

REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

# London bounding box (approx, for filtering stray coordinates): (min_lon, min_lat, max_lon, max_lat)
LONDON_BBOX = (-0.51, 51.28, 0.33, 51.69)
