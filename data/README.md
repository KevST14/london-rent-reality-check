# Data sources

All raw data is git-ignored. Download into the folders below.

## `raw/` — primary listings dataset (Kaggle)

- **Dataset:** _TODO: fill in Kaggle slug, e.g. `owner/london-rent-dataset`_
- **Download:**
  ```bash
  uv run kaggle datasets download -d <owner/dataset> -p data/raw --unzip
  ```
  (Needs `~/.kaggle/kaggle.json` — Kaggle account → Settings → Create New Token.)
- **Key columns:** _TODO after first look_

## `external/` — joins for feature engineering and trends

- **OpenStreetMap POIs** (transit stops, parks, supermarkets) via `osmnx` — pulled in code, cached here.
- **London rent index by borough** — ONS private rental market statistics / London Datastore.
  Used for the time-series pillar.
- **London borough / LSOA boundaries** (GeoJSON) — London Datastore, for spatial joins.

## Flow

`raw/` → clean → `interim/` → feature-engineered → `processed/` (parquet, model-ready).
