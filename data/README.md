# Data sources

All raw data is git-ignored. Structure: `raw/` -> clean -> `interim/` -> feature-engineered -> `processed/`.

## `raw/` — Inside Airbnb, London (primary)

Snapshot **2026-06-19**, from https://insideairbnb.com/get-the-data/

| File | Size (gz) | Rows | Notes |
|---|---|---|---|
| `listings.csv.gz` | 48 MB | 92,638 | 90 columns. The tabular + NLP dataset. |
| `calendar.csv.gz` | 79 MB | 33.9 M | `listing_id, date, available, minimum_nights, maximum_nights`. One year forward (2026-06-19 -> 2027-06-30). **No price column** in this snapshot — it's forward availability only. |
| `neighbourhoods.geojson` | 1 MB | 33 | Borough boundaries for spatial joins / maps. |

Re-download:
```bash
base=https://data.insideairbnb.com/united-kingdom/england/london/2026-06-19
curl -sS -o data/raw/listings.csv.gz     $base/data/listings.csv.gz
curl -sS -o data/raw/calendar.csv.gz     $base/data/calendar.csv.gz
curl -sS -o data/raw/neighbourhoods.geojson $base/visualisations/neighbourhoods.geojson
```

### listings — fields that matter

- **Target:** `price` — string like `$1,234.50`, parsed to float in `londonrent.data`. **33% missing.** Junk in the tails (min £2, max £527k); work in a £10–£1000/night window (~66% of rows).
- **Location:** `latitude`, `longitude` (100% present), `neighbourhood_cleansed` (33 boroughs).
- **Structure:** `property_type`, `room_type` (4 values), `accommodates`, `bedrooms` (74% present), `beds`, `bathrooms_text` (parsed to `bathrooms` + `bath_is_shared`), `amenities` (parsed to `amenity_count`).
- **Host / reviews:** `host_since`, `host_is_superhost`, `number_of_reviews`, `reviews_per_month`, `review_scores_rating`, `review_scores_location`, `estimated_occupancy_l365d`.
- **Text (NLP pillar):** `description` — **98% present, median 69 words**. `neighborhood_overview` is **empty in this snapshot** — do not rely on it.

## `external/` — joins (pull as we reach each pillar)

- **OSM POIs** (Tube/rail stations, parks, supermarkets) via `osmnx` — for distance features. Cache here.
- **"Housing in London"** (Kaggle: `justinas/housing-in-london`) — borough monthly mean **sale** price 1995–2020 + yearly socioeconomic vars. Drives the long-horizon time-series pillar.
  ```bash
  # needs ~/.kaggle/kaggle.json (Kaggle -> Settings -> Create New Token)
  uv run kaggle datasets download -d justinas/housing-in-london -p data/external --unzip
  ```
