# Rent Reality Check — London

Given a London rental listing, predict a fair price, explain what's driving it,
and show where that area's rents are heading.

**Status:** 🚧 in progress — week 1 (data + EDA)

## The three questions

| Pillar | Question | Method |
|---|---|---|
| Tabular ML | What's a fair price for *this* listing? | Gradient boosting + geospatial feature engineering, SHAP for explanation |
| NLP | Does *how* a host writes the description predict price on top of the actual features? | Structured-only model vs. structured + text, nested comparison |
| Time series | Where are rents in this area heading over the next 6–12 months? | Trend/seasonality decomposition, walk-forward backtest vs. naive baseline |

## Deliverable

A Streamlit app: enter a listing → predicted fair price + SHAP waterfall +
"~X% above/below model" verdict + area rent-trend chart. Plus this repo and a writeup.

## Data

See [`data/README.md`](data/README.md) for sources and download steps. Raw data is
git-ignored; the folder structure is kept.

## Setup

```bash
uv sync
uv run jupyter lab
```

## Layout

```
data/            raw -> interim -> processed  (git-ignored)
notebooks/       narrative: EDA, modeling, time series
src/londonrent/  reusable code (data loading, features, geo, models)
app/             Streamlit app
reports/         figures and the writeup
tests/           tests for the src package
```
