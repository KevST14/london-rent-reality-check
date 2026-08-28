# Rent Reality Check — London

Given a London rental listing, predict a fair price, explain what's driving it,
and show where that area's rents are heading.

**Status:** 🚧 in progress — week 1 (data + EDA)

## The three questions

| Pillar | Question | Method |
|---|---|---|
| Tabular ML | What's a fair nightly price for *this* listing? | Gradient boosting on listing attributes + engineered distance-to-Tube / centre / POI features; SHAP to explain each prediction |
| NLP | Does *how* a host writes the `description` predict price on top of the structured features? | Structured-only model vs. structured + text (TF-IDF / embeddings), nested comparison; which tokens move price |
| Time series | Where are prices heading, and how does demand move through the year? | Primary: 25-year borough sale-price trend + forecast from "Housing in London", walk-forward backtest vs. seasonal-naive. Secondary: Airbnb forward-availability -> borough booking-curve seasonality |

Data: Inside Airbnb London (2026-06-19 snapshot) + Kaggle `justinas/housing-in-london`.
It's short-let, not long-let — the framing is "is this listing priced right for its area and features?"

## Deliverable

A Streamlit app: enter a listing → predicted fair price + SHAP waterfall +
"~X% above/below model" verdict + borough price-trend chart. Plus this repo and a writeup.

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
