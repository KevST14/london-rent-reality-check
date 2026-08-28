# Rent Reality Check — London

**Given a London short-let listing, predict a fair nightly price, explain the
number, and show where that area's prices have been heading.**

A student portfolio project. The goal is not just a model — it's to show the
*reasoning*: every notebook explains what it's doing and why in plain language,
then adds a note for readers who know ML, and cites where each technique is
commonly learned from.

**Status:** pillars 1–2 of 3 built (price model + explanations). Long-run
time-series pillar and the write-up are next.

---

## What's in here

| Notebook | Question | Headline result |
|---|---|---|
| [`01_eda`](notebooks/01_eda.ipynb) | What are we predicting, on which listings, and what's the score to beat? | Target = `log(price)`, scope = £10–1000/night entire-home + private-room. No-model baseline: **£-MAE ≈ 90 / R² ≈ 0.49**. |
| [`02_geo_features`](notebooks/02_geo_features.ipynb) | Does the exact spot matter, beyond the borough label? | Yes. Six OpenStreetMap features (distance to Tube / centre / food / park) lift R² 0.60 → **0.64** and cut error £78 → **£73**. |
| [`03_price_model`](notebooks/03_price_model.ipynb) | Build, check, tune, and **explain** the real model. | Held-out test: **£-MAE ≈ £60 · median error ≈ 21% · R²(log) ≈ 0.71**. Beats a linear baseline by ~£9. SHAP explains every prediction. |
| _`04` (planned)_ | Does the *description text* predict price on top of the structured features? | — |
| _`05` (planned)_ | Where have London prices been heading? (25-year borough trend) | — |

**Live demo:** [`app/streamlit_app.py`](app/streamlit_app.py) — enter a listing,
get a predicted price, a SHAP breakdown of what drove it, and how a price you're
considering compares.

```bash
uv sync
uv run streamlit run app/streamlit_app.py     # the demo
uv run jupyter lab                             # the notebooks
```

---

## The method, in one paragraph each

**What we're modelling.** Nightly price is lopsided (lots of £80–£250 listings, a
thin tail past £1000), so we predict `log(price)` — that makes the model's errors
*percentages* ("~15% too high"), which is how anyone thinks about price, and stops
a few huge listings dominating. We keep listings priced £10–£1000 that are whole
homes or private rooms. A third of listings have no price, and that gap isn't
random (it varies by area and room type), so the model is honestly described as
"about listings that publish a price".

**How location is handled.** The borough is a blunt instrument — "Camden" spans a
3-minute and a 25-minute walk to a station. We pull stations, parks, and food
venues from OpenStreetMap and, for each listing, compute distance to the nearest
one (and counts within a radius), plus distance to Charing Cross. All distances are
computed on the *British National Grid*, because you can't measure metres on raw
latitude/longitude — 1° east and 1° north are different real distances. These six
features add signal the borough label misses.

**The model and how it's checked.** A gradient-boosted tree
(`HistGradientBoostingRegressor`): hundreds of small trees, each fixing the last
one's mistakes. It handles missing values itself, so we don't impute `bedrooms` or
`review_scores`. It's evaluated with **spatial** cross-validation — hold out whole
2 km blocks of London, not random rows — because nearby listings are near-twins and
random splits let the model "study its own test". It beats a linear baseline by a
clear margin (price depends on interactions trees capture for free), and light
hyperparameter tuning didn't improve it (reported as-is).

**Explaining predictions.** A single accuracy number doesn't tell a host *why*
their listing was priced at £140. **SHAP** does: it splits each prediction into
per-feature contributions ("+£30 Camden, −£15 no dishwasher, …") that add up
exactly to the number. SHAP values come from cooperative game theory (Shapley,
1953); Lundberg & Lee (2017) applied them to ML with an exact fast algorithm for
trees.

**Honesty.** The model under-prices the luxury tail, is weakest in thinly-sampled
outer boroughs, and its worst individual misses are source-data edge cases (hotel
inventory at the £1000 cap). It ships *without* review scores so it can price
brand-new listings — that choice costs ~£6.5/night of accuracy, which we measured.

---

## Data

See [`data/README.md`](data/README.md). Two sources:

* **Inside Airbnb — London**, snapshot 19 June 2026 (listings + descriptions +
  coordinates + the forward-availability calendar). Git-ignored; re-downloadable.
* **OpenStreetMap** extracts (stations, parks, food venues) — *committed* to the
  repo as GeoPackages (ODbL, © OpenStreetMap contributors) so the notebooks
  reproduce without hitting the Overpass API.
* _Planned:_ Kaggle `justinas/housing-in-london` for the 25-year borough price
  trend (notebook 05).

## Layout

```
data/            raw -> interim -> processed  (git-ignored; OSM extracts kept)
notebooks/       the narrative: EDA, geo features, the model
src/londonrent/  reusable code
    data.py        load + clean the Inside Airbnb snapshot
    geo.py         lat/long -> location features (with the CRS explanation)
    features.py    assemble the model table (amenities, property type, scope)
    model.py       encoding, the estimator, spatial CV, metrics, save/load
app/             the Streamlit demo
models/          the saved model the app loads
```

## A note on the "where this comes from" citations

The notebooks attribute techniques to widely-used public sources — *An
Introduction to Statistical Learning* (ISLR), Géron's *Hands-On Machine Learning*,
StatQuest, Christoph Molnar's *Interpretable ML*, the scikit-learn user guide, and
a few papers. They're where these ideas are *commonly* learned, not a claim about
any specific course. Swap in your own reading where you'd rather.
