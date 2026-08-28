# Rent Reality Check — London

**Given a London short-let listing, predict a fair nightly price, explain the
number, and show where that area's prices have been heading.**

*Created by **Kevin Steepan**.*

A student portfolio project built on real-world data and a problem anyone can
research and observe. The aim is to show the **skills that turn into job
experience** — framing a real question, cleaning messy public data, engineering
features with a reason behind each one, validating honestly instead of
optimistically, and making the model explain itself — and to show the **reasoning**,
not just the final number.

Every notebook is written to be read by someone with no ML background *and* by
someone who wants into the field:

- a plain-English walk-through of what each step does and why,
- `###` sub-sections that go deeper for a student learning ML — the concept, why
  it works, the trade-offs, what to try next,
- a plain reading of every result, then a deeper note,
- a **"where this comes from"** pointer to the common public source for each
  technique (ISLR, Géron, StatQuest, Molnar's *Interpretable ML*, the scikit-learn
  guide, key papers),
- code comments that narrate *how the code was arrived at* — what was tried, what
  broke, why this choice.

**Status:** price model + explanations + text pillar done. Long-run time-series
pillar and the write-up are next.

---

## What's in here

| Notebook | Question | Headline result |
|---|---|---|
| [`01_eda`](notebooks/01_eda.ipynb) | What are we predicting, on which listings, and what's the score to beat? | Target = `log(price)`, scope = £10–1000/night entire-home + private-room. No-model baseline: **£-MAE ≈ 90 / R² ≈ 0.49**. |
| [`02_geo_features`](notebooks/02_geo_features.ipynb) | Does the exact spot matter, beyond the borough label? | Yes. Six OpenStreetMap features (distance to Tube / centre / food / park) lift R² 0.60 → **0.64** and cut error £78 → **£73**. |
| [`03_price_model`](notebooks/03_price_model.ipynb) | Build, check, tune, and **explain** the real model. | Held-out test: **£-MAE ≈ £60 · median error ≈ 21% · R²(log) ≈ 0.71**. Beats a linear baseline by ~£9. SHAP explains every prediction. |
| [`04_text`](notebooks/04_text.ipynb) | Does the *wording* of the description predict price on top of the facts? | Barely. Text alone gets R² ≈ 0.45, but on top of the facts it adds only **~£1.7 / +0.01 R²** — a useful *negative* result (the app keeps the facts-only model). Marketing words do track price (corr +0.24). |
| _`05` (planned)_ | Where have London prices been heading? (25-year borough trend) | — |

**Live demo:** [`app/streamlit_app.py`](app/streamlit_app.py) — three tabs:
*Price a listing* (prediction + a plain verdict on a price you enter + a SHAP
breakdown of what drove it), *How it works* (the five-step method in plain
language), and *How accurate is it* (the held-out test numbers).

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
StatQuest, Christoph Molnar's *Interpretable ML* and the scikit-learn user guide.
