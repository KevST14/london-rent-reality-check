# Findings

The one-page version. The full working is in the four notebooks.

## The question

What is a fair nightly price for a London short-let, and can a model say *why* it
priced a listing the way it did? Data: about 93,000 Inside Airbnb listings (London,
June 2026), plus stations, parks and food venues from OpenStreetMap.

## What the model does

- Predicts `log(price)` for whole homes and private rooms priced £10 to £1000 per
  night, minus about 1,200 hotel rooms sold through Airbnb.
- Held-out test (whole areas of London kept out of training): **mean error about
  £60 per night, median error about 22%, R-squared 0.72 on log price.**
- For comparison, guessing the borough-and-room-type median gets mean error about
  £90 / R-squared 0.49. The model roughly halves the error of a sensible lookup
  table.
- Every prediction comes with a SHAP breakdown of which features moved it.

## Five things worth knowing

**1. Exact location adds real signal on top of the borough, but not a lot.**
Six OpenStreetMap features (distance to a station, to the centre, to food, to a
park, plus two counts) lift R-squared from 0.62 to 0.65 and cut mean error from
£78 to £72. The gain is modest because the borough label already captures the
coarse geography; these features add the within-borough part, which is a smaller
slice.

**2. The wording of the description barely helps once you have the facts.**
The description on its own predicts price at R-squared 0.46 (about the no-model
baseline). But almost all of that is the words leaking the facts: "room" vs
"townhouse", borough names, "studio" vs "family". Add text features on top of the
structured model and the error moves by only about £1 to £2 per night. So the
deployed model is text-free, which keeps it simpler and cheaper to run. This is a
useful negative result: a pricing tool for this data does not need NLP.
One thing did survive: descriptions that lean on marketing words ("stunning",
"luxurious") really are pricier (correlation 0.24), but mostly because those flats
are also bigger and more central.

**3. A naive random cross-validation split only flattered the score by 0.02
R-squared.** Nearby Airbnbs are near-duplicates, so a random split lets the model
study for its own test. Switching to spatial cross-validation (hold out whole 2 km
blocks) is the honest thing to do, and it cost only about 0.02 R-squared and £2.70
of mean error here. That small gap is itself informative: spatial autocorrelation
in this dataset is mild at the neighbourhood scale. On data with strong grouping
(repeated users, a handful of sites, daily time series) the same experiment can
move the metric ten times as much.

**4. Hyperparameter tuning did nothing.** A randomised search over the main knobs
landed on settings that scored the same as the defaults, within noise.
`HistGradientBoostingRegressor` with early stopping already regularises itself.
Worth stating plainly rather than presenting as a win.

**5. The worst individual errors are data-quality edge cases, not a broken model.**
After dropping hotel rooms, the single largest miss is a £1000-per-night whole flat
that sleeps one person, in Hounslow near Heathrow. That is the £1000 price cap
catching a listing that is priced like a luxury home but is not one. The model is
also weakest at the top of the price range in general (a known side effect of a
logged target and a thin luxury sample) and in thinly-sampled outer boroughs.
Brent stands out: it is well sampled (555 test listings) yet still predicted at 25%
median error, so it would be the first place to dig further.

## What I would do next

- **Finish the third pillar: a 25-year borough price trend and forecast** using the
  Kaggle "Housing in London" dataset, with a proper walk-forward backtest against a
  seasonal-naive baseline.
- **Handle the price tail properly** instead of a hard £1000 cap: either model the
  tail separately, or fit with a tail-robust loss (Huber, or quantile loss for
  prediction intervals).
- **Better geo features:** distance to a park's boundary instead of its centre, and
  distances to several centres (City, West End, Canary Wharf) instead of just
  Charing Cross.
- **Fit the text components inside each CV fold** rather than once on all rows, to
  remove the mild leakage in the current TF-IDF / SVD step.
- **Partial-dependence and ICE plots** for the top features, to complement SHAP:
  SHAP shows attribution, PDP shows the shape of the relationship.
- **Dig into Brent** specifically, since it is the one well-sampled borough the
  model handles badly.
