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

## What I tried that did not work (or barely did)

Portfolio projects usually only show the happy path. Here is the rest.

- **Hyperparameter tuning.** I expected a randomised search to buy a point or two
  of R-squared. It bought nothing: the best settings scored the same as the
  defaults, within noise. The model already stops early, which does most of the
  regularising. I kept the tuned params (they cost nothing) and reported the zero
  gain rather than hunting for a seed that made tuning look good.

- **Using the description text.** The hypothesis was that hosts who write glossy
  copy charge a premium the structured fields miss. Ten hand-crafted "style"
  features plus 50 TF-IDF word-topics added about £1 to £2 per night on top of the
  facts. Almost all of what the text "knows" is just the words leaking the facts
  ("room" vs "townhouse", borough names). Useful negative result: the deployed
  model is text-free.

- **`neighborhood_overview` as a second text field.** Planned to use it alongside
  `description`. It is completely empty in this data snapshot, so notebook 04 uses
  `description` alone. Found this out in the first EDA pass rather than halfway
  through the text work.

- **Keeping every property type.** The first model kept hotel rooms sold through
  Airbnb. The SHAP error analysis in notebook 03 kept surfacing them as the worst
  individual misses (they price on hotel logic, not home logic). I went back to
  the scope decision and cut them: about 1,200 listings, and the held-out metrics
  barely moved, but the question got cleaner.

- **The £1000 price cap.** Dropping hotels removed the hotel-room edge cases, but
  the hard cap still produces new ones: the current worst miss is a one-person
  whole flat in Hounslow priced at £1000. A proper fix (model the tail separately,
  or a quantile loss) is on the "next" list.

- **The first OpenStreetMap station query.** `railway=station` on its own misses
  Underground-only stops. The distance-to-station feature was quietly wrong until I
  added `station=subway` and light-rail to the query.

- **Euclidean distance on latitude and longitude.** The first
  distance-to-station numbers were nonsense. Near London, one degree of longitude
  is about 43 km on the ground and one degree of latitude is about 111 km, so
  `sqrt(dlat^2 + dlon^2)` is meaningless. Fixed by projecting to the British
  National Grid (metres) before any distance maths.

- **Park distance as distance-to-centroid.** Measuring to a park's centre means a
  listing right next to a huge park can look "far" from it. I shipped the simple
  version and flagged it; distance-to-polygon-boundary is the fix.

## Two deeper looks (notebook 03b)

- **Partial dependence turned up a surprise.** Once the model knows how central a
  listing is, the average marginal effect of walking distance to a *station* is
  close to flat. Station distance still matters for individual listings (SHAP
  shows that), but the model has folded most of the "how connected is this"
  signal into distance-to-centre, because the two features overlap. SHAP shows
  attribution; partial-dependence shows the shape, and here they say different
  things.
- **A naive "biggest mispricing" scan mostly finds junk.** Listing every
  listing's predicted-minus-actual gap and taking the extremes surfaces
  data-entry artefacts (single rooms listed at £475 a night, Westminster flats at
  £45 a night), not real bargains. A useful mispricing detector would need
  per-prediction confidence intervals and a plausibility filter on the listing.

## What I would do next

- **Finish the third pillar: a 25-year borough price trend and forecast** using the
  Kaggle "Housing in London" dataset, with a walk-forward backtest against a
  seasonal-naive baseline.
- **Handle the price tail properly** instead of a hard £1000 cap: model the tail
  separately, or fit with a tail-robust loss (Huber, or quantile loss for
  prediction intervals). Quantile loss would also give the per-prediction
  intervals the mispricing scan needs.
- **Better geo features:** distance to a park's boundary instead of its centre, and
  distances to several centres (City, West End, Canary Wharf) instead of just
  Charing Cross.
- **Fit the text components inside each CV fold** rather than once on all rows, to
  remove the mild leakage in the current TF-IDF / SVD step.
- **Dig into Brent** specifically, since it is the one well-sampled borough the
  model handles badly.
