"""The price model: encoding, the estimator, honest cross-validation, metrics.

Everything here operates on the frame produced by
:func:`londonrent.features.build_model_frame`.

Two things a reader should know up front:

* **We predict ``log_price`` and convert back with ``exp``.** All the £-denominated
  metrics below un-log the predictions first, so "MAE = £48" means 48 real pounds.
* **The estimator is ``HistGradientBoostingRegressor``.** It handles missing values
  by itself (it learns which way to send a NaN at each split), so we do *not*
  impute ``bedrooms`` / ``review_scores`` - the gaps are passed straight through.
  That's a real advantage of this model family; a linear model would need the gaps
  filled first.

  > Native NaN handling in histogram gradient boosting: see the scikit-learn
  > user guide, "Histogram-Based Gradient Boosting", and LightGBM (Ke et al. 2017),
  > which introduced the approach.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

# A modest, hand-set configuration. Notebook 03 tunes around this; these are the
# defaults it starts from and the ones the saved app model falls back to.
DEFAULT_PARAMS: dict = {
    "learning_rate": 0.05,  # how big a step each new tree takes; smaller = steadier
    "max_leaf_nodes": 31,  # complexity of each individual tree
    "min_samples_leaf": 40,  # don't split down to tiny groups -> less overfitting
    "l2_regularization": 1.0,
    "max_iter": 600,  # up to 600 trees...
    "early_stopping": True,  # ...but stop once a held-out slice stops improving
    "validation_fraction": 0.1,
    "n_iter_no_change": 30,
    "random_state": 0,
}


# --- encoding --------------------------------------------------------------
def fit_encoder(frame: pd.DataFrame, cat_features: list[str]) -> OneHotEncoder:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype="float32")
    enc.fit(frame[cat_features])
    return enc


def build_design_matrix(
    frame: pd.DataFrame, groups: dict[str, list[str]], encoder: OneHotEncoder
) -> pd.DataFrame:
    """One-hot the categoricals, glue the numeric columns on unchanged.

    Returns a plain dense DataFrame with readable column names (e.g.
    ``neighbourhood_cleansed=Camden``) so SHAP plots later are legible.
    """
    cat, num = groups["categorical"], groups["numeric"]
    cat_arr = encoder.transform(frame[cat])
    cat_cols = encoder.get_feature_names_out(cat)
    cat_df = pd.DataFrame(cat_arr, columns=cat_cols, index=frame.index)
    return pd.concat([cat_df, frame[num].astype("float32")], axis=1)


def make_model(params: dict | None = None) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**{**DEFAULT_PARAMS, **(params or {})})


def make_quantile_model(
    quantile: float, params: dict | None = None
) -> HistGradientBoostingRegressor:
    """Same booster, but trained to predict a given quantile of log price instead
    of the mean. Fit one at 0.1 and one at 0.9 to get a rough 10-to-90 band.

    This gives an *empirical* band, not a calibrated guarantee. Notebook 03 checks
    what fraction of held-out prices actually land inside it.
    """
    cfg = {**DEFAULT_PARAMS, "loss": "quantile", "quantile": quantile, **(params or {})}
    return HistGradientBoostingRegressor(**cfg)


# --- honest cross-validation --------------------------------------------------
def spatial_block_folds(
    lat: np.ndarray, lon: np.ndarray, n_splits: int = 5, block_deg: float = 0.02, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cross-validation folds that keep nearby listings together.

    Why not plain random folds? Two listings on the same street are almost
    duplicates. Split them at random and the model gets to "study" one and be
    "tested" on its near-twin, so the score looks better than real life. Here we
    lay a ~2 km grid over London (``block_deg`` degrees ~ 1.4 km of latitude),
    give every listing its grid-cell id, and make sure a whole cell is either in
    training or in test for a given fold - never both.

    > Roberts et al. (2017), "Cross-validation strategies for data with
    > spatial, temporal, hierarchical, or phylogenetic structure", *Ecography* -
    > the standard reference for blocked/grouped CV under autocorrelation.
    """
    cell = (
        np.floor(lat / block_deg).astype(int).astype(str)
        + "_"
        + np.floor(lon / block_deg).astype(int).astype(str)
    )
    cells = pd.Series(cell)
    uniq = cells.drop_duplicates().sample(frac=1.0, random_state=seed).to_numpy()
    buckets = np.array_split(uniq, n_splits)

    folds = []
    idx = np.arange(len(lat))
    for b in buckets:
        test_mask = cells.isin(set(b)).to_numpy()
        folds.append((idx[~test_mask], idx[test_mask]))
    return folds


# --- metrics --------------------------------------------------------------
def evaluate(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> dict[str, float]:
    """Score a set of predictions, reported both in log space and in real £."""
    yt, yp = np.exp(y_log_true), np.exp(y_log_pred)
    abs_pct = np.abs(yp - yt) / yt
    ss_res = np.sum((y_log_true - y_log_pred) ** 2)
    ss_tot = np.sum((y_log_true - y_log_true.mean()) ** 2)
    return {
        "MAE_gbp": float(np.mean(np.abs(yp - yt))),
        "RMSE_gbp": float(np.sqrt(np.mean((yp - yt) ** 2))),
        "MdAPE": float(np.median(abs_pct)),  # median absolute % error
        "within_15pct": float(np.mean(abs_pct <= 0.15)),
        "MAE_log": float(np.mean(np.abs(y_log_true - y_log_pred))),
        "R2_log": float(1 - ss_res / ss_tot),
    }


# --- persistence --------------------------------------------------------------
def save_bundle(
    path: str | Path,
    *,
    encoder,
    model,
    groups: dict,
    metrics: dict,
    quantile_models: dict[float, object] | None = None,
) -> None:
    """Save everything the Streamlit app needs in one file.

    ``quantile_models`` is an optional ``{0.1: model, 0.9: model}`` for the rough
    prediction band; the app shows a point estimate if it is absent.
    """
    obj = {"encoder": encoder, "model": model, "groups": groups, "metrics": metrics}
    if quantile_models:
        obj["quantile_models"] = quantile_models
    dump(obj, path)


def load_bundle(path: str | Path) -> dict:
    return load(path)
