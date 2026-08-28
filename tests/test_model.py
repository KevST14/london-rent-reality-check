"""Tests for londonrent.model - the CV splitter and the metrics."""

import numpy as np

from londonrent.model import evaluate, spatial_block_folds


def _grid(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(51.30, 51.65, n)
    lon = rng.uniform(-0.45, 0.20, n)
    return lat, lon


def test_spatial_folds_partition_every_row_exactly_once():
    lat, lon = _grid()
    folds = spatial_block_folds(lat, lon, n_splits=5, seed=0)
    test_sets = [set(te.tolist()) for _, te in folds]

    # pairwise disjoint
    for i in range(len(test_sets)):
        for j in range(i + 1, len(test_sets)):
            assert test_sets[i].isdisjoint(test_sets[j])
    # cover everything
    assert set().union(*test_sets) == set(range(len(lat)))


def test_spatial_folds_no_grid_cell_spans_train_and_test():
    lat, lon = _grid()
    block = 0.02
    cell = (
        np.floor(lat / block).astype(int).astype(str)
        + "_"
        + np.floor(lon / block).astype(int).astype(str)
    )
    for tr, te in spatial_block_folds(lat, lon, n_splits=5, seed=0):
        assert set(cell[tr]).isdisjoint(set(cell[te]))


def test_evaluate_perfect_prediction():
    y = np.log(np.array([80.0, 150.0, 400.0, 250.0]))
    m = evaluate(y, y.copy())
    assert m["MAE_gbp"] < 1e-6
    assert m["R2_log"] > 0.999
    assert m["within_15pct"] == 1.0
    assert set(m) == {"MAE_gbp", "RMSE_gbp", "MdAPE", "within_15pct", "MAE_log", "R2_log"}


def test_evaluate_reports_pounds_after_unlogging():
    y_true = np.log(np.array([100.0, 200.0]))
    y_pred = np.log(np.array([110.0, 180.0]))  # £10 and £20 off
    m = evaluate(y_true, y_pred)
    assert abs(m["MAE_gbp"] - 15.0) < 1e-6
