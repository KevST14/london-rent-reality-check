"""Smoke test for the saved model artifact the Streamlit app loads.

Guards against a bundle that loads but predicts nonsense (wrong encoder, feature
order drift, a retrain that broke something). Uses the committed OpenStreetMap
extracts, so it runs in CI; skipped only if the bundle or extracts are missing.
"""

import numpy as np
import pandas as pd
import pytest

from londonrent import model as M
from londonrent.config import EXTERNAL, PROJECT_ROOT
from londonrent.features import AMENITY_FEATURES

BUNDLE = PROJECT_ROOT / "models" / "price_model.joblib"
_HAVE = BUNDLE.exists() and all(
    (EXTERNAL / f"osm_{n}.gpkg").exists() for n in ("stations", "parks", "food")
)

pytestmark = pytest.mark.skipif(not _HAVE, reason="model bundle or OSM extracts not present")


def _predict(borough, room_type, accommodates, bedrooms, lat, lon):
    from londonrent.geo import add_location_features, fetch_pois

    bundle = M.load_bundle(BUNDLE)
    row = {
        "neighbourhood_cleansed": borough,
        "room_type": room_type,
        "property_class": "rental_unit",
        "accommodates": float(accommodates),
        "bedrooms": float(bedrooms),
        "beds": float(bedrooms + 1),
        "bathrooms": 1.0,
        "bath_is_shared": 0.0,
        "amenity_count": 30.0,
        "latitude": lat,
        "longitude": lon,
    }
    for k in AMENITY_FEATURES:
        row[k] = 0
    one = add_location_features(pd.DataFrame([row]), fetch_pois())
    X = M.build_design_matrix(one, bundle["groups"], bundle["encoder"])
    return float(np.exp(bundle["model"].predict(X)[0]))


def test_bundle_has_expected_keys():
    bundle = M.load_bundle(BUNDLE)
    assert {"encoder", "model", "groups", "metrics"} <= set(bundle)
    assert "review_scores_rating" not in bundle["groups"]["numeric"]  # cold-start model


def test_central_two_bed_prices_in_a_sane_range():
    # a 2-bed entire flat near the middle of Westminster
    price = _predict("Westminster", "Entire home/apt", 4, 2, 51.4975, -0.1357)
    assert 120 < price < 700, price


def test_more_central_and_bigger_costs_more():
    outer = _predict("Croydon", "Private room", 1, 1, 51.372, -0.098)
    inner = _predict("Camden", "Entire home/apt", 4, 2, 51.539, -0.143)
    assert inner > outer
