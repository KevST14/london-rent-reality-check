"""Tests for londonrent.features — the pure transforms and the scope rules."""

import pandas as pd
import pytest

from londonrent.data import LISTINGS_RAW
from londonrent.features import (
    DROP_PROPERTY_CLASSES,
    PRICE_MAX,
    PRICE_MIN,
    PROPERTY_CLASSES,
    add_amenity_flags,
    build_model_frame,
    collapse_property_type,
    parse_amenities,
)


def test_collapse_property_type_buckets():
    s = pd.Series(
        [
            "Entire rental unit",
            "Private room in townhouse",
            "Room in boutique hotel",
            "Entire condo",
            "Private room in castle",
        ]
    )
    out = collapse_property_type(s).tolist()
    assert out == ["rental_unit", "townhouse", "hotel", "condo", "other"]


def test_parse_amenities_handles_bad_json():
    s = pd.Series(['["Wifi", "Kitchen"]', "not json at all", None])
    out = parse_amenities(s).tolist()
    assert out[0] == ["wifi", "kitchen"]
    assert out[1] == [] and out[2] == []


def test_add_amenity_flags_substring_match():
    df = pd.DataFrame(
        {"amenities": ['["Dishwasher", "Shared gym in building", "Bergamot shower gel"]']}
    )
    out = add_amenity_flags(df)
    assert out["amen_dishwasher"].iloc[0] == 1
    assert out["amen_gym"].iloc[0] == 1  # "shared gym ..." still counts
    assert out["amen_pool"].iloc[0] == 0


def test_property_classes_constant_excludes_dropped():
    for dropped in DROP_PROPERTY_CLASSES:
        assert dropped not in PROPERTY_CLASSES
    assert "rental_unit" in PROPERTY_CLASSES


@pytest.mark.skipif(not LISTINGS_RAW.exists(), reason="raw Inside Airbnb CSV not present")
def test_build_model_frame_respects_scope():
    frame, groups = build_model_frame(with_reviews=False)
    assert frame["price"].between(PRICE_MIN, PRICE_MAX).all()
    assert not frame["property_class"].isin(DROP_PROPERTY_CLASSES).any()
    assert {"categorical", "numeric"} == set(groups)
    assert "log_price" in frame.columns
    # cold-start frame must not carry the review/host columns
    assert "review_scores_rating" not in groups["numeric"]
