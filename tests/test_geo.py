"""Tests for londonrent.geo.

The distance maths is pure. `add_location_features` needs the OpenStreetMap
extracts, which ARE committed to the repo (data/external/osm_*.gpkg), so it runs
in CI too; it's skipped only if someone deleted them.
"""

import numpy as np
import pandas as pd
import pytest

from londonrent.config import EXTERNAL
from londonrent.geo import (
    CHARING_CROSS,
    LOCATION_FEATURES,
    _points_to_bng_xy,
    add_location_features,
    fetch_pois,
)

_HAVE_POIS = all((EXTERNAL / f"osm_{n}.gpkg").exists() for n in ("stations", "parks", "food"))


def test_bng_projection_distance_is_in_real_metres():
    # two points ~1 km apart (0.009° of latitude ≈ 1 km near London)
    xy = _points_to_bng_xy(np.array([51.50, 51.509]), np.array([-0.12, -0.12]))
    d = np.hypot(*(xy[0] - xy[1]))
    assert 950 < d < 1050, f"expected ~1000 m, got {d:.0f}"


def test_bng_longitude_shorter_than_latitude_for_same_degree_step():
    # 1° of longitude near London is much shorter on the ground than 1° of latitude
    lat_step = np.hypot(
        *(
            _points_to_bng_xy(np.array([51.5, 52.5]), np.array([0.0, 0.0]))[0]
            - _points_to_bng_xy(np.array([51.5, 52.5]), np.array([0.0, 0.0]))[1]
        )
    )
    lon_step = np.hypot(
        *(
            _points_to_bng_xy(np.array([51.5, 51.5]), np.array([0.0, 1.0]))[0]
            - _points_to_bng_xy(np.array([51.5, 51.5]), np.array([0.0, 1.0]))[1]
        )
    )
    assert lon_step < lat_step


@pytest.mark.skipif(not _HAVE_POIS, reason="OSM extracts not available")
def test_add_location_features_shapes_and_sanity():
    pois = fetch_pois()
    df = pd.DataFrame(
        {
            # 1: dead centre (Charing Cross). 2: far south-east edge.
            "latitude": [CHARING_CROSS[0], 51.36],
            "longitude": [CHARING_CROSS[1], 0.10],
        }
    )
    out = add_location_features(df, pois)

    for col in LOCATION_FEATURES:
        assert col in out.columns
        assert out[col].notna().all()

    # the central listing must be closer to the centre and have more stations near
    assert out["dist_center_m"].iloc[0] < 500
    assert out["dist_center_m"].iloc[1] > out["dist_center_m"].iloc[0]
    assert out["stations_within_1km"].iloc[0] >= out["stations_within_1km"].iloc[1]
    # distances are non-negative metres
    assert (out[["dist_station_m", "dist_restaurant_m", "dist_park_m"]] >= 0).all().all()
