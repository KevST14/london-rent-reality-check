"""Turn a latitude/longitude into features that describe *where* a listing is.

Why this module exists
----------------------
The borough name (`neighbourhood_cleansed`) is a very coarse description of location.
Two flats in the same borough can be a 3-minute walk from a Tube station or a
25-minute walk from one, and in London that gap is worth real money. This module
adds the finer-grained "how well connected / how lively is this exact spot"
signal that a borough label can't capture.

The features we build, and the intuition for each
-------------------------------------------------
* ``dist_station_m``        - metres to the nearest rail/Tube/DLR station.
                              Closer = more convenient for guests = commands more.
* ``stations_within_1km``   - how many stations are in easy reach (transport choice).
* ``dist_center_m``         - metres to Charing Cross, the traditional "centre of
                              London". A blunt proxy for "how touristy / central".
* ``dist_park_m``           - metres to the nearest green space (0 if inside one).
* ``dist_restaurant_m`` /   - nearest, and count within 500 m, of
  ``food_within_500m``        restaurants + cafes + pubs + bars: a "local buzz" proxy.

We deliberately skip schools/GP surgeries here. For a *long-term* rental those
matter a lot; for a *short-let* (a few nights) guests don't care, so including
them would just add noise. Choosing features to fit the problem is part of the
work.

The one concept to understand first: coordinate reference systems (CRS)
----------------------------------------------------------------------
Latitude/longitude are *angles* on a globe, measured in degrees. You cannot do
ordinary "flat" geometry on them: near London, moving 1 degree of longitude is
about 43 km, but 1 degree of latitude is about 111 km, so the same number of
degrees means a different real-world distance in each direction. Pythagoras on
raw lat/long is simply wrong.

The fix is to *project* the coordinates onto a flat grid whose units are metres.
For the UK the standard grid is the **British National Grid (EPSG:27700)**. Over
an area the size of London, plain straight-line distance on that grid is accurate
to well under 1%, which is plenty here. So the pattern throughout this module is:

    load points as EPSG:4326 (lat/long)  ->  .to_crs(27700)  ->  do metre maths
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.spatial import cKDTree

from .config import EXTERNAL, LONDON_BBOX

# EPSG codes: 4326 is "raw" GPS latitude/longitude; 27700 is the metre-based
# British National Grid we project onto before measuring anything.
WGS84 = 4326
BNG = 27700

# Charing Cross - by convention, distances "to London" are measured from here.
CHARING_CROSS = (51.5074, -0.1278)  # (lat, lon)

# What to ask OpenStreetMap for. Keys/values are OSM "tags" - OSM describes every
# feature with tags like `railway=station` or `amenity=cafe`.
_OSM_TAGS = {
    "stations": {"railway": ["station", "halt"], "station": ["subway", "light_rail"]},
    "parks": {"leisure": ["park", "garden", "nature_reserve", "common"]},
    "food": {"amenity": ["restaurant", "cafe", "pub", "bar", "fast_food"]},
}


def _cache_path(name: str):
    return EXTERNAL / f"osm_{name}.gpkg"


def fetch_pois(force: bool = False) -> dict[str, gpd.GeoDataFrame]:
    """Download the OSM points of interest we need and cache them to disk.

    OpenStreetMap is queried through a public server (Overpass). That's slow and
    rate-limited, so we save each result as a GeoPackage in ``data/external`` and
    reuse it on later runs. Delete those files (or pass ``force=True``) to refresh.
    """
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    out: dict[str, gpd.GeoDataFrame] = {}
    for name, tags in _OSM_TAGS.items():
        path = _cache_path(name)
        if path.exists() and not force:
            out[name] = gpd.read_file(path)
            continue
        # osmnx 2.x bbox order is (left, bottom, right, top) = LONDON_BBOX's order
        gdf = ox.features_from_bbox(bbox=LONDON_BBOX, tags=tags)
        # Keep it small and serialisable: just the geometry and a name if OSM has one.
        # (OSM's own id is a MultiIndex here and we don't need it, so we drop it.)
        keep = ["geometry"] + (["name"] if "name" in gdf.columns else [])
        gdf = gdf[gdf.geometry.notna()].to_crs(WGS84).reset_index(drop=True)[keep]
        gdf.to_file(path, driver="GPKG")
        out[name] = gdf
    return out


def _points_to_bng_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """(lat, lon) in degrees -> (N, 2) array of (easting, northing) in metres."""
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=WGS84).to_crs(BNG)
    return np.c_[pts.x.to_numpy(), pts.y.to_numpy()]


def _nearest_and_count(listing_xy: np.ndarray, poi_xy: np.ndarray, radius_m: float):
    """For each listing point: distance to the nearest POI, and how many POIs
    lie within ``radius_m``.

    We use a KD-tree: a data structure that recursively splits the plane so that
    "find the closest point" costs about log(N) checks instead of comparing
    against every one of the tens of thousands of POIs. ``query`` gives the
    nearest; ``query_ball_point`` gives everything inside a radius.
    """
    tree = cKDTree(poi_xy)
    nearest_dist, _ = tree.query(listing_xy, k=1)
    counts = tree.query_ball_point(listing_xy, r=radius_m, return_length=True)
    return nearest_dist, np.asarray(counts)


def add_location_features(
    df: pd.DataFrame, pois: dict[str, gpd.GeoDataFrame] | None = None
) -> pd.DataFrame:
    """Return ``df`` with the location feature columns added.

    ``df`` must have ``latitude`` and ``longitude`` columns (degrees).
    """
    if pois is None:
        pois = fetch_pois()

    out = df.copy()
    listing_xy = _points_to_bng_xy(out["latitude"].to_numpy(), out["longitude"].to_numpy())

    # POI geometries can be points (a cafe) or polygons (a park). Reduce each to a
    # single representative point so the KD-tree has something to index. For parks
    # this means we measure to the park's centre, not its nearest edge - a known
    # simplification we note in the notebook.
    def xy_of(name: str) -> np.ndarray:
        g = pois[name].to_crs(BNG)
        reps = g.geometry.representative_point()
        return np.c_[reps.x.to_numpy(), reps.y.to_numpy()]

    d_station, n_station = _nearest_and_count(listing_xy, xy_of("stations"), radius_m=1000)
    out["dist_station_m"] = d_station.round(1)
    out["stations_within_1km"] = n_station

    d_food, n_food = _nearest_and_count(listing_xy, xy_of("food"), radius_m=500)
    out["dist_restaurant_m"] = d_food.round(1)
    out["food_within_500m"] = n_food

    d_park, _ = _nearest_and_count(listing_xy, xy_of("parks"), radius_m=500)
    out["dist_park_m"] = d_park.round(1)

    centre_xy = _points_to_bng_xy(np.array([CHARING_CROSS[0]]), np.array([CHARING_CROSS[1]]))
    out["dist_center_m"] = np.hypot(
        listing_xy[:, 0] - centre_xy[0, 0], listing_xy[:, 1] - centre_xy[0, 1]
    ).round(1)

    return out


LOCATION_FEATURES = [
    "dist_station_m",
    "stations_within_1km",
    "dist_restaurant_m",
    "food_within_500m",
    "dist_park_m",
    "dist_center_m",
]


if __name__ == "__main__":
    from .data import build_interim

    pois = fetch_pois()
    for k, v in pois.items():
        print(f"{k:10s} {len(v):>7,} features")
    sample = build_interim().head(2000)
    feat = add_location_features(sample, pois)
    print(feat[LOCATION_FEATURES].describe().round(1))
