"""Assemble the table the price model trains on.

This module takes the lightly-cleaned listings from :mod:`londonrent.data`, bolts
on the location features from :mod:`londonrent.geo`, turns the messy ``amenities``
text into a handful of yes/no columns, tidies ``property_type``, and hands back a
model-ready frame plus the lists of which columns are which.

Design choice worth stating: we keep the *feature engineering* here (deterministic,
testable, reusable by the Streamlit app) and leave the *modelling* (encoding,
the estimator, tuning) to :mod:`londonrent.model`. Mixing the two is the usual way
notebooks rot.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from .data import build_interim
from .geo import LOCATION_FEATURES, add_location_features, fetch_pois

# --- scope (the Week-1 decisions, in code) ----------------------------------
PRICE_MIN, PRICE_MAX = 10, 1000
KEEP_ROOM_TYPES = ("Entire home/apt", "Private room")

# --- amenities -------------------------------------------------------------
# Inside Airbnb lists ~7,800 distinct amenity strings, most of them near-universal
# ("wifi") or one-offs. Rather than one-hot all of them, we hand-pick amenities
# that plausibly signal a pricier listing, and match them as substrings (so
# "Shared gym in building" and "Private gym" both count as gym). This is a
# judgement call: a purely data-driven top-N is in the notebook as a comparison.
PREMIUM_AMENITIES: dict[str, tuple[str, ...]] = {
    "amen_dishwasher": ("dishwasher",),
    "amen_dryer": ("dryer",),
    "amen_aircon": ("air conditioning", "portable air conditioning"),
    "amen_lift": ("elevator",),
    "amen_free_parking": ("free parking on premises", "free driveway parking", "free carport"),
    "amen_gym": ("gym",),
    "amen_pool": ("pool",),
    "amen_hot_tub": ("hot tub",),
    "amen_balcony_patio": ("patio or balcony", "balcony", "private patio"),
    "amen_bathtub": ("bathtub",),
    "amen_workspace": ("dedicated workspace",),
    "amen_self_checkin": ("self check-in",),
    "amen_pets_allowed": ("pets allowed",),
    "amen_ev_charger": ("ev charger",),
    "amen_waterfront_view": ("waterfront", "river view", "canal view", "lake view"),
    "amen_doorman": ("doorman", "building staff"),
}
AMENITY_FEATURES = list(PREMIUM_AMENITIES)


def parse_amenities(raw: pd.Series) -> pd.Series:
    """Turn the JSON-ish amenity strings into lists of lowercase strings."""

    def _one(s):
        if not isinstance(s, str):
            return []
        try:
            return [a.strip().lower() for a in ast.literal_eval(s)]
        except (ValueError, SyntaxError):
            return []

    return raw.map(_one)


def add_amenity_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add one 0/1 column per entry in :data:`PREMIUM_AMENITIES`.

    We test each needle as a substring of each amenity string, working on the
    parsed lists directly. (A few amenity strings in the raw data contain broken
    unicode, which is why we don't first glue them into one big string.)
    """
    out = df.copy()
    lists = parse_amenities(out["amenities"])
    for col, needles in PREMIUM_AMENITIES.items():
        out[col] = lists.map(
            lambda xs, _n=needles: int(any(n in x for x in xs for n in _n))
        ).astype("int8")
    return out


# --- property type ------------------------------------------------------------
# ~100 raw values ("Entire rental unit", "Private room in townhouse", ...). The
# room-type column already says entire-vs-private, so here we only keep the
# *building* kind, collapsed to a handful.
_PROPERTY_MAP = {
    "rental unit": "rental_unit",
    "condo": "condo",
    "home": "house",
    "townhouse": "townhouse",
    "serviced apartment": "serviced_apartment",
    "loft": "loft",
    "bed and breakfast": "bnb_guesthouse",
    "guesthouse": "bnb_guesthouse",
    "guest suite": "guest_suite",
    "hotel": "hotel",
    "boutique hotel": "hotel",
}


def collapse_property_type(raw: pd.Series) -> pd.Series:
    low = raw.str.lower().fillna("")
    out = pd.Series("other", index=raw.index, dtype="object")
    for needle, label in _PROPERTY_MAP.items():
        out[low.str.contains(needle)] = label
    return out


# --- put it together --------------------------------------------------------
CAT_FEATURES = ["neighbourhood_cleansed", "room_type", "property_class"]
STRUCT_FEATURES = ["accommodates", "bedrooms", "beds", "bathrooms", "bath_is_shared", "amenity_count"]
REVIEW_HOST_FEATURES = [
    "number_of_reviews",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_location",
    "host_is_superhost",
    "host_years",
    "host_listings_count",
    "minimum_nights",
    "availability_365",
]


def build_model_frame(
    df: pd.DataFrame | None = None,
    *,
    with_reviews: bool = True,
    keep_text: bool = False,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return ``(frame, feature_groups)`` ready for :mod:`londonrent.model`.

    ``frame`` has every feature column plus ``price`` and ``log_price``.
    ``feature_groups`` maps ``"categorical"`` / ``"numeric"`` to column-name lists
    so the caller can build the right encoder.

    ``with_reviews=False`` drops the review/host columns. That gives the
    "cold-start" model — what you'd use to price a brand-new listing that has no
    reviews yet — and lets the notebook measure how much the review signals are
    really worth (and flag the mild leakage risk they carry).

    ``keep_text=True`` also carries the raw ``name`` and ``description`` columns
    through (unused by the model, but notebook 04 needs them for the text pillar).
    """
    if df is None:
        df = build_interim()

    df = df[
        df["price"].between(PRICE_MIN, PRICE_MAX)
        & df["room_type"].isin(KEEP_ROOM_TYPES)
    ].copy()
    df["log_price"] = np.log(df["price"])

    df = add_location_features(df, fetch_pois())
    df = add_amenity_flags(df)
    df["property_class"] = collapse_property_type(df["property_type"])

    # booleans -> 0/1 so the passthrough numeric branch can take them
    for col in ["host_is_superhost", "bath_is_shared"]:
        df[col] = df[col].astype("float").fillna(0.0)

    numeric = STRUCT_FEATURES + LOCATION_FEATURES + AMENITY_FEATURES
    if with_reviews:
        numeric = numeric + REVIEW_HOST_FEATURES

    groups = {"categorical": CAT_FEATURES, "numeric": numeric}
    keep = CAT_FEATURES + numeric + ["price", "log_price", "latitude", "longitude", "id"]
    if keep_text:
        keep = keep + [c for c in ("name", "description") if c in df.columns]
    return df[keep].reset_index(drop=True), groups


if __name__ == "__main__":
    frame, groups = build_model_frame()
    print("frame:", frame.shape)
    print("categorical:", groups["categorical"])
    print("numeric   :", groups["numeric"])
    print(frame[groups["numeric"]].describe().T.round(2).to_string())
