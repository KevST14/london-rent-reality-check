"""Load and clean the Inside Airbnb London snapshot.

Raw files live in ``data/raw`` (git-ignored). The cleaned listings table is written
to ``data/interim/listings.parquet`` so notebooks don't re-parse the CSV every time.

Snapshot: London, 2026-06-19 (https://insideairbnb.com/get-the-data/).
"""

from __future__ import annotations

import pandas as pd

from .config import INTERIM, LONDON_BBOX, RAW

LISTINGS_RAW = RAW / "listings.csv.gz"
CALENDAR_RAW = RAW / "calendar.csv.gz"
LISTINGS_INTERIM = INTERIM / "listings.parquet"

# Columns we carry forward from the 90-column raw file. Everything else is dropped
# in cleaning; add back here if a notebook needs it.
_KEEP = [
    "id",
    "last_scraped",
    "name",
    "description",
    "neighborhood_overview",
    "host_id",
    # This snapshot leaves `host_since` blank and instead ships the host's tenure
    # pre-computed as years + months; we fold those into a single `host_years`.
    "hosts_time_as_host_years",
    "hosts_time_as_host_months",
    "host_is_superhost",
    "host_listings_count",
    "host_identity_verified",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "amenities",
    "price",
    "minimum_nights",
    "maximum_nights",
    "has_availability",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_location",
    "estimated_occupancy_l365d",
    "first_review",
    "last_review",
]

_DATE_COLS = ["last_scraped", "first_review", "last_review"]
_TF_COLS = ["host_is_superhost", "host_identity_verified", "has_availability"]


def _money_to_float(s: pd.Series) -> pd.Series:
    """'$1,234.50' -> 1234.5 ; blanks/NaN -> NaN."""
    return pd.to_numeric(
        s.astype("string").str.replace(r"[$,]", "", regex=True).str.strip(),
        errors="coerce",
    )


def _parse_bathrooms(text: pd.Series) -> pd.DataFrame:
    """'1.5 shared baths' -> (1.5, True). 'Half-bath' -> (0.5, False)."""
    t = text.astype("string").str.lower().fillna("")
    num = t.str.extract(r"([\d.]+)")[0].astype(float)
    num = num.where(~t.str.contains("half"), 0.5)
    return pd.DataFrame({"bathrooms": num, "bath_is_shared": t.str.contains("shared")})


def load_listings_raw() -> pd.DataFrame:
    """Read the raw listings CSV with no cleaning."""
    return pd.read_csv(LISTINGS_RAW, low_memory=False)


def clean_listings(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a tidy, analysis-ready listings frame.

    Steps: subset columns, parse money/dates/booleans, parse bathrooms text,
    drop listings outside the London bounding box, drop exact-duplicate ids.
    Rows with a missing ``price`` are kept here (67% coverage) so EDA can see
    them; drop them at model time.
    """
    if df is None:
        df = load_listings_raw()

    df = df[[c for c in _KEEP if c in df.columns]].copy()

    df["price"] = _money_to_float(df["price"])
    for c in _DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in _TF_COLS:
        df[c] = df[c].map({"t": True, "f": False}).astype("boolean")

    df[["bathrooms", "bath_is_shared"]] = _parse_bathrooms(df["bathrooms_text"])
    df["amenity_count"] = (
        df["amenities"].astype("string").str.count(",").add(1).where(df["amenities"].notna())
    )
    # How long this person has been a host, as a single decimal number of years.
    df["host_years"] = (
        df["hosts_time_as_host_years"].fillna(0) + df["hosts_time_as_host_months"].fillna(0) / 12
    ).where(df["hosts_time_as_host_years"].notna() | df["hosts_time_as_host_months"].notna())
    df = df.drop(columns=["hosts_time_as_host_years", "hosts_time_as_host_months"])

    min_lon, min_lat, max_lon, max_lat = LONDON_BBOX
    in_box = df["longitude"].between(min_lon, max_lon) & df["latitude"].between(min_lat, max_lat)
    df = df.loc[in_box].drop_duplicates(subset="id").reset_index(drop=True)

    return df


def build_interim(force: bool = False) -> pd.DataFrame:
    """Clean the raw listings once and cache to parquet. Returns the frame."""
    if LISTINGS_INTERIM.exists() and not force:
        return pd.read_parquet(LISTINGS_INTERIM)
    df = clean_listings()
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LISTINGS_INTERIM, index=False)
    return df


def load_calendar(usecols: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Read the forward-availability calendar (~34M rows, one year ahead).

    Columns: listing_id, date, available ('t'/'f'), minimum_nights, maximum_nights.
    Note: this snapshot's calendar has no price column.
    """
    cal = pd.read_csv(CALENDAR_RAW, usecols=usecols, nrows=nrows, low_memory=False)
    if "date" in cal:
        cal["date"] = pd.to_datetime(cal["date"])
    if "available" in cal:
        cal["is_booked"] = cal["available"].map({"t": False, "f": True})
    return cal


if __name__ == "__main__":
    out = build_interim(force=True)
    print(f"wrote {LISTINGS_INTERIM}  shape={out.shape}")
    print(out[["price", "bedrooms", "room_type", "neighbourhood_cleansed"]].describe(include="all"))
