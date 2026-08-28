"""Tests for the raw-data parsing helpers in londonrent.data.

These are pure functions over small hand-built Series, so they need no data files
and run anywhere.
"""

import pandas as pd

from londonrent.data import _money_to_float, _parse_bathrooms


def test_money_to_float_parses_currency_strings():
    got = _money_to_float(pd.Series(["$1,234.50", "$80.00", "$2", "", None]))
    assert got.tolist()[:3] == [1234.5, 80.0, 2.0]
    assert pd.isna(got.iloc[3]) and pd.isna(got.iloc[4])


def test_parse_bathrooms_number_and_shared_flag():
    out = _parse_bathrooms(pd.Series(["1.5 shared baths", "1 bath", "2 baths", "Half-bath", None]))
    assert list(out["bathrooms"][:4]) == [1.5, 1.0, 2.0, 0.5]
    assert list(out["bath_is_shared"]) == [True, False, False, False, False]


def test_parse_bathrooms_missing_text_gives_nan_not_crash():
    out = _parse_bathrooms(pd.Series([None, ""]))
    assert out["bathrooms"].isna().all()
    assert (~out["bath_is_shared"]).all()
