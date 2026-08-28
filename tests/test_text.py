"""Tests for londonrent.text - description cleaning and the style features."""

import pandas as pd

from londonrent.text import STYLE_FEATURES, clean_description, style_features


def test_clean_description_strips_html_and_whitespace():
    s = pd.Series(["<b>Bright</b> flat.<br /><br />  Near   the   tube.", None])
    out = clean_description(s)
    assert out.iloc[0] == "Bright flat. Near the tube."
    assert out.iloc[1] == ""  # NaN -> empty string, never NaN


def test_style_features_counts():
    s = pd.Series(["STUNNING luxurious flat!! 2 min walk to the station."])
    f = style_features(s).iloc[0]
    assert f["desc_word_count"] == 9  # whitespace-split tokens
    assert f["desc_exclamations"] == 2
    assert f["desc_luxury_hits"] == 2  # "stunning", "luxurious"
    assert f["desc_practical_hits"] == 2  # "walk", "station"
    assert f["desc_tone"] == f["desc_luxury_hits"] - f["desc_practical_hits"]


def test_style_features_uppercase_ratio():
    shouty = style_features(pd.Series(["AMAZING PLACE IN LONDON"])).iloc[0]
    calm = style_features(pd.Series(["a quiet place in london"])).iloc[0]
    assert shouty["desc_uppercase_ratio"] > 0.9
    assert calm["desc_uppercase_ratio"] == 0.0


def test_style_features_shape_and_no_nan():
    f = style_features(pd.Series(["a", "", None, "words words words"]))
    assert list(f.columns) == STYLE_FEATURES
    assert f.notna().all().all()
