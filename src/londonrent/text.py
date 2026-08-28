"""Turn a listing's free-text ``description`` into things a model can use.

Two very different approaches live here, and notebook 04 compares them:

1. **Hand-crafted "style" features** (:func:`style_features`) - a dozen numbers
   that describe *how* the description is written, not *what* it says: how long it
   is, how SHOUTY, how many exclamation marks, how many marketing adjectives
   ("stunning", "luxurious") vs practical words ("wifi", "station"). Cheap, and
   every feature is self-explaining.

2. **Bag-of-words** (done inline in the notebook with scikit-learn's
   ``TfidfVectorizer``) - treat the description as an unordered pile of words and
   let the data say which words track higher or lower prices.

Neither "understands" English the way a large language model would; both are the
sensible first things to try, and both are fully transparent, which matters for a
project whose point is explaining itself.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Marketing / aspiration words vs practical / logistics words. Hand-picked from
# skimming a few hundred descriptions - the hypothesis is that hosts leaning on
# the first list are signalling "premium" and charging for it, while the second
# list is just describing the place. Notebook 04 tests whether that holds up.
LUXURY_WORDS = {
    "stunning",
    "luxury",
    "luxurious",
    "boutique",
    "elegant",
    "exclusive",
    "prestigious",
    "breathtaking",
    "spectacular",
    "immaculate",
    "stylish",
    "designer",
    "bespoke",
    "sumptuous",
    "opulent",
    "chic",
    "beautiful",
    "gorgeous",
    "premium",
    "upscale",
}

PRACTICAL_WORDS = {
    "wifi",
    "transport",
    "tube",
    "station",
    "minutes",
    "walk",
    "bus",
    "supermarket",
    "parking",
    "quiet",
    "clean",
    "comfortable",
    "ideal",
    "close",
    "near",
    "easy",
    "spacious",
    "bright",
    "cosy",
    "modern",
}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_EMOJI_RE = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]")


def clean_description(raw: pd.Series) -> pd.Series:
    """Strip HTML tags (Inside Airbnb leaves ``<br />`` etc. in), collapse
    whitespace, and fill missing with an empty string so downstream code never
    has to special-case NaN."""
    s = raw.fillna("").astype(str)
    s = s.str.replace(_TAG_RE, " ", regex=True)
    s = s.str.replace(_WS_RE, " ", regex=True).str.strip()
    return s


def _count_hits(text_lower: str, vocab: set[str]) -> int:
    # split on non-letters so "station." and "station," both count, but
    # "stationery" (a different token) does not
    return sum(1 for w in re.findall(r"[a-z']+", text_lower) if w in vocab)


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """numer / denom, but 0 (not NaN or error) wherever denom is 0."""
    d = denom.to_numpy(dtype="float64")
    n = numer.to_numpy(dtype="float64")
    out = np.divide(n, d, out=np.zeros_like(n), where=d > 0)
    return pd.Series(out, index=numer.index)


def style_features(raw: pd.Series) -> pd.DataFrame:
    """A DataFrame of 'how is this written' features, one row per description.

    All are counts or ratios, so they drop straight into the numeric branch of
    the model alongside the structured features.
    """
    clean = clean_description(raw)

    word_count = clean.str.count(r"\S+")
    char_count = clean.str.len()
    letters = clean.str.count(r"[A-Za-z]")
    uppercase = clean.str.count(r"[A-Z]")

    out = pd.DataFrame(index=raw.index)
    out["desc_word_count"] = word_count
    out["desc_avg_word_len"] = _safe_ratio(char_count, word_count)
    out["desc_exclamations"] = clean.str.count("!")
    out["desc_question_marks"] = clean.str.count(r"\?")
    # SHOUTY-ness: share of letters that are capitals. Ordinary prose (names,
    # sentence starts) sits around 3-6%; marketing-heavy text runs much higher.
    out["desc_uppercase_ratio"] = _safe_ratio(uppercase, letters)
    out["desc_digit_ratio"] = _safe_ratio(clean.str.count(r"\d"), char_count)
    out["desc_has_emoji"] = clean.str.contains(_EMOJI_RE, regex=True).astype("int8")

    lower = clean.str.lower()
    out["desc_luxury_hits"] = lower.map(lambda t: _count_hits(t, LUXURY_WORDS)).astype(int)
    out["desc_practical_hits"] = lower.map(lambda t: _count_hits(t, PRACTICAL_WORDS)).astype(int)
    # net "tone": positive = leans marketing, negative = leans practical
    out["desc_tone"] = out["desc_luxury_hits"] - out["desc_practical_hits"]
    return out


STYLE_FEATURES = [
    "desc_word_count",
    "desc_avg_word_len",
    "desc_exclamations",
    "desc_question_marks",
    "desc_uppercase_ratio",
    "desc_digit_ratio",
    "desc_has_emoji",
    "desc_luxury_hits",
    "desc_practical_hits",
    "desc_tone",
]


if __name__ == "__main__":
    from .features import build_model_frame

    frame, _ = build_model_frame(keep_text=True)
    feats = style_features(frame["description"])
    print(feats.describe().T.round(2).to_string())
