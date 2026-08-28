"""Rent Reality Check — London · interactive demo.

Created by Kevin Steepan as a student portfolio project.

Run:  uv run streamlit run app/streamlit_app.py

Loads the cold-start model saved by notebook 03 (`models/price_model.joblib`),
takes a listing's details, and returns a predicted fair nightly price, a
plain-language verdict against a price you enter, and a SHAP breakdown of what
drove the number. The location features (distance to the nearest Tube, food
density, distance to the centre) are computed on the fly from the chosen point.
"""

from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from londonrent import model as M
from londonrent.config import PROJECT_ROOT
from londonrent.features import _PROPERTY_MAP, AMENITY_FEATURES, PREMIUM_AMENITIES
from londonrent.geo import add_location_features, fetch_pois

MODEL_PATH = PROJECT_ROOT / "models" / "price_model.joblib"

st.set_page_config(page_title="Rent Reality Check — London", page_icon="🏠", layout="wide")


# --------------------------------------------------------------------------- load
@st.cache_resource
def load_everything():
    bundle = M.load_bundle(MODEL_PATH)
    pois = fetch_pois()
    hoods = gpd.read_file(PROJECT_ROOT / "data" / "raw" / "neighbourhoods.geojson").to_crs(4326)
    pt = hoods.geometry.representative_point()
    hoods = hoods.assign(lat=pt.y.to_numpy(), lon=pt.x.to_numpy())
    centroids = hoods.set_index("neighbourhood")[["lat", "lon"]].to_dict("index")
    return bundle, pois, centroids


bundle, POIS, CENTROIDS = load_everything()
GROUPS, ENCODER, MODEL, METRICS = (
    bundle["groups"], bundle["encoder"], bundle["model"], bundle["metrics"],
)
BOROUGHS = sorted(ENCODER.categories_[GROUPS["categorical"].index("neighbourhood_cleansed")])
PROPERTY_CLASSES = sorted(set(_PROPERTY_MAP.values()) | {"other"})


def pretty(col: str) -> str:
    """Turn a design-matrix column name into something a person can read."""
    if col.startswith("neighbourhood_cleansed_"):
        return "Borough: " + col.split("_", 2)[2]
    if col.startswith("room_type_"):
        return col.split("_", 2)[2]
    if col.startswith("property_class_"):
        return "Type: " + col.split("_", 2)[2].replace("_", " ")
    if col.startswith("amen_"):
        return "Has " + col[5:].replace("_", " ")
    return {
        "accommodates": "Sleeps", "amenity_count": "Total amenities",
        "bath_is_shared": "Shared bathroom", "bathrooms": "Bathrooms",
        "bedrooms": "Bedrooms", "beds": "Beds",
        "dist_station_m": "Metres to nearest station",
        "stations_within_1km": "Stations within 1 km",
        "dist_restaurant_m": "Metres to nearest café/restaurant",
        "food_within_500m": "Food places within 500 m",
        "dist_park_m": "Metres to nearest park",
        "dist_center_m": "Metres to central London",
    }.get(col, col)


# --------------------------------------------------------------------------- header
st.title("🏠 Rent Reality Check — London")
st.markdown(
    "**What's a fair nightly price for a London short-let — and *why*?** "
    "Enter a listing below. The model was trained on ~61,000 real London Airbnb "
    "listings (Inside Airbnb, June 2026) and tested on parts of the city it never "
    f"saw: it gets a typical listing to within **±{METRICS['MdAPE']:.0%}**."
)
st.caption(
    "Created by **Kevin Steepan** — a student project to show applied ML on real-world "
    "data: framing a real problem, engineering features, validating honestly, and making "
    "the model explain itself. The point is the reasoning, not just the number."
)

tab_price, tab_how, tab_acc = st.tabs(
    ["🔮  Price a listing", "🛠️  How it works", "🎯  How accurate is it"]
)

# ========================================================================= PRICE
with tab_price:
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.subheader("Your listing")
        borough = st.selectbox("Borough", BOROUGHS,
                               index=BOROUGHS.index("Hackney") if "Hackney" in BOROUGHS else 0)
        c1, c2 = st.columns(2)
        room_type = c1.selectbox("Room type", ["Entire home/apt", "Private room"])
        property_class = c2.selectbox("Property type", PROPERTY_CLASSES,
                                      index=PROPERTY_CLASSES.index("rental_unit"))
        c3, c4, c5 = st.columns(3)
        accommodates = c3.number_input("Sleeps", 1, 16, 3)
        bedrooms = c4.number_input("Bedrooms", 0, 10, 1)
        beds = c5.number_input("Beds", 1, 16, 2)
        c6, c7 = st.columns(2)
        bathrooms = c6.number_input("Bathrooms", 0.0, 8.0, 1.0, step=0.5)
        bath_is_shared = c7.checkbox("Bathroom is shared")
        amenity_count = st.slider("Roughly how many amenities are listed?", 5, 90, 30,
                                  help="A full Airbnb amenity list is usually 20–45 items.")

        st.markdown("**Notable amenities**")
        acols = st.columns(4)
        labels = {k: k.replace("amen_", "").replace("_", " ") for k in PREMIUM_AMENITIES}
        checked = {k: acols[i % 4].checkbox(labels[k], key=k)
                   for i, k in enumerate(PREMIUM_AMENITIES)}

        st.markdown("**Exact spot** — drag off the borough centre if you know it")
        d = CENTROIDS.get(borough, {"lat": 51.5074, "lon": -0.1278})
        lc1, lc2 = st.columns(2)
        lat = lc1.number_input("Latitude", value=float(d["lat"]), format="%.5f")
        lon = lc2.number_input("Longitude", value=float(d["lon"]), format="%.5f")

        your_price = st.number_input("A price you're considering (£/night) — optional",
                                     0, 2000, 0, step=5)

    # ---- assemble one row and predict
    row = {
        "neighbourhood_cleansed": borough, "room_type": room_type,
        "property_class": property_class, "accommodates": float(accommodates),
        "bedrooms": float(bedrooms), "beds": float(beds), "bathrooms": float(bathrooms),
        "bath_is_shared": float(bath_is_shared), "amenity_count": float(amenity_count),
        "latitude": lat, "longitude": lon,
    }
    for k in AMENITY_FEATURES:
        row[k] = int(checked.get(k, False))

    one = add_location_features(pd.DataFrame([row]), POIS)
    X_one = M.build_design_matrix(one, GROUPS, ENCODER)
    pred_gbp = float(np.exp(MODEL.predict(X_one)[0]))

    with right:
        st.subheader("The model's take")
        mc1, mc2 = st.columns(2)
        mc1.metric("Predicted fair price", f"£{pred_gbp:,.0f}", help="per night")
        if your_price > 0:
            diff = (your_price - pred_gbp) / pred_gbp
            mc2.metric("Your price vs model", f"{diff:+.0%}",
                       delta=f"£{your_price - pred_gbp:+,.0f}", delta_color="off")
            if abs(diff) <= 0.10:
                st.success(f"**£{your_price:,.0f} looks about right** — within 10% of the model.")
            elif diff > 0:
                st.warning(f"**£{your_price:,.0f} is {diff:.0%} above the model.** "
                           "Might be justified by something it can't see (a view, a recent refurb) "
                           "— or it's a stretch.")
            else:
                st.info(f"**£{your_price:,.0f} is {abs(diff):.0%} below the model.** "
                        "You may be leaving money on the table.")

            # simple visual: where your price sits around the prediction
            fig_b, axb = plt.subplots(figsize=(7, 0.9))
            axb.axvspan(pred_gbp * 0.9, pred_gbp * 1.1, color="#22c55e", alpha=0.25)
            axb.axvline(pred_gbp, color="#22c55e", lw=2, label="model")
            axb.axvline(your_price, color="#111827", lw=2, ls="--", label="your price")
            axb.set_xlim(0, max(pred_gbp, your_price) * 1.6)
            axb.set_yticks([]); axb.legend(loc="upper right", fontsize=8, frameon=False)
            axb.set_xlabel("£ / night")
            st.pyplot(fig_b, clear_figure=True)

        st.markdown("##### Why this price?")
        st.caption(
            "Each bar is one feature nudging the price away from the **average London "
            "listing**. Red pushes the price up, blue pushes it down; the bars add up "
            "exactly to the prediction. (This is a *SHAP* plot — see the **How it works** tab.)"
        )
        X_pretty = X_one.rename(columns={c: pretty(c) for c in X_one.columns})
        sv = shap.TreeExplainer(MODEL)(X_pretty)
        shap.plots.waterfall(sv[0], max_display=11, show=False)
        fig = plt.gcf(); fig.set_size_inches(8, 5)
        st.pyplot(fig, clear_figure=True)

        st.markdown(
            f"**Worked out for this exact spot:** "
            f"{one['dist_station_m'].iat[0]:.0f} m to a station · "
            f"{one['food_within_500m'].iat[0]:.0f} places to eat within 500 m · "
            f"{one['dist_center_m'].iat[0] / 1000:.1f} km to central London."
        )

# ====================================================================== HOW
with tab_how:
    st.subheader("From a spreadsheet to a price, in five steps")
    steps = [
        ("1 · Real data", (
            "~93,000 London Airbnb listings from Inside Airbnb (June 2026). A third have "
            "no price, and those aren't random, so the model is honestly about listings "
            "that *publish* a price — whole homes and private rooms, £10–£1000/night."
        )),
        ("2 · Predict the *log* of price", (
            "Nightly prices are lopsided — most are £80–£250, a few run past £1000. "
            "Predicting `log(price)` stops those few dominating and turns the model's "
            "errors into percentages, which is how anyone actually thinks about price."
        )),
        ("3 · Turn location into numbers", (
            "A borough is huge. From each listing's coordinates the app measures distance "
            "to the nearest Tube/rail station, how many stations and food venues are "
            "nearby, and distance to central London — all computed in real metres, not "
            "raw latitude/longitude."
        )),
        ("4 · A gradient-boosted tree model", (
            "Hundreds of small decision trees, each correcting the last one's mistakes. "
            "It captures 'it depends' effects (an extra bedroom is worth more in "
            "Kensington than in Barnet) that a straight-line model can't, and it beats a "
            "tuned linear baseline by about £9/night. Checked with *spatial* "
            "cross-validation — whole 2 km areas of London held out — so the score "
            "reflects genuinely new listings."
        )),
        ("5 · Make it explain itself (SHAP)", (
            "SHAP splits every prediction into per-feature contributions that add up to "
            "the number: '+£30 for the borough, −£15 for no dishwasher, +£8 for being "
            "close to a station…'. It comes from cooperative game theory (Shapley, 1953). "
            "That's the waterfall chart on the first tab."
        )),
    ]
    for title, body in steps:
        st.markdown(f"**{title}**")
        st.write(body)

    st.divider()
    st.markdown(
        "**What got left out, and why.** The project also tested whether the *wording* "
        "of a listing's description helps — it barely does (about £1–2/night on top of "
        "the facts), so the model stays text-free and simpler to run. Review scores are "
        "left out on purpose so it works for brand-new listings (costs ~£6.50/night of "
        "accuracy, measured not hidden)."
    )
    st.markdown(
        "**Honest limitations.** The model under-prices the luxury end of the market, "
        "is weakest in outer boroughs with few listings, and its worst individual misses "
        "are quirks in the source data (hotel rooms priced at the £1000 cap)."
    )
    st.caption(
        "Full write-up, code, and the three notebooks (data exploration → location "
        "features → the model + explanations) are in the project repository. "
        "Techniques are attributed to their common public sources (ISLR, Géron's "
        "*Hands-On ML*, StatQuest, Molnar's *Interpretable ML*, the scikit-learn guide)."
    )

# ====================================================================== ACCURACY
with tab_acc:
    st.subheader("Measured on London it never trained on")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Median error", f"{METRICS['MdAPE']:.0%}",
              help="Half of listings are predicted closer than this, half further.")
    a2.metric("Within ±15%", f"{METRICS['within_15pct']:.0%}")
    a3.metric("Mean £ error", f"£{METRICS['MAE_gbp']:.0f}")
    a4.metric("R² (log price)", f"{METRICS['R2_log']:.2f}",
              help="0 = no better than guessing the average; 1 = perfect.")

    st.markdown(
        "These come from a **held-out test**: whole grid-cell areas of London were "
        "kept out of training entirely, and the model was scored on them exactly once. "
        "For comparison, a no-model baseline (guess the borough × room-type median) "
        "manages about **£90 mean error / R² 0.49** — so the model roughly halves the "
        "error a sensible lookup table would make."
    )
    st.markdown(
        "**Where it's weakest:** the top of the price range (logging the target biases "
        "the luxury tail low) and thinly-sampled outer boroughs (Havering, Sutton, "
        "Richmond) where there simply isn't much to learn from."
    )

st.divider()
st.caption("Rent Reality Check — London · built by Kevin Steepan · data: Inside Airbnb & OpenStreetMap")
