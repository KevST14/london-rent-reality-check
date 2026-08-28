"""Rent Reality Check — the interactive demo.

Run with:  uv run streamlit run app/streamlit_app.py

It loads the cold-start model saved by notebook 03
(`models/price_model.joblib`), takes a listing's details, and shows:

* a predicted fair nightly price,
* how a price you enter compares to that, and
* a SHAP breakdown of what drove the number.

Everything the model needs that the user can't reasonably type (distance to the
nearest Tube, how many restaurants are within 500 m, …) is computed from the
chosen location via `londonrent.geo`.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

from londonrent import model as M
from londonrent.config import PROJECT_ROOT
from londonrent.features import _PROPERTY_MAP, AMENITY_FEATURES, PREMIUM_AMENITIES
from londonrent.geo import add_location_features, fetch_pois

MODEL_PATH = PROJECT_ROOT / "models" / "price_model.joblib"

st.set_page_config(page_title="Rent Reality Check — London", page_icon="🏠", layout="wide")


@st.cache_resource
def load_everything():
    bundle = M.load_bundle(MODEL_PATH)
    pois = fetch_pois()
    # borough centroids, so picking a borough gives a sensible default lat/long
    hoods = gpd.read_file(PROJECT_ROOT / "data" / "raw" / "neighbourhoods.geojson").to_crs(4326)
    pt = hoods.geometry.representative_point()
    hoods = hoods.assign(lat=pt.y.to_numpy(), lon=pt.x.to_numpy())
    centroids = hoods.set_index("neighbourhood")[["lat", "lon"]].to_dict("index")
    return bundle, pois, centroids


bundle, POIS, CENTROIDS = load_everything()
GROUPS = bundle["groups"]
ENCODER = bundle["encoder"]
MODEL = bundle["model"]
BOROUGHS = sorted(ENCODER.categories_[GROUPS["categorical"].index("neighbourhood_cleansed")])
PROPERTY_CLASSES = sorted(set(_PROPERTY_MAP.values()) | {"other"})

st.title("🏠 Rent Reality Check — London")
st.caption(
    "A student project. Predicts a fair nightly price for a London short-let from its "
    "attributes and location, and explains the number. Trained on the Inside Airbnb "
    "snapshot of 19 June 2026; tested on parts of London it never saw "
    f"(median error ≈ {bundle['metrics']['MdAPE']:.0%})."
)

# ------------------------------------------------------------------ inputs
left, right = st.columns([1, 1.3])

with left:
    st.subheader("Your listing")
    borough = st.selectbox("Borough", BOROUGHS, index=BOROUGHS.index("Hackney") if "Hackney" in BOROUGHS else 0)
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
                              help="The full Airbnb amenity list is usually 20–45 items.")

    st.markdown("**Notable amenities**")
    acols = st.columns(4)
    labels = {k: k.replace("amen_", "").replace("_", " ") for k in PREMIUM_AMENITIES}
    checked = {
        k: acols[i % 4].checkbox(labels[k], key=k)
        for i, k in enumerate(PREMIUM_AMENITIES)
    }

    st.markdown("**Exact spot** (defaults to the middle of the borough)")
    d = CENTROIDS.get(borough, {"lat": 51.5074, "lon": -0.1278})
    lc1, lc2 = st.columns(2)
    lat = lc1.number_input("Latitude", value=float(d["lat"]), format="%.5f")
    lon = lc2.number_input("Longitude", value=float(d["lon"]), format="%.5f")

    your_price = st.number_input("A price you're considering (£/night, optional)", 0, 2000, 0)

# ------------------------------------------------------------------ assemble one row
row = {
    "neighbourhood_cleansed": borough,
    "room_type": room_type,
    "property_class": property_class,
    "accommodates": float(accommodates),
    "bedrooms": float(bedrooms),
    "beds": float(beds),
    "bathrooms": float(bathrooms),
    "bath_is_shared": float(bath_is_shared),
    "amenity_count": float(amenity_count),
    "latitude": lat,
    "longitude": lon,
}
for k in AMENITY_FEATURES:
    row[k] = int(checked.get(k, False))

one = pd.DataFrame([row])
one = add_location_features(one, POIS)             # distance-to-Tube etc. from lat/long
X_one = M.build_design_matrix(one, GROUPS, ENCODER)
pred_log = float(MODEL.predict(X_one)[0])
pred_gbp = float(np.exp(pred_log))

# ------------------------------------------------------------------ output
with right:
    st.subheader("The model's take")
    st.metric("Predicted fair price", f"£{pred_gbp:,.0f} / night")

    if your_price > 0:
        diff = (your_price - pred_gbp) / pred_gbp
        verdict = "about right" if abs(diff) <= 0.10 else ("above the model" if diff > 0 else "below the model")
        st.metric("Your price vs model", f"{diff:+.0%}", help=verdict)
        st.progress(min(max(0.5 + diff, 0.0), 1.0))

    st.markdown("##### Why this price?")
    import matplotlib.pyplot as plt
    import shap

    explainer = shap.TreeExplainer(MODEL)
    sv = explainer(X_one)
    shap.plots.waterfall(sv[0], max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    st.pyplot(fig, clear_figure=True)
    st.caption(
        "Each bar is one feature's push, in log-price, away from the average listing. "
        "Bars add up to the prediction. Distances are in metres; `=1` on a borough/"
        "amenity row means that box is ticked."
    )

    st.markdown(
        f"**Computed for this spot:** "
        f"{one['dist_station_m'].iat[0]:.0f} m to a station · "
        f"{one['food_within_500m'].iat[0]:.0f} places to eat within 500 m · "
        f"{one['dist_center_m'].iat[0]/1000:.1f} km to the centre"
    )

with st.expander("How good is this model? (honest test-set numbers)"):
    m = bundle["metrics"]
    st.write(pd.Series({
        "Median % error": f"{m['MdAPE']:.1%}",
        "Within ±15%": f"{m['within_15pct']:.0%}",
        "Mean £ error": f"£{m['MAE_gbp']:.0f}",
        "R² (log price)": f"{m['R2_log']:.2f}",
    }))
    st.caption(
        "Measured on whole areas of London held out of training (spatial cross-validation). "
        "The model uses no review scores, so it works for brand-new listings; a listing with "
        "a long track record could be priced a little more tightly."
    )
