"""Sydney House Price Predictor – SIT307 8.1D (Ayaan Ali) – Streamlit app"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sydney House Price Predictor", page_icon="🏠", layout="wide")

TYPE_MAP = {"Unit": "Apartment/Unit", "Apartment": "Apartment/Unit", "Studio": "Apartment/Unit",
            "Villa": "Townhouse/Villa", "Townhouse": "Townhouse/Villa", "Duplex/semi-detached": "House"}
INPUT_COLS = ["suburb", "property_type", "bedrooms", "bathrooms", "car_spaces", "land_size_sqm", "station_min"]


@st.cache_resource
def load_bundle():
    return joblib.load("model.joblib")


bundle = load_bundle()
model = bundle["model"]


def build_features(d):
    """Create the same engineered features the notebook used."""
    d = d.copy()
    for col in ["bedrooms", "bathrooms", "car_spaces", "land_size_sqm", "station_min"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["land_size_sqm"] = d["land_size_sqm"].replace(0, np.nan)
    d["days_since_start"] = (pd.Timestamp.today() - bundle["start_date"]).days
    d["total_rooms"] = d["bedrooms"] + d["bathrooms"]
    d["land_per_bed"] = d["land_size_sqm"] / d["bedrooms"]
    d["is_strata"] = (d["property_type"] != "House").astype(int)
    desc = d["description"].fillna("").str.lower() if "description" in d else pd.Series("", index=d.index)
    for col, pattern in bundle["keywords"].items():
        if col not in d:
            d[col] = desc.str.contains(pattern, regex=True).astype(int)
    return d


st.title("🏠 Sydney House Price Predictor")
st.caption(f"{bundle['model_name']} trained on 116 sold properties in Marrickville, Cronulla and "
           "Springwood (2024–2026). Estimates only – not a formal valuation.")

tab1, tab2 = st.tabs(["Single property", "Upload CSV"])

with tab1:
    left, right = st.columns(2)
    with left:
        suburb = st.selectbox("Suburb", bundle["suburbs"])
        ptype = st.selectbox("Property type", bundle["types"], index=bundle["types"].index("House"))
        beds = st.slider("Bedrooms", 1, 6, 3)
        baths = st.slider("Bathrooms", 1, 4, 2)
        cars = st.slider("Car spaces", 0, 5, 1)
        land = st.number_input("Land size m² (0 if unit / unknown)", min_value=0.0, value=0.0, step=10.0)
        station = st.number_input("Minutes to nearest station", min_value=0, value=3)
    with right:
        pool = st.checkbox("Pool")
        reno = st.checkbox("Renovated / brand new")
        needs_work = st.checkbox("Needs work / renovator's opportunity")
        view = st.checkbox("Views (water, city, ocean)")
        granny = st.checkbox("Granny flat")
        if st.button("Predict price", type="primary", use_container_width=True):
            row = pd.DataFrame([{
                "suburb": suburb, "property_type": ptype, "bedrooms": beds, "bathrooms": baths,
                "car_spaces": cars, "land_size_sqm": land, "station_min": station,
                "has_pool": int(pool), "is_renovated": int(reno), "needs_work": int(needs_work),
                "has_view": int(view), "has_granny_flat": int(granny),
            }])
            price = model.predict(build_features(row))[0]
            st.metric("Estimated sale price", f"${price:,.0f}")
            st.write(f"Typical range: **\\${price * 0.86:,.0f} – \\${price * 1.14:,.0f}** (±14%, the model's median error)")

with tab2:
    st.write("Upload a CSV with columns: " + ", ".join(INPUT_COLS) +
             " (optional: description, sale_price).")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        d = pd.read_csv(file)
        missing = [c for c in INPUT_COLS if c not in d.columns]
        if missing:
            st.error(f"CSV is missing columns: {missing}")
        else:
            d["property_type"] = d["property_type"].replace(TYPE_MAP)
            d = d[d["property_type"].isin(bundle["types"])].copy()
            d["predicted_price"] = model.predict(build_features(d)).round(0)
            out = d[INPUT_COLS + ["predicted_price"]].copy()
            if "sale_price" in d:
                out["sale_price"] = d["sale_price"]
                out["pct_error"] = (100 * (out["predicted_price"] - d["sale_price"]).abs() / d["sale_price"]).round(1)
            st.dataframe(out, use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")
