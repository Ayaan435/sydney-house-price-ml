"""Sydney House Price Predictor – SIT307 8.1D (Ayaan Ali)"""
import gradio as gr
import joblib
import numpy as np
import pandas as pd

bundle = joblib.load("model.joblib")
model = bundle["model"]
start_date = bundle["start_date"]
keywords = bundle["keywords"]

INPUT_COLS = ["suburb", "property_type", "bedrooms", "bathrooms", "car_spaces",
              "land_size_sqm", "station_min"]


def build_features(d):
    """Create the same engineered features the notebook used."""
    d = d.copy()
    for col in ["bedrooms", "bathrooms", "car_spaces", "land_size_sqm", "station_min"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["land_size_sqm"] = d["land_size_sqm"].replace(0, np.nan)
    d["car_spaces"] = d["car_spaces"].replace(0, np.nan)
    if "sale_date" in d:
        dates = pd.to_datetime(d["sale_date"], dayfirst=True, errors="coerce").fillna(pd.Timestamp.today())
    else:
        dates = pd.Series(pd.Timestamp.today(), index=d.index)
    d["days_since_start"] = (dates - start_date).dt.days
    d["total_rooms"] = d["bedrooms"] + d["bathrooms"]
    d["land_per_bed"] = d["land_size_sqm"] / d["bedrooms"]
    d["is_strata"] = (d["property_type"] != "House").astype(int)
    desc = d["description"].fillna("").str.lower() if "description" in d else pd.Series("", index=d.index)
    for col, pattern in keywords.items():
        if col not in d:
            d[col] = desc.str.contains(pattern, regex=True).astype(int)
    return d


def predict_one(suburb, ptype, beds, baths, cars, land, station, pool, reno, needs_work, view, granny):
    row = pd.DataFrame([{
        "suburb": suburb, "property_type": ptype, "bedrooms": beds, "bathrooms": baths,
        "car_spaces": cars, "land_size_sqm": land, "station_min": station,
        "has_pool": int(pool), "is_renovated": int(reno), "needs_work": int(needs_work),
        "has_view": int(view), "has_granny_flat": int(granny),
    }])
    price = model.predict(build_features(row))[0]
    low, high = price * 0.86, price * 1.14   # ± median % error from the notebook
    return (f"## Estimated sale price: ${price:,.0f}\n"
            f"Typical range: ${low:,.0f} – ${high:,.0f}\n\n"
            f"*Model: {bundle['model_name']}. An estimate only – not a formal valuation.*")


def predict_csv(file):
    d = pd.read_csv(file)
    missing = [c for c in INPUT_COLS if c not in d.columns]
    if missing:
        raise gr.Error(f"CSV is missing columns: {missing}")
    d["property_type"] = d["property_type"].replace({
        "Unit": "Apartment/Unit", "Apartment": "Apartment/Unit", "Studio": "Apartment/Unit",
        "Villa": "Townhouse/Villa", "Townhouse": "Townhouse/Villa",
        "Duplex/semi-detached": "House",
    })
    d["predicted_price"] = model.predict(build_features(d)).round(0)
    out = d[INPUT_COLS + ["predicted_price"]].copy()
    if "sale_price" in d:
        out["sale_price"] = d["sale_price"]
        out["pct_error"] = (100 * (out["predicted_price"] - d["sale_price"]).abs() / d["sale_price"]).round(1)
    out.to_csv("predictions.csv", index=False)
    return out, "predictions.csv"


with gr.Blocks(title="Sydney House Price Predictor") as demo:
    gr.Markdown("# Sydney House Price Predictor\n"
                "Trained on 116 sold properties in Marrickville, Cronulla and Springwood (2024–2026).")
    with gr.Tab("Single property"):
        with gr.Row():
            with gr.Column():
                suburb = gr.Dropdown(bundle["suburbs"], value=bundle["suburbs"][0], label="Suburb")
                ptype = gr.Dropdown(bundle["types"], value="House", label="Property type")
                beds = gr.Slider(1, 6, value=3, step=1, label="Bedrooms")
                baths = gr.Slider(1, 4, value=2, step=1, label="Bathrooms")
                cars = gr.Slider(0, 5, value=1, step=1, label="Car spaces")
                land = gr.Number(value=0, label="Land size m² (0 if unit / unknown)")
                station = gr.Number(value=3, label="Minutes to nearest station")
            with gr.Column():
                pool = gr.Checkbox(label="Pool")
                reno = gr.Checkbox(label="Renovated / brand new")
                needs_work = gr.Checkbox(label="Needs work / renovator's opportunity")
                view = gr.Checkbox(label="Views (water, city, ocean)")
                granny = gr.Checkbox(label="Granny flat")
                btn = gr.Button("Predict price", variant="primary")
                out = gr.Markdown()
        btn.click(predict_one, [suburb, ptype, beds, baths, cars, land, station,
                                pool, reno, needs_work, view, granny], out)
    with gr.Tab("Upload CSV"):
        gr.Markdown("Upload a CSV with columns: " + ", ".join(INPUT_COLS) +
                    " (optional: description, sale_date, sale_price).")
        f = gr.File(label="CSV file", file_types=[".csv"])
        run = gr.Button("Predict all", variant="primary")
        table = gr.Dataframe()
        dl = gr.File(label="Download predictions")
        run.click(predict_csv, f, [table, dl])

if __name__ == "__main__":
    demo.launch()
