# Sydney House Price Prediction – SIT307 8.1D

Ayaan Ali · Deakin University

A machine learning project that predicts house sale prices in three different Sydney markets, using a dataset I collected by hand from realestate.com.au.

| Suburb | Area type | Approx. distance from CBD |
|---|---|---|
| Marrickville | Inner-west, near industrial area | ~7 km |
| Cronulla | Beach | ~26 km |
| Springwood | Lower Blue Mountains | ~70 km |

## Links

- **Live app (Streamlit):** https://sydney-house-price-ayaan.streamlit.app
- **Video walkthrough:** https://youtu.be/FmggkhNbgVg
- **Dataset:** [sydney_sold_properties.csv](sydney_sold_properties.csv)

## Dataset

- 117 sold properties (116 after removing one whole block of units), sold May 2024 – Sep 2026
- Columns: suburb, address, sale price, sale date, property type, bedrooms, bathrooms, car spaces, land size, building size, minutes to nearest station, agent description, listing URL
- Every row links to its original listing so it can be checked

## Method

1. **Cleaning:** removed duplicates and the block of units; grouped property types into House, Apartment/Unit and Townhouse/Villa
2. **Exploration:** price is right-skewed (skew 2.07), so models train on log(price)
3. **Feature engineering:** total rooms, land per bedroom, strata flag, days since first sale, and keyword flags from the agent descriptions (pool, renovated, needs work, views, granny flat)
4. **Models:** Linear Regression, KNN and Random Forest, compared with 5-fold cross-validation
5. **Error analysis:** out-of-fold predictions, largest errors, permutation feature importance, and an experiment comparing extra features against removing outliers

## Results

| Model | Validation MAE | Validation R² |
|---|---|---|
| Linear Regression | ~$374K | 0.71 |
| KNN | ~$527K | 0.56 |
| **Random Forest** | **~$367K** | **0.72** |

- Median error of the best model: about $177K (13.8%)
- Suburb is by far the most important feature
- The largest errors are unique, high-value Cronulla homes (waterfront and beachfront)

## Repository structure

| File | Purpose |
|---|---|
| `sydney_sold_properties.csv` | Hand-collected dataset (117 sold properties, with listing links) |
| `SIT307_8_1D_Answers.ipynb` | Full analysis: cleaning, EDA, feature engineering, 3 models, error analysis |
| `model.joblib` | Final trained Random Forest pipeline, saved from the notebook |
| `streamlit_app.py` | Deployed web app (single prediction + CSV upload) |
| `requirements.txt` | Packages for the Streamlit app (Python 3.12) |
| `app.py` | Earlier Gradio version of the app, tested in Colab |

## How to run

**Notebook:** open it in Google Colab, upload the CSV when asked, then click Runtime → Run all.

**App locally (Python 3.12):**

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Deploy:** Streamlit Community Cloud → Create app → this repo, branch `main`, file `streamlit_app.py`, Python 3.12 (Advanced settings).

## Limitations

Small dataset (116 homes), uneven sale dates across suburbs, missing land sizes for units, and no features for water frontage, condition or school zones. Predictions are estimates only, not formal valuations.
