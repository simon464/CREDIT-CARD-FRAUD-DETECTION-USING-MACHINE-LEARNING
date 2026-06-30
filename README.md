# SafiCard Streamlit Fraud Detection App

This Streamlit app hosts your already-trained fraud detection models from the notebook.

## Folder Structure

Place your saved model files in a folder named `models` next to `app.py`.

```text
safecard_streamlit_app/
  app.py
  requirements.txt
  models/
    consensus_package.pkl        # optional

    lr_model.pkl
    rf_model.pkl
    xgb_model.pkl
    lgbm_model.pkl

    lr_threshold.pkl
    rf_threshold.pkl
    xgb_threshold.pkl
    lgbm_threshold.pkl
    thresholds.json              # optional
```

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Required CSV Columns for Batch Prediction

Your CSV must contain:

```text
trans_date_trans_time, category, amt, gender, city_pop, dob, lat, long, merch_lat, merch_long
```

## Important Note

The app does not retrain models. It only loads the trained `.pkl` files from the `models/` folder.
