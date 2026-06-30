import os
import sys
import json
import math
import joblib
import __main__
import numpy as np
import pandas as pd
import streamlit as st

try:
    import shap
    SHAP_AVAILABLE = True
    SHAP_IMPORT_ERROR = None
except Exception as shap_import_exc:
    shap = None
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = shap_import_exc

from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin


# =============================================================================
# CUSTOM TRANSFORMER REQUIRED BY THE SAVED .pkl MODELS
# =============================================================================
# Your saved pipelines were trained/saved while this custom class existed in the
# notebook as __main__.SafiCardFeatureEngineer. If this class is not available
# BEFORE joblib.load(...), model loading fails with:
# AttributeError: Can't get attribute 'SafiCardFeatureEngineer' on <module '__main__'>
# =============================================================================

FINAL_FEATURES = [
    "category",
    "amt",
    "gender",
    "city_pop",
    "hour",
    "day",
    "month",
    "dayofweek",
    "age",
    "distance_km",
]

NUMERIC_FEATURES = [
    "amt",
    "city_pop",
    "hour",
    "day",
    "month",
    "dayofweek",
    "age",
    "distance_km",
]

CATEGORICAL_FEATURES = [
    "category",
    "gender",
]


class SafiCardFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineer used by the trained SafiCard fraud detection pipelines.

    It accepts either:
    1. Raw transaction columns:
       trans_date_trans_time, category, amt, gender, city_pop, dob,
       lat, long, merch_lat, merch_long

    2. Already-engineered columns:
       category, amt, gender, city_pop, hour, day, month, dayofweek, age, distance_km

    It returns the exact final feature columns used during training.
    """

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _to_float_series(values):
        return pd.to_numeric(pd.Series(values), errors="coerce").astype(float)

    @staticmethod
    def haversine_distance_km(lat1, lon1, lat2, lon2):
        """Calculate Haversine distance in kilometers."""
        radius_km = 6371.0

        lat1 = np.radians(SafiCardFeatureEngineer._to_float_series(lat1))
        lon1 = np.radians(SafiCardFeatureEngineer._to_float_series(lon1))
        lat2 = np.radians(SafiCardFeatureEngineer._to_float_series(lat2))
        lon2 = np.radians(SafiCardFeatureEngineer._to_float_series(lon2))

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )

        c = 2 * np.arcsin(np.sqrt(a))
        return radius_km * c

    @staticmethod
    def _apply_aliases(X: pd.DataFrame) -> pd.DataFrame:
        """
        Supports common UI/CSV column names without breaking the trained pipeline.
        Existing correct columns are never overwritten.
        """
        alias_map = {
            "amount": "amt",
            "transaction_amount": "amt",
            "trans_hour": "hour",
            "transaction_hour": "hour",
            "trans_dom": "day",
            "day_of_month": "day",
            "transaction_day": "day",
            "trans_month": "month",
            "transaction_month": "month",
            "trans_dow": "dayofweek",
            "day_of_week": "dayofweek",
            "dayof_week": "dayofweek",
            "weekday": "dayofweek",
            "geo_distance": "distance_km",
            "distance": "distance_km",
            "merch_distance_km": "distance_km",
            "merchant_distance": "distance_km",
            "longitude": "long",
            "lon": "long",
            "merchant_latitude": "merch_lat",
            "merchant_longitude": "merch_long",
            "cardholder_latitude": "lat",
            "cardholder_longitude": "long",
        }

        X = X.copy()
        lower_to_original = {str(col).lower().strip(): col for col in X.columns}

        for alias, canonical in alias_map.items():
            if canonical not in X.columns and alias in lower_to_original:
                X[canonical] = X[lower_to_original[alias]]

        return X

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = self._apply_aliases(X)

        # Date/time features: create only when raw datetime exists.
        # Otherwise preserve already-engineered columns if the CSV/app supplied them.
        if "trans_date_trans_time" in X.columns:
            trans_dt = pd.to_datetime(X["trans_date_trans_time"], errors="coerce")
            X["hour"] = trans_dt.dt.hour
            X["day"] = trans_dt.dt.day
            X["month"] = trans_dt.dt.month
            X["dayofweek"] = trans_dt.dt.dayofweek
        else:
            for col in ["hour", "day", "month", "dayofweek"]:
                if col not in X.columns:
                    X[col] = np.nan

        # Age feature: create from dob and transaction date if possible.
        # Otherwise preserve existing age if it was supplied.
        if "dob" in X.columns and "trans_date_trans_time" in X.columns:
            dob = pd.to_datetime(X["dob"], errors="coerce")
            trans_dt = pd.to_datetime(X["trans_date_trans_time"], errors="coerce")

            X["age"] = trans_dt.dt.year - dob.dt.year
            birthday_not_reached = (
                (trans_dt.dt.month < dob.dt.month)
                | ((trans_dt.dt.month == dob.dt.month) & (trans_dt.dt.day < dob.dt.day))
            )
            X.loc[birthday_not_reached.fillna(False), "age"] -= 1
        elif "age" not in X.columns:
            X["age"] = np.nan

        # Distance feature: create from coordinates if possible.
        # Otherwise preserve existing distance_km if it was supplied.
        required_distance_columns = {"lat", "long", "merch_lat", "merch_long"}
        if required_distance_columns.issubset(set(X.columns)):
            X["distance_km"] = self.haversine_distance_km(
                X["lat"],
                X["long"],
                X["merch_lat"],
                X["merch_long"],
            )
        elif "distance_km" not in X.columns:
            X["distance_km"] = np.nan

        # Make sure final columns exist and are in the correct order.
        for col in FINAL_FEATURES:
            if col not in X.columns:
                X[col] = np.nan

        return X[FINAL_FEATURES]


# Make the class available exactly where pickle/joblib expects it.
# This is the key fix for the __main__.SafiCardFeatureEngineer loading error.
SafiCardFeatureEngineer.__module__ = "__main__"
setattr(__main__, "SafiCardFeatureEngineer", SafiCardFeatureEngineer)
sys.modules["__main__"].SafiCardFeatureEngineer = SafiCardFeatureEngineer


# =============================================================================
# APP CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="💳 Streamlit based application for Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_DISPLAY_NAMES = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}

MODEL_FILES = {
    "lr": "lr_model.pkl",
    "rf": "rf_model.pkl",
    "xgb": "xgb_model.pkl",
    "lgbm": "lgbm_model.pkl",
}

THRESHOLD_FILES = {
    "lr": "lr_threshold.pkl",
    "rf": "rf_threshold.pkl",
    "xgb": "xgb_threshold.pkl",
    "lgbm": "lgbm_threshold.pkl",
}

DEFAULT_THRESHOLDS = {
    "lr": 0.50,
    "rf": 0.50,
    "xgb": 0.50,
    "lgbm": 0.50,
}

RAW_REQUIRED_COLUMNS = [
    "trans_date_trans_time",
    "category",
    "amt",
    "gender",
    "city_pop",
    "dob",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
]

ENGINEERED_REQUIRED_COLUMNS = [
    "category",
    "amt",
    "gender",
    "city_pop",
    "hour",
    "day",
    "month",
    "dayofweek",
    "age",
    "distance_km",
]

CATEGORY_OPTIONS = [
    "gas_transport",
    "grocery_pos",
    "home",
    "shopping_pos",
    "kids_pets",
    "shopping_net",
    "entertainment",
    "food_dining",
    "personal_care",
    "health_fitness",
    "misc_pos",
    "misc_net",
    "grocery_net",
    "travel",
]

DECISION_DETAILS = {
    "APPROVE": "0 or 1 model flagged the transaction as fraud.",
    "MANUAL REVIEW": "Exactly 2 models flagged the transaction as fraud.",
    "BLOCK": "3 or 4 models flagged the transaction as fraud.",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_model_key(key) -> str:
    """Converts different saved dictionary keys into lr, rf, xgb, or lgbm."""
    text = str(key).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")

    aliases = {
        "lr": "lr",
        "logistic": "lr",
        "logistic_regression": "lr",
        "smote_logistic_regression": "lr",
        "random_forest": "rf",
        "rf": "rf",
        "smote_random_forest": "rf",
        "xgb": "xgb",
        "xgboost": "xgb",
        "smote_xgboost": "xgb",
        "lgbm": "lgbm",
        "lightgbm": "lgbm",
        "light_gbm": "lgbm",
        "tuned_smote_lightgbm": "lgbm",
        "smote_lightgbm": "lgbm",
    }

    return aliases.get(text, text)


def safe_float(value, default=0.50) -> float:
    """Converts saved threshold values to plain float safely."""
    try:
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            value = np.ravel(value)[0]
        value = float(value)
        if math.isnan(value):
            return float(default)
        return value
    except Exception:
        return float(default)


def available_model_files(models_dir: Path):
    if not models_dir.exists():
        return []
    return sorted([p.name for p in models_dir.iterdir() if p.is_file()])


def validate_input_columns(df: pd.DataFrame):
    """
    CSV is valid if it has raw columns or already-engineered columns.
    Aliases are handled inside SafiCardFeatureEngineer, but this validation
    checks the common canonical forms.
    """
    cols = set(df.columns)

    has_raw = all(col in cols for col in RAW_REQUIRED_COLUMNS)
    has_engineered = all(col in cols for col in ENGINEERED_REQUIRED_COLUMNS)

    if has_raw or has_engineered:
        return []

    missing_raw = [col for col in RAW_REQUIRED_COLUMNS if col not in cols]
    missing_engineered = [col for col in ENGINEERED_REQUIRED_COLUMNS if col not in cols]

    return {
        "missing_raw_columns": missing_raw,
        "missing_engineered_columns": missing_engineered,
    }


def consensus_decision(vote_count: int) -> str:
    if vote_count >= 3:
        return "BLOCK"
    if vote_count == 2:
        return "MANUAL REVIEW"
    return "APPROVE"


def style_decision(decision: str):
    if decision == "BLOCK":
        return "🚫 BLOCK", "High fraud risk"
    if decision == "MANUAL REVIEW":
        return "⚠️ MANUAL REVIEW", "Medium fraud risk"
    return "✅ APPROVE", "Low fraud risk"


def prepare_single_transaction(
    trans_date,
    trans_time,
    category,
    amount,
    gender,
    city_pop,
    dob,
    lat,
    long,
    merch_lat,
    merch_long,
):
    transaction_datetime = f"{trans_date} {trans_time}"

    return pd.DataFrame(
        [
            {
                "trans_date_trans_time": transaction_datetime,
                "category": category,
                "amt": float(amount),
                "gender": gender,
                "city_pop": int(city_pop),
                "dob": str(dob),
                "lat": float(lat),
                "long": float(long),
                "merch_lat": float(merch_lat),
                "merch_long": float(merch_long),
            }
        ]
    )


def parse_float_input(value, field_name: str) -> float:
    """Parse text inputs like 1000, 1,000, or 35.2698 into float."""
    text = str(value).strip().replace(",", "")
    if text == "":
        raise ValueError(f"{field_name} cannot be empty.")
    return float(text)


def parse_int_input(value, field_name: str) -> int:
    """Parse text inputs like 50000 or 50,000 into int."""
    number = parse_float_input(value, field_name)
    return int(round(number))


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_artifacts(models_dir: Path = MODELS_DIR):
    """
    Loads trained model pipelines and thresholds.

    Supported saved files:
    1. Optional package:
       models/consensus_package.pkl

    2. Individual files:
       models/lr_model.pkl, models/rf_model.pkl, models/xgb_model.pkl, models/lgbm_model.pkl
       models/lr_threshold.pkl, models/rf_threshold.pkl, models/xgb_threshold.pkl, models/lgbm_threshold.pkl

    3. Optional JSON thresholds:
       models/thresholds.json
    """
    models_dir = Path(models_dir)

    errors = []
    models = {}
    thresholds = DEFAULT_THRESHOLDS.copy()

    if not models_dir.exists():
        return models, thresholds, [f"Missing models folder: {models_dir}"]

    # 1. Try consensus package first, but do not stop there.
    package_path = models_dir / "consensus_package.pkl"
    if package_path.exists():
        try:
            package = joblib.load(package_path)

            if isinstance(package, dict):
                packaged_models = package.get("models", {})
                packaged_thresholds = package.get("thresholds", {})

                if isinstance(packaged_models, dict):
                    for raw_key, model in packaged_models.items():
                        key = normalize_model_key(raw_key)
                        models[key] = model

                if isinstance(packaged_thresholds, dict):
                    for raw_key, threshold in packaged_thresholds.items():
                        key = normalize_model_key(raw_key)
                        thresholds[key] = safe_float(threshold, thresholds.get(key, 0.50))

        except Exception as exc:
            errors.append(f"Could not load consensus_package.pkl: {type(exc).__name__}: {exc}")

    # 2. Load individual model files. These fill missing models or replace failed package loading.
    for key, filename in MODEL_FILES.items():
        path = models_dir / filename
        if path.exists() and key not in models:
            try:
                models[key] = joblib.load(path)
            except Exception as exc:
                errors.append(f"Could not load {filename}: {type(exc).__name__}: {exc}")
        elif not path.exists() and key not in models:
            errors.append(f"Missing model file: {filename}")

    # 3. Load individual threshold files.
    for key, filename in THRESHOLD_FILES.items():
        path = models_dir / filename
        if path.exists():
            try:
                thresholds[key] = safe_float(joblib.load(path), thresholds.get(key, 0.50))
            except Exception as exc:
                errors.append(f"Could not load {filename}: {type(exc).__name__}: {exc}")

    # 4. Optional thresholds.json overrides or fills thresholds.
    json_threshold_path = models_dir / "thresholds.json"
    if json_threshold_path.exists():
        try:
            with open(json_threshold_path, "r", encoding="utf-8") as f:
                json_thresholds = json.load(f)

            if isinstance(json_thresholds, dict):
                for raw_key, value in json_thresholds.items():
                    key = normalize_model_key(raw_key)
                    thresholds[key] = safe_float(value, thresholds.get(key, 0.50))

        except Exception as exc:
            errors.append(f"Could not load thresholds.json: {type(exc).__name__}: {exc}")

    # Keep only known model keys in normal display order, then append any extra model keys.
    ordered_models = {}
    for key in MODEL_FILES.keys():
        if key in models:
            ordered_models[key] = models[key]

    for key, model in models.items():
        if key not in ordered_models:
            ordered_models[key] = model

    return ordered_models, thresholds, errors


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def get_fraud_probabilities(model, raw_df: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(raw_df)
        probabilities = np.asarray(probabilities)

        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]

        if probabilities.ndim == 1:
            return probabilities

    raise ValueError("This model does not support predict_proba().")


def predict_with_consensus(raw_df: pd.DataFrame, models: dict, thresholds: dict):
    if not models:
        raise ValueError("No trained models were loaded.")

    result_df = raw_df.copy()
    probability_columns = []
    prediction_columns = []

    for key, model in models.items():
        display_name = MODEL_DISPLAY_NAMES.get(key, str(key).upper())
        probability_col = f"{display_name} Probability"
        prediction_col = f"{display_name} Prediction"

        probabilities = get_fraud_probabilities(model, raw_df)
        threshold = safe_float(thresholds.get(key, 0.50), 0.50)
        predictions = (probabilities >= threshold).astype(int)

        result_df[probability_col] = probabilities
        result_df[prediction_col] = predictions

        probability_columns.append(probability_col)
        prediction_columns.append(prediction_col)

    result_df["Fraud Votes"] = result_df[prediction_columns].sum(axis=1).astype(int)
    result_df["Consensus Decision"] = result_df["Fraud Votes"].apply(consensus_decision)
    result_df["Average Fraud Probability"] = result_df[probability_columns].mean(axis=1)
    result_df["Highest Fraud Probability"] = result_df[probability_columns].max(axis=1)

    return result_df, probability_columns, prediction_columns


def build_model_summary(models: dict, thresholds: dict):
    rows = []

    for key in MODEL_FILES.keys():
        rows.append(
            {
                "Model": MODEL_DISPLAY_NAMES.get(key, key.upper()),
                "Loaded": "Yes" if key in models else "No",
                "Threshold": thresholds.get(key, "Not found"),
            }
        )

    # Show any extra keys that came from a package.
    for key in models.keys():
        if key not in MODEL_FILES:
            rows.append(
                {
                    "Model": MODEL_DISPLAY_NAMES.get(key, str(key)),
                    "Loaded": "Yes",
                    "Threshold": thresholds.get(key, "Not found"),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# SHAP EXPLAINABILITY FUNCTIONS
# =============================================================================

TREE_SHAP_MODEL_PRIORITY = ["xgb", "lgbm", "rf"]


def clean_feature_name(feature_name: str) -> str:
    """Make transformed feature names easier for users to understand."""
    name = str(feature_name)

    # Remove ColumnTransformer prefixes such as num__ and cat__.
    if "__" in name:
        name = name.split("__", 1)[1]

    # Convert one-hot encoded names into readable labels.
    for cat_col in CATEGORICAL_FEATURES:
        prefix = f"{cat_col}_"
        if name.startswith(prefix):
            return f"{cat_col} = {name[len(prefix):]}"

    readable_names = {
        "amt": "Transaction Amount",
        "city_pop": "City Population",
        "hour": "Transaction Hour",
        "day": "Day of Month",
        "month": "Transaction Month",
        "dayofweek": "Day of Week",
        "age": "Cardholder Age",
        "distance_km": "Distance Between Cardholder and Merchant (km)",
        "category": "Transaction Category",
        "gender": "Gender",
    }
    return readable_names.get(name, name)


def get_final_estimator(model):
    """Return the classifier/regressor at the end of a sklearn/imblearn pipeline."""
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    return model


def transform_input_for_shap(model, raw_df: pd.DataFrame):
    """
    Run the same preprocessing steps used by the pipeline, then return the final
    transformed matrix and feature names for SHAP.
    """
    X = raw_df.copy()
    feature_names = list(X.columns)

    # If the saved object is not a pipeline, use the engineered features directly.
    if not hasattr(model, "steps"):
        X = SafiCardFeatureEngineer().transform(X)
        return X.values, list(X.columns)

    for _, step in model.steps[:-1]:
        # Transform normal preprocessing steps.
        if hasattr(step, "transform"):
            X = step.transform(X)

            if hasattr(X, "columns"):
                feature_names = list(X.columns)
            elif hasattr(step, "get_feature_names_out"):
                try:
                    feature_names = list(step.get_feature_names_out())
                except Exception:
                    try:
                        feature_names = list(step.get_feature_names_out(feature_names))
                    except Exception:
                        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Skip samplers such as SMOTE during inference/explainability.
        elif hasattr(step, "fit_resample"):
            continue

    if hasattr(X, "toarray"):
        X = X.toarray()

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if len(feature_names) != X.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    return X, feature_names


def explain_transaction_with_shap(model, raw_df: pd.DataFrame, top_n: int = 10):
    """
    Use SHAP to explain one transaction. Positive SHAP values push the model
    toward fraud; negative SHAP values push the model toward legitimate.
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed in this Python environment. Install it with: python -m pip install shap"
        )

    final_estimator = get_final_estimator(model)
    X_transformed, feature_names = transform_input_for_shap(model, raw_df)

    explainer = shap.TreeExplainer(final_estimator)
    shap_values = explainer.shap_values(X_transformed)
    expected_value = explainer.expected_value

    # Different SHAP/model versions return different shapes. Normalize to class-1 values.
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif hasattr(shap_values, "values"):
        values = shap_values.values
    else:
        values = shap_values

    values = np.asarray(values)
    if values.ndim == 3:
        # Shape can be: rows x features x classes. Use fraud class index 1 when available.
        values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]

    values = values.reshape(X_transformed.shape[0], -1)[0]
    feature_values = np.asarray(X_transformed[0]).reshape(-1)

    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_arr = np.asarray(expected_value).reshape(-1)
        base_value = expected_arr[1] if len(expected_arr) > 1 else expected_arr[0]
    else:
        base_value = expected_value

    explanation_df = pd.DataFrame(
        {
            "Feature": [clean_feature_name(name) for name in feature_names],
            "Feature Value": feature_values,
            "SHAP Contribution": values,
        }
    )

    explanation_df["Effect"] = np.where(
        explanation_df["SHAP Contribution"] > 0,
        "Increases fraud risk",
        "Reduces fraud risk",
    )

    explanation_df["Absolute Impact"] = explanation_df["SHAP Contribution"].abs()
    explanation_df = explanation_df.sort_values("Absolute Impact", ascending=False).head(top_n)

    return explanation_df.drop(columns=["Absolute Impact"]), base_value


def choose_default_shap_model(models: dict):
    for key in TREE_SHAP_MODEL_PRIORITY:
        if key in models:
            return key
    return next(iter(models.keys()), None)


def show_shap_explanation(raw_transaction: pd.DataFrame, models: dict):
    st.subheader("SHAP Explainability")
    st.write(
        "SHAP explains which features pushed the selected model toward **fraud** or toward **legitimate** for this transaction."
    )

    if not SHAP_AVAILABLE:
        st.warning(
            f"SHAP is not installed or could not be imported: {type(SHAP_IMPORT_ERROR).__name__}: {SHAP_IMPORT_ERROR}"
        )
        st.code("python -m pip install shap", language="bash")
        return

    explainable_keys = [key for key in TREE_SHAP_MODEL_PRIORITY if key in models]
    if not explainable_keys:
        st.info("SHAP explainability is enabled for XGBoost, LightGBM, or Random Forest models.")
        return

    selected_key = choose_default_shap_model(models)
    selected_model = models[selected_key]
    selected_name = MODEL_DISPLAY_NAMES.get(selected_key, selected_key.upper())

    try:
        shap_table, base_value = explain_transaction_with_shap(selected_model, raw_transaction, top_n=10)

        st.caption(
            f"Explanation generated using **{selected_name}**. Positive values increase fraud risk; negative values reduce fraud risk."
        )

        chart_data = shap_table[["Feature", "SHAP Contribution"]].copy()
        chart_data = chart_data.sort_values("SHAP Contribution")
        st.bar_chart(chart_data.set_index("Feature"))

        display_table = shap_table.copy()
        display_table["SHAP Contribution"] = display_table["SHAP Contribution"].map(lambda x: f"{x:.6f}")
        st.dataframe(display_table, use_container_width=True, hide_index=True)

        positive_reasons = shap_table[shap_table["Effect"] == "Increases fraud risk"].head(3)
        negative_reasons = shap_table[shap_table["Effect"] == "Reduces fraud risk"].head(3)

        if not positive_reasons.empty:
            st.markdown("**Main reasons increasing fraud risk:**")
            for _, reason in positive_reasons.iterrows():
                st.write(f"- {reason['Feature']} contributed positively to the fraud score.")

        if not negative_reasons.empty:
            st.markdown("**Main reasons reducing fraud risk:**")
            for _, reason in negative_reasons.iterrows():
                st.write(f"- {reason['Feature']} contributed negatively to the fraud score.")

        with st.expander("Technical note"):
            st.write(
                "SHAP values show each feature's contribution to the model output. The scale may be log-odds for some tree models, "
                "so use the sign and relative size to understand the explanation."
            )
            st.write(f"Base value: {safe_float(base_value, 0.0):.6f}")

    except Exception as exc:
        st.error(f"SHAP explanation failed: {type(exc).__name__}: {exc}")
        st.info(
            "This can happen if the saved model pipeline structure is different from expected or if SHAP does not support that model object directly."
        )


# =============================================================================
# UI
# =============================================================================

st.title("💳 Streamlit based application for Credit Card Fraud Detection System")
st.caption("Streamlit hosting app for the already-trained fraud detection pipelines.")

models, thresholds, load_errors = load_artifacts(MODELS_DIR)

with st.sidebar:
    st.header("Model Status")
    st.dataframe(build_model_summary(models, thresholds), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("App Paths")
    st.code(
        f"""App folder:
{BASE_DIR}

Models folder:
{MODELS_DIR}""",
        language="text",
    )

    st.subheader("Files found in models/")
    found_files = available_model_files(MODELS_DIR)
    if found_files:
        st.code("\n".join(found_files), language="text")
    else:
        st.code("No files found or models folder is missing.", language="text")

    st.divider()
    st.subheader("Expected model folder")
    st.code(
        """models/
  consensus_package.pkl   # optional

  lr_model.pkl
  rf_model.pkl
  xgb_model.pkl
  lgbm_model.pkl

  lr_threshold.pkl
  rf_threshold.pkl
  xgb_threshold.pkl
  lgbm_threshold.pkl
  thresholds.json         # optional""",
        language="text",
    )

    if load_errors:
        st.warning("Some files had loading issues.")
        for err in load_errors:
            st.caption(err)

if not models:
    st.error(
        "No trained models were loaded. The app found the folder, but Python could not open the `.pkl` models. "
        "Check the loading errors in the sidebar. The most common causes are missing `lightgbm`/`xgboost` packages "
        "or an older app.py without the `SafiCardFeatureEngineer` class."
    )
    st.stop()

tab_single, tab_batch, tab_about = st.tabs(
    ["Single Transaction Prediction", "Batch CSV Prediction", "About the System"]
)

with tab_single:
    st.subheader("Enter Transaction Details")

    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            trans_date = st.date_input("Transaction Date")
            trans_time = st.time_input("Transaction Time")
            category = st.selectbox("Transaction Category", CATEGORY_OPTIONS, index=0)
            amount = st.text_input("Transaction Amount", value="100.00", help="Type the amount directly, for example 1000 or 2500.50")

        with col2:
            gender = st.selectbox("Gender", ["M", "F"])
            city_pop = st.text_input("City Population", value="50000", help="Type the population directly, for example 50000 or 1,000,000")
            dob = st.date_input("Cardholder Date of Birth")

        with col3:
            lat = st.text_input("Cardholder Latitude", value="0.514300")
            long = st.text_input("Cardholder Longitude", value="35.269800")
            merch_lat = st.text_input("Merchant Latitude", value="0.520000")
            merch_long = st.text_input("Merchant Longitude", value="35.280000")

        submitted = st.form_submit_button("Predict Fraud Risk", type="primary")

    if submitted:
        try:
            amount_value = parse_float_input(amount, "Transaction Amount")
            city_pop_value = parse_int_input(city_pop, "City Population")
            lat_value = parse_float_input(lat, "Cardholder Latitude")
            long_value = parse_float_input(long, "Cardholder Longitude")
            merch_lat_value = parse_float_input(merch_lat, "Merchant Latitude")
            merch_long_value = parse_float_input(merch_long, "Merchant Longitude")

            raw_transaction = prepare_single_transaction(
                trans_date=trans_date,
                trans_time=trans_time,
                category=category,
                amount=amount_value,
                gender=gender,
                city_pop=city_pop_value,
                dob=dob,
                lat=lat_value,
                long=long_value,
                merch_lat=merch_lat_value,
                merch_long=merch_long_value,
            )

            prediction_result, probability_cols, prediction_cols = predict_with_consensus(
                raw_transaction,
                models,
                thresholds,
            )

            row = prediction_result.iloc[0]
            decision = row["Consensus Decision"]
            decision_label, risk_label = style_decision(decision)

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("Consensus Decision", decision_label)
            metric_col2.metric("Risk Level", risk_label)
            metric_col3.metric("Fraud Votes", f"{int(row['Fraud Votes'])} / {len(models)}")
            metric_col4.metric("Highest Fraud Probability", f"{row['Highest Fraud Probability']:.2%}")

            st.info(DECISION_DETAILS.get(decision, ""))

            model_rows = []
            for key in models.keys():
                display_name = MODEL_DISPLAY_NAMES.get(key, str(key).upper())
                model_rows.append(
                    {
                        "Model": display_name,
                        "Fraud Probability": row[f"{display_name} Probability"],
                        "Saved Threshold": safe_float(thresholds.get(key, 0.50)),
                        "Prediction": "Fraud" if row[f"{display_name} Prediction"] == 1 else "Legitimate",
                    }
                )

            st.subheader("Individual Model Results")
            st.dataframe(
                pd.DataFrame(model_rows),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("View Raw Input Sent to Model"):
                st.dataframe(raw_transaction, use_container_width=True, hide_index=True)

            show_shap_explanation(raw_transaction, models)

        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")

with tab_batch:
    st.subheader("Upload CSV for Batch Prediction")

    st.write(
        "Your CSV can contain either the raw columns or the final engineered columns."
    )

    with st.expander("Accepted CSV column formats"):
        st.markdown(
            """
            **Raw CSV columns:**
            `trans_date_trans_time`, `category`, `amt`, `gender`, `city_pop`, `dob`,
            `lat`, `long`, `merch_lat`, `merch_long`

            **Or already-engineered columns:**
            `category`, `amt`, `gender`, `city_pop`, `hour`, `day`, `month`,
            `dayofweek`, `age`, `distance_km`
            """
        )

    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            missing_info = validate_input_columns(batch_df)

            if missing_info:
                st.error(
                    "Your CSV does not have a complete raw-column set or a complete engineered-column set."
                )
                st.json(missing_info)
            else:
                batch_results, _, _ = predict_with_consensus(batch_df, models, thresholds)

                st.success(f"Predicted {len(batch_results):,} transactions.")

                # Show only a small preview in the browser. Showing the full table
                # can cause Streamlit MessageSizeError for large CSV files.
                preview_rows = min(1000, len(batch_results))
                st.info(
                    f"Showing only the first {preview_rows:,} rows to keep the app fast. "
                    "The full prediction results are saved in your app folder."
                )
                st.dataframe(batch_results.head(preview_rows), use_container_width=True)

                # Save full output locally instead of sending hundreds of MB to the browser.
                output_path = BASE_DIR / "safecard_predictions_full.csv"
                batch_results.to_csv(output_path, index=False)
                st.success(f"Full predictions saved here: {output_path}")

                # Provide a safe small preview download only.
                preview_csv = batch_results.head(preview_rows).to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Preview CSV",
                    data=preview_csv,
                    file_name="safecard_predictions_preview.csv",
                    mime="text/csv",
                )

        except Exception as exc:
            st.error(f"Could not process the uploaded CSV: {type(exc).__name__}: {exc}")

with tab_about:
    st.subheader("How This App Works")

    st.write(
        """
        This app hosts the trained SafiCard-style fraud detection models from your notebook.
        It does not train models again. It only loads the saved pipelines and thresholds from
        the `models/` folder.
        """
    )

    st.markdown(
        """
        **Pipeline used by each saved model:**

        Raw transaction data → `SafiCardFeatureEngineer` → Preprocessing → Model prediction

        **Final engineered features:**
        - `category`
        - `amt`
        - `gender`
        - `city_pop`
        - `hour`
        - `day`
        - `month`
        - `dayofweek`
        - `age`
        - `distance_km`

        **Consensus rule:**
        - **BLOCK**: 3 or 4 models vote fraud
        - **MANUAL REVIEW**: exactly 2 models vote fraud
        - **APPROVE**: 0 or 1 model votes fraud

        **SHAP explainability:**
        - Shows the main features that pushed the prediction toward fraud
        - Positive SHAP values increase fraud risk
        - Negative SHAP values reduce fraud risk
        """
    )

    st.subheader("Run Command")
    st.code("python -m streamlit run app.py", language="bash")

    st.subheader("Important Model Loading Fix")
    st.code(
        """# This class must exist before joblib.load(...)
class SafiCardFeatureEngineer(BaseEstimator, TransformerMixin):
    ...

import __main__
setattr(__main__, "SafiCardFeatureEngineer", SafiCardFeatureEngineer)""",
        language="python",
    )
