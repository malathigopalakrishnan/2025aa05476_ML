
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Diabetes Prediction - ML Assignment 2", layout="wide")

MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")


@st.cache_resource
def load_assets():
    # Load metadata
    meta_path = MODEL_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            "metadata.json not found. Run the training notebook first to create model/ artifacts."
        )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # Load metrics (optional)
    metrics_path = RESULTS_DIR / "metrics.csv"
    metrics_df = None
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)

    # Load all joblib models
    models = {}
    for p in MODEL_DIR.glob("*.joblib"):
        pretty = p.stem.replace("_", " ").title()
        models[pretty] = joblib.load(p)

    return metadata, metrics_df, models


def clean_columns(cols):
    out = []
    for c in cols:
        c2 = str(c).strip().replace(" ", "_").replace("-", "_")
        out.append(c2)
    return out


def preprocess_dataframe(df_in: pd.DataFrame, feature_order):
    """Apply the same preprocessing used during training."""
    dfp = df_in.copy()
    dfp.columns = clean_columns(dfp.columns)

    # Map Gender
    if "Gender" in dfp.columns:
        dfp["Gender"] = dfp["Gender"].map(
            {"Male": 1, "Female": 0, "male": 1, "female": 0}
        )

    # Map Yes/No
    yn_map = {"Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0}

    for c in dfp.columns:
        if c == "class":
            continue
        if dfp[c].dtype == "object":
            uniq = set(dfp[c].dropna().unique().tolist())
            if uniq.issubset(set(["Yes", "No", "yes", "no"])):
                dfp[c] = dfp[c].map(yn_map)

    # Age numeric
    if "Age" in dfp.columns:
        dfp["Age"] = pd.to_numeric(dfp["Age"], errors="coerce")

    # Ensure all expected cols exist
    missing = [c for c in feature_order if c not in dfp.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = dfp[feature_order].copy()
    X = X.dropna()

    return X


def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


# ---------------------- STREAMLIT UI ------------------------

st.title("🧪 Diabetes Classification — ML Assignment 2")
st.caption("Upload a CSV (preferably test data). Select a model and view predictions + metrics.")

try:
    metadata, metrics_df, models = load_assets()
except Exception as e:
    st.error(str(e))
    st.stop()

feature_order = metadata["feature_order"]

with st.expander("📌 Assignment-required metrics table (from local hold-out test)", expanded=True):
    if metrics_df is not None:
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("metrics.csv not found. Run the training notebook to generate results/metrics.csv")


# ----------------------- UPLOAD CSV -------------------------
st.subheader("1) Upload CSV")
file = st.file_uploader("Upload your CSV file", type=["csv"])

# ----------------------- MODEL SELECTION ---------------------
st.subheader("2) Choose a Model")
model_name = st.selectbox("Select model", options=sorted(models.keys()))
model = models[model_name]

if file is None:
    st.warning("Please upload a CSV to continue.")
    st.stop()

raw_df = pd.read_csv(file)
raw_df.columns = clean_columns(raw_df.columns)

has_label = "class" in raw_df.columns

try:
    X = preprocess_dataframe(raw_df, feature_order)
except Exception as e:
    st.error(f"Preprocessing error: {e}")
    st.stop()

# Align y to X (after dropna)
if has_label:
    y_raw = raw_df.loc[X.index, "class"]
    y_true = y_raw.map(
        {"Positive": 1, "Negative": 0, "positive": 1, "negative": 0}
    ).astype("float")

    valid_idx = y_true.dropna().index
    X = X.loc[valid_idx]
    y_true = y_true.loc[valid_idx].astype(int)

# ----------------------- PREDICT ----------------------------
pred = model.predict(X)
proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

pred_label = np.where(pred == 1, "Positive", "Negative")

out = X.copy()
out["predicted_class"] = pred_label
if proba is not None:
    out["prob_positive"] = proba

st.subheader("3) Predictions")
st.dataframe(out.head(50), use_container_width=True)

st.download_button(
    label="⬇️ Download predictions CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv"
)

# ----------------------- METRICS ----------------------------
if has_label and len(y_true) > 0:
    st.subheader("4) Evaluation on uploaded data")
