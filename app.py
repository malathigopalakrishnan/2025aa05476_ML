
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

import urllib.request
import os

# GitHub RAW dataset URL
GITHUB_DATA_URL = (
    "https://raw.githubusercontent.com/malathigopalakrishnan/"
    "2025aa05476_ML/main/diabetes_data_upload.csv"
)

def download_dataset_from_github():
    """Download dataset from GitHub and save it in same folder as this .py file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "diabetes_data_upload.csv")

    try:
        urllib.request.urlretrieve(GITHUB_DATA_URL, save_path)
        return save_path
    except Exception as e:
        return str(e)


# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Diabetes Prediction - ML Assignment 2", layout="wide")

MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")


@st.cache_resource
def load_assets():
    meta_path = MODEL_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            "metadata.json not found. Run training notebook first to create model/"
        )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    metrics_path = RESULTS_DIR / "metrics.csv"
    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else None

    models = {}
    for p in MODEL_DIR.glob("*.joblib"):
        models[p.stem.replace("_", " ").title()] = joblib.load(p)

    return metadata, metrics_df, models


def clean_columns(cols):
    out = []
    for c in cols:
        c2 = str(c).strip().replace(" ", "_").replace("-", "_")
        out.append(c2)
    return out


def preprocess_dataframe(df_in: pd.DataFrame, feature_order):
    dfp = df_in.copy()
    dfp.columns = clean_columns(dfp.columns)

    # Map Gender
    if "Gender" in dfp.columns:
        dfp["Gender"] = dfp["Gender"].map(
            {"Male": 1, "Female": 0, "male": 1, "female": 0}
        )

    yn_map = {"Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0}
    for c in dfp.columns:
        if c == "class":
            continue
        if dfp[c].dtype == "object":
            uniq = set(dfp[c].dropna().unique().tolist())
            if uniq.issubset({"Yes", "No", "yes", "no"}):
                dfp[c] = dfp[c].map(yn_map)

    if "Age" in dfp.columns:
        dfp["Age"] = pd.to_numeric(dfp["Age"], errors="coerce")

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


# ---------------------------------------------------------
# UI START
# ---------------------------------------------------------
st.title("🧪 Diabetes Classification — ML Assignment 2")
st.caption("Upload a CSV OR download sample dataset. Select a model and view predictions + metrics.")


try:
    metadata, metrics_df, models = load_assets()
except Exception as e:
    st.error(str(e))
    st.stop()

feature_order = metadata["feature_order"]


# ---------------------------------------------------------
# DOWNLOAD DATASET BUTTON
# ---------------------------------------------------------
st.subheader("📥 Download Dataset from GitHub")

if st.button("⬇️ Download diabetes_data_upload.csv"):
    result = download_dataset_from_github()
    if os.path.exists(result):
        st.success(f"Dataset downloaded successfully at:\n{result}")
    else:
        st.error(f"Error downloading dataset:\n{result}")


# ---------------------------------------------------------
# UPLOAD CSV OR USE DOWNLOADED FILE
# ---------------------------------------------------------
st.subheader("1) Upload CSV")
file = st.file_uploader("Upload your CSV file", type=["csv"])

if file is None:
    local_dataset = "diabetes_data_upload.csv"
    if os.path.exists(local_dataset):
        st.success("Using downloaded GitHub dataset for prediction.")
        file = open(local_dataset, "rb")
    else:
        st.warning("Upload a CSV OR click the download button above.")
        st.stop()


# ---------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------
st.subheader("2) Choose a Model")
model_name = st.selectbox("Select model", options=sorted(models.keys()))
model = models[model_name]


# ---------------------------------------------------------
# LOAD AND PROCESS CSV
# ---------------------------------------------------------
raw_df = pd.read_csv(file)
raw_df.columns = clean_columns(raw_df.columns)

has_label = "class" in raw_df.columns

try:
    X = preprocess_dataframe(raw_df, feature_order)
except Exception as e:
    st.error(f"Preprocessing error: {e}")
    st.stop()


# ---------------------------------------------------------
# ALIGN LABELS
# ---------------------------------------------------------
if has_label:
    y_raw = raw_df.loc[X.index, "class"]
    y_true = y_raw.map(
        {"Positive": 1, "Negative": 0, "positive": 1, "negative": 0}
    ).astype("float")

    valid_idx = y_true.dropna().index
    y_true = y_true.loc[valid_idx].astype(int)
    X = X.loc[valid_idx]


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
pred = model.predict(X)
proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

pred_label = np.where(pred == 1, "Positive", "Negative")

out = X.copy()
out["predicted_class"] = pred_label
if proba is not None:
    out["prob_positive"] = proba

st.subheader("3) Predictions Preview")
st.dataframe(out.head(50), use_container_width=True)

st.download_button(
    label="⬇️ Download predictions CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv"
)


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
if has_label and len(y_true) > 0:
    st.subheader("4) Evaluation on Uploaded/Downloaded Data")

    auc = safe_auc(y_true, proba) if proba is not None else np.nan

    metrics = {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": auc,
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
    }

    st.subheader("📊 Metrics Summary")
    metrics_df_display = (
        pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
        .assign(Value=lambda df: df["Value"].apply(
            lambda x: f"{x:.4f}" if isinstance(x, float) else x
        ))
    )
    st.dataframe(metrics_df_display, use_container_width=True)

    cm = confusion_matrix(y_true, pred)
    st.write("**Confusion Matrix**")
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["True_Neg", "True_Pos"],
            columns=["Pred_Neg", "Pred_Pos"]
        ),
        use_container_width=True
    )

    st.write("**Classification Report**")
    st.code(
        classification_report(
            y_true, pred, target_names=["Negative", "Positive"]
        )
    )

else:
    st.info("Labels not found. Predictions only.")
