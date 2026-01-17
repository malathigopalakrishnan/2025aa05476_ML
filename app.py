

# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Optional: XGBoost
XGB_AVAILABLE = True
try:
    from xgboost import XGBClassifier
except Exception:
    XGB_AVAILABLE = False

RSEED = 42

st.set_page_config(page_title="ML Assignment - Streamlit App", layout="wide")

# -----------------------------
# Helper functions
# -----------------------------
def detect_feature_types(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_cols, categorical_cols

def make_preprocessor(numeric_cols, categorical_cols):
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor

def prepare_models(n_classes):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RSEED),
        "Decision Tree": DecisionTreeClassifier(random_state=RSEED),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=300, random_state=RSEED),
    }
    if XGB_AVAILABLE:
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        extra = {} if n_classes == 2 else {"num_class": n_classes}
        models["XGBoost (Ensemble)"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective=objective, eval_metric="logloss",
            random_state=RSEED, n_jobs=-1, tree_method="hist", **extra
        )
    return models

def compute_metrics(y_true_enc, y_pred_enc, y_proba, n_classes):
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true_enc, y_pred_enc)

    if n_classes == 2:
        metrics["Precision"] = precision_score(y_true_enc, y_pred_enc, average="binary", zero_division=0)
        metrics["Recall"] = recall_score(y_true_enc, y_pred_enc, average="binary", zero_division=0)
        metrics["F1"] = f1_score(y_true_enc, y_pred_enc, average="binary", zero_division=0)
        metrics["AUC"] = roc_auc_score(y_true_enc, y_proba[:, 1]) if (y_proba is not None and y_proba.shape[1] >= 2) else np.nan
    else:
        metrics["Precision"] = precision_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)
        metrics["Recall"] = recall_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)
        metrics["F1"] = f1_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)
        metrics["AUC"] = roc_auc_score(y_true_enc, y_proba, multi_class="ovr", average="macro") if (y_proba is not None and y_proba.shape[1] == n_classes) else np.nan

    metrics["MCC"] = matthews_corrcoef(y_true_enc, y_pred_enc)
    return metrics

def format_metrics(metrics):
    return {k: (round(v, 4) if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()}

# -----------------------------
# Sidebar: Upload & Settings
# -----------------------------
st.sidebar.title("Configuration")

uploaded = st.sidebar.file_uploader("Upload CSV (test data allowed)", type=["csv"])
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
random_state = st.sidebar.number_input("Random seed", min_value=0, value=RSEED, step=1)

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("### Preview of uploaded data")
    st.dataframe(df.head())

    # Target selection
  
    target_col = st.selectbox("Select target column", options=df.columns.tolist(),index=len(df.columns) - 1)
    if target_col:
        y_raw = df[target_col]
        X = df.drop(columns=[target_col])

        # Encode target
        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw)
        class_names = le.classes_
        n_classes = len(class_names)

        # Train-test split
        X_train, X_test, y_train_enc, y_test_enc = train_test_split(
            X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
        )

        # Models and preprocessing
        models = prepare_models(n_classes)
        model_name = st.selectbox("Choose a model", options=list(models.keys()))
        num_cols, cat_cols = detect_feature_types(X)
        preprocessor = make_preprocessor(num_cols, cat_cols)

        # Build pipeline
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", models[model_name])])

        # Train
        st.write(f"### Training: {model_name}")
        pipeline.fit(X_train, y_train_enc)

        # Predict + probabilities
        y_pred = pipeline.predict(X_test)
        y_proba = None
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception:
            pass

        # Metrics
        metrics = compute_metrics(y_test_enc, y_pred, y_proba, n_classes)
        st.subheader("Evaluation Metrics")
        st.write(format_metrics(metrics))

        # Classification report
        st.subheader("Classification Report")
        report_text = classification_report(
            y_test_enc, y_pred, target_names=[str(c) for c in class_names], zero_division=0
        )
        st.code(report_text, language="text")

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test_enc, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

        # Download metrics as CSV (for README table)
        results_row = {
            "ML Model Name": model_name,
            "Accuracy": metrics["Accuracy"],
            "AUC": metrics["AUC"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "MCC": metrics["MCC"]
        }
        out_df = pd.DataFrame([results_row])
        csv_buf = io.StringIO()
        out_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download this model’s metrics (CSV)",
            data=csv_buf.getvalue(),
            file_name=f"{model_name.replace(' ', '_').lower()}_metrics.csv",
            mime="text/csv"
        )
else:
    st.info("Upload a CSV to begin. Use test data if your dataset is large (Streamlit free tier has limits).")


