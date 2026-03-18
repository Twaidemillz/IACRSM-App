# =============================================================================
# utils/predictor.py
# Prediction logic — single and batch scoring
# Intelligent Adaptive Credit Risk Scoring Model
# =============================================================================

import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DISPLAY_NAMES = {
    "LogisticRegression":     "Logistic Regression",
    "DecisionTreeClassifier": "Decision Tree",
    "RandomForestClassifier": "Random Forest",
    "XGBClassifier":          "XGBoost",
}

METRIC_DESCRIPTIONS = {
    "ROC-AUC":   "Area under the ROC curve. Primary ranking metric (target >= 0.75).",
    "Recall":    "True positive rate — proportion of actual defaults correctly identified.",
    "Precision": "Of predicted defaults, proportion that were actual defaults.",
    "F1":        "Harmonic mean of Precision and Recall.",
    "Accuracy":  "Overall proportion of correct predictions.",
    "MCC":       "Matthews Correlation Coefficient — robust metric for imbalanced datasets.",
}

NUMERIC_FEATURES = {
    "duration":    {"label": "Loan Duration (months)",     "min": 4,   "max": 72,    "default": 18,   "step": 1},
    "amount":      {"label": "Credit Amount (DM)",         "min": 250, "max": 20000, "default": 3000, "step": 50},
    "installment": {"label": "Installment Rate (%)",       "min": 1,   "max": 4,     "default": 2,    "step": 1},
    "residence":   {"label": "Residence Since (years)",    "min": 1,   "max": 4,     "default": 2,    "step": 1},
    "age":         {"label": "Age (years)",                "min": 18,  "max": 75,    "default": 35,   "step": 1},
    "cards":       {"label": "Number of Existing Credits", "min": 1,   "max": 4,     "default": 1,    "step": 1},
    "liable":      {"label": "Number of Dependents",       "min": 1,   "max": 2,     "default": 1,    "step": 1},
}

CATEGORICAL_FEATURES = {
    "checkingstatus1": {
        "label":   "Checking Account Status",
        "options": ["A11", "A12", "A13", "A14"],
        "display": ["< 0 DM (Negative)", "0-200 DM", "> 200 DM", "No Checking Account"],
    },
    "history": {
        "label":   "Credit History",
        "options": ["A30", "A31", "A32", "A33", "A34"],
        "display": ["No Credits Taken", "All Credits Paid at This Bank",
                    "Existing Credits Paid", "Delayed Previously",
                    "Critical / Other Credits"],
    },
    "purpose": {
        "label":   "Loan Purpose",
        "options": ["A40", "A41", "A42", "A43", "A44", "A45",
                    "A46", "A48", "A49", "A410"],
        "display": ["New Car", "Used Car", "Furniture/Equipment", "Radio/TV",
                    "Domestic Appliances", "Repairs", "Education",
                    "Retraining", "Business", "Others"],
    },
    "savings": {
        "label":   "Savings Account Balance",
        "options": ["A61", "A62", "A63", "A64", "A65"],
        "display": ["< 100 DM", "100-500 DM", "500-1000 DM", "> 1000 DM", "No Known Savings"],
    },
    "employ": {
        "label":   "Employment Since",
        "options": ["A71", "A72", "A73", "A74", "A75"],
        "display": ["Unemployed", "< 1 Year", "1-4 Years", "4-7 Years", "> 7 Years"],
    },
    "status": {
        "label":   "Personal Status & Gender",
        "options": ["A91", "A92", "A93", "A94"],
        "display": ["Male / Divorced", "Female / Divorced or Married",
                    "Male / Single", "Male / Married"],
    },
    "others": {
        "label":   "Other Debtors / Guarantors",
        "options": ["A101", "A102", "A103"],
        "display": ["None", "Co-applicant", "Guarantor"],
    },
    "property": {
        "label":   "Property / Collateral",
        "options": ["A121", "A122", "A123", "A124"],
        "display": ["Real Estate", "Life Insurance", "Car / Other", "No Known Property"],
    },
    "otherplans": {
        "label":   "Other Payment Plans",
        "options": ["A141", "A142", "A143"],
        "display": ["At Bank", "At Stores", "None"],
    },
    "housing": {
        "label":   "Housing",
        "options": ["A151", "A152", "A153"],
        "display": ["Rent", "Own", "Free"],
    },
    "job": {
        "label":   "Employment Type",
        "options": ["A171", "A172", "A173", "A174"],
        "display": ["Unskilled / Non-Resident", "Unskilled / Resident",
                    "Skilled Employee", "Highly Skilled"],
    },
    "tele": {
        "label":   "Own Telephone",
        "options": ["A191", "A192"],
        "display": ["No", "Yes"],
    },
    "foreign": {
        "label":   "Foreign Worker",
        "options": ["A201", "A202"],
        "display": ["Yes", "No"],
    },
}

SCALED_COLS = ["duration", "amount", "installment", "residence", "age", "cards", "liable"]


@st.cache_resource(show_spinner=False)
def load_artefacts(artefacts_dir: str = "artefacts") -> dict:
    try:
        artefacts = {}
        artefacts["X_train"] = pd.read_csv(f"{artefacts_dir}/X_train.csv")
        artefacts["X_test"]  = pd.read_csv(f"{artefacts_dir}/X_test.csv")
        artefacts["y_train"] = pd.read_csv(f"{artefacts_dir}/y_train.csv").squeeze()
        artefacts["y_test"]  = pd.read_csv(f"{artefacts_dir}/y_test.csv").squeeze()
        artefacts["feature_names"] = artefacts["X_train"].columns.tolist()
        artefacts["best_model"]      = joblib.load(f"{artefacts_dir}/best_model.pkl")
        artefacts["best_model_name"] = joblib.load(f"{artefacts_dir}/best_model_name.pkl")
        artefacts["class_weights"]   = joblib.load(f"{artefacts_dir}/class_weights.pkl")
        artefacts["scaler"]         = joblib.load(f"{artefacts_dir}/scaler.pkl")
        artefacts["label_encoders"] = joblib.load(f"{artefacts_dir}/label_encoders.pkl")

        model_map  = {}
        model_dirs = {
            "LogisticRegression":     "logistic_regression",
            "DecisionTreeClassifier": "decision_tree",
            "RandomForestClassifier": "random_forest",
            "XGBClassifier":          "xgboost",
        }
        for cls_name, folder in model_dirs.items():
            path = f"{artefacts_dir}/models/{folder}/{folder}.pkl"
            if os.path.exists(path):
                model_map[cls_name] = joblib.load(path)
        if not model_map:
            model_map[artefacts["best_model_name"]] = artefacts["best_model"]
        artefacts["models"] = model_map

        ver_path = f"{artefacts_dir}/versions/version_registry.json"
        if os.path.exists(ver_path):
            import json
            with open(ver_path) as f:
                artefacts["version_registry"] = json.load(f)
        else:
            artefacts["version_registry"] = {}

        log_path = f"{artefacts_dir}/psi/monitoring_log.csv"
        artefacts["monitoring_log"] = (
            pd.read_csv(log_path) if os.path.exists(log_path) else None
        )
        artefacts["loaded"] = True
        return artefacts
    except Exception as e:
        return {"loaded": False, "error": str(e)}


def build_input_df(form_values: dict,
                   feature_names: list,
                   label_encoders: dict,
                   scaler) -> pd.DataFrame:
    row = {}
    for col, le in label_encoders.items():
        raw_val = form_values.get(col, le.classes_[0])
        row[col] = int(le.transform([raw_val])[0]) if raw_val in le.classes_ else 0
    for col in SCALED_COLS:
        row[col] = float(form_values.get(col, 0))
    df_raw = pd.DataFrame([row])[feature_names]
    df_scaled = df_raw.copy()
    df_scaled[SCALED_COLS] = scaler.transform(df_raw[SCALED_COLS])
    return df_scaled


def predict_single(model, input_df: pd.DataFrame):
    prob_array   = model.predict_proba(input_df)
    prob_default = float(prob_array[0, 1])
    pred_class   = int(model.predict(input_df)[0])
    return prob_default, pred_class, prob_array[0]


def predict_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    results = df.copy()
    probs   = model.predict_proba(df)[:, 1]
    preds   = model.predict(df)
    results["prob_default"]    = np.round(probs, 4)
    results["predicted_class"] = preds
    results["risk_band"]       = pd.cut(
        probs, bins=[0, 0.35, 0.60, 1.0],
        labels=["Low", "Moderate", "High"], include_lowest=True
    )
    return results


def validate_batch_csv(df: pd.DataFrame, feature_names: list) -> tuple:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        return False, f"Missing {len(missing)} columns: {missing[:5]}"
    return True, "Valid"


def batch_csv_template(feature_names: list) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(columns=feature_names).to_csv(buf, index=False)
    return buf.getvalue()
