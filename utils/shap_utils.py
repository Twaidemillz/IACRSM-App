# =============================================================================
# utils/shap_utils.py
# SHAP computation and visualisation helpers
# Intelligent Adaptive Credit Risk Scoring Model
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# -----------------------------------------------------------------------------
# EXPLAINER FACTORY
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_explainer(_model, _X_train: pd.DataFrame, model_name: str):
    """
    Build and cache a SHAP explainer appropriate to the model type.
    Uses TreeExplainer for tree-based models, LinearExplainer for LR.
    Underscore prefixes prevent Streamlit from hashing the model objects.
    """
    if not SHAP_AVAILABLE:
        return None

    try:
        if model_name in ("XGBClassifier", "RandomForestClassifier",
                          "DecisionTreeClassifier"):
            explainer = shap.TreeExplainer(_model)
        elif model_name == "LogisticRegression":
            # LinearExplainer needs a background dataset
            background = shap.sample(_X_train, min(100, len(_X_train)), random_state=42)
            explainer  = shap.LinearExplainer(_model, background,
                                              feature_perturbation="interventional")
        else:
            # Fallback: KernelExplainer (slower)
            background = shap.sample(_X_train, 50, random_state=42)
            explainer  = shap.KernelExplainer(_model.predict_proba, background)
        return explainer
    except Exception:
        return None


# -----------------------------------------------------------------------------
# SHAP VALUES
# -----------------------------------------------------------------------------

def compute_shap_values(_explainer, X: pd.DataFrame, model_name: str):
    """
    Compute SHAP values for a DataFrame.
    Returns shap_values array (for class 1 — default).
    """
    if _explainer is None:
        return None
    try:
        sv = _explainer.shap_values(X)
        # Tree models return list [class0, class1]; take class 1
        if isinstance(sv, list) and len(sv) == 2:
            return sv[1]
        # Some XGB versions return 2D directly
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            return sv[:, :, 1]
        return sv
    except Exception:
        return None


# -----------------------------------------------------------------------------
# WATERFALL PLOT — single prediction
# -----------------------------------------------------------------------------

def waterfall_plot(shap_vals: np.ndarray,
                   feature_names: list,
                   base_value: float,
                   predicted_prob: float,
                   top_n: int = 12) -> plt.Figure:
    """
    Custom waterfall chart showing feature contributions for one prediction.
    """
    # Pair features with their SHAP values
    pairs = sorted(zip(feature_names, shap_vals),
                   key=lambda x: abs(x[1]), reverse=True)
    pairs = pairs[:top_n]
    feats, vals = zip(*pairs)
    feats = [f[:32] for f in feats]  # truncate long names

    colours = ["#D62728" if v > 0 else "#2CA02C" for v in vals]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))
    y_pos   = np.arange(len(feats))

    ax.barh(y_pos, vals, color=colours, alpha=0.85,
            edgecolor="white", linewidth=0.5)

    for i, (v, feat) in enumerate(zip(vals, feats)):
        ha    = "left" if v >= 0 else "right"
        xpos  = v + (0.003 if v >= 0 else -0.003)
        ax.text(xpos, i, f"{v:+.4f}", va="center", ha=ha,
                fontsize=8, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP Value (impact on log-odds of default)", fontsize=10)
    ax.set_title(
        f"SHAP Feature Contributions — Predicted Default Probability: "
        f"{predicted_prob:.3f}\n(Base value: {base_value:.3f})",
        fontsize=11, fontweight="bold", pad=10
    )

    pos_patch = mpatches.Patch(color="#D62728", alpha=0.85,
                                label="Increases default risk")
    neg_patch = mpatches.Patch(color="#2CA02C", alpha=0.85,
                                label="Decreases default risk")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9,
              loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SUMMARY / BEE SWARM PLOT — dataset-level
# -----------------------------------------------------------------------------

def summary_plot(shap_values: np.ndarray,
                 X: pd.DataFrame,
                 max_display: int = 20) -> plt.Figure:
    """
    SHAP summary dot plot (beeswarm) for dataset-level feature importance.
    """
    if not SHAP_AVAILABLE:
        return _fallback_importance_bar(shap_values, X.columns.tolist())

    fig, ax = plt.subplots(figsize=(10, max(5, min(max_display, 20) * 0.4)))
    shap.summary_plot(
        shap_values, X,
        max_display=max_display,
        show=False,
        plot_size=None,
        color_bar_label="Feature value"
    )
    plt.title("SHAP Summary — Feature Impact on Default Probability",
              fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    return plt.gcf()


def mean_abs_shap_bar(shap_values: np.ndarray,
                      feature_names: list,
                      top_n: int = 15) -> plt.Figure:
    """
    Horizontal bar chart of mean |SHAP| per feature.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    pairs    = sorted(zip(feature_names, mean_abs),
                      key=lambda x: x[1])[-top_n:]
    feats, vals = zip(*pairs)
    feats = [f[:35] for f in feats]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.38)))
    colours = plt.cm.Blues(np.linspace(0.3, 0.85, len(feats)))
    ax.barh(feats, vals, color=colours, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("Global Feature Importance (Mean Absolute SHAP)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# FALLBACK — if SHAP not installed
# -----------------------------------------------------------------------------

def _fallback_importance_bar(shap_values, feature_names) -> plt.Figure:
    mean_abs = np.abs(shap_values).mean(axis=0)
    return mean_abs_shap_bar(shap_values, feature_names)
