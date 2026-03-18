# =============================================================================
# pages/4_Model_Comparison.py
# Side-by-side comparison of all four classifiers across six metrics
# =============================================================================

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                             f1_score, accuracy_score, matthews_corrcoef,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay)

st.set_page_config(page_title="Model Comparison · IACRSM",
                   page_icon="📊", layout="wide")

from utils.styling   import (inject_global_css, page_header, section_header,
                              banner, divider, metric_card, COLOURS,
                              METRIC_DESCRIPTIONS, MODEL_DISPLAY_NAMES)
from utils.predictor import load_artefacts

inject_global_css()

art = load_artefacts()
page_header("📊 Model Comparison",
            "Six-metric evaluation of all classifiers on the held-out test set.")
divider()

if not art.get("loaded"):
    banner(f"⚠️  Artefacts not loaded: {art.get('error','')}", kind="warn")
    st.stop()

X_test  = art["X_test"]
y_test  = art["y_test"]
models  = art.get("models", {})

if not models:
    banner("No individual model files found. Ensure artefacts/models/ is present.", kind="warn")
    st.stop()

METRICS = ["ROC-AUC", "Recall", "Precision", "F1", "Accuracy", "MCC"]

# ── Compute metrics for all models ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_all_metrics(_models_dict, _X_test, _y_test):
    rows = []
    roc_data = {}
    for name, model in _models_dict.items():
        y_prob = model.predict_proba(_X_test)[:, 1]
        y_pred = model.predict(_X_test)
        display = MODEL_DISPLAY_NAMES.get(name, name)
        row = {
            "Model":     display,
            "ROC-AUC":   round(roc_auc_score(_y_test, y_prob),        4),
            "Recall":    round(recall_score(_y_test, y_pred),          4),
            "Precision": round(precision_score(_y_test, y_pred,
                               zero_division=0),                       4),
            "F1":        round(f1_score(_y_test, y_pred),              4),
            "Accuracy":  round(accuracy_score(_y_test, y_pred),        4),
            "MCC":       round(matthews_corrcoef(_y_test, y_pred),      4),
        }
        rows.append(row)
        fpr, tpr, _ = roc_curve(_y_test, y_prob)
        roc_data[display] = (fpr, tpr, row["ROC-AUC"])
    return pd.DataFrame(rows), roc_data

with st.spinner("Computing metrics…"):
    metrics_df, roc_data = compute_all_metrics(models, X_test, y_test)

best_model_display = MODEL_DISPLAY_NAMES.get(art["best_model_name"], art["best_model_name"])

# ── Summary metric cards for best model ───────────────────────────────────────
section_header(f"Best Model: {best_model_display}")
best_row = metrics_df[metrics_df["Model"] == best_model_display]

if not best_row.empty:
    best_row = best_row.iloc[0]
    cols = st.columns(6)
    for col, m in zip(cols, METRICS):
        with col:
            st.markdown(
                metric_card(m, str(best_row[m]),
                            "Primary" if m in ("ROC-AUC","Recall") else ""),
                unsafe_allow_html=True
            )
    st.markdown("<br>", unsafe_allow_html=True)

divider()

# ── Full metrics table ─────────────────────────────────────────────────────────
section_header("Full Metrics Table — All Classifiers")

# Highlight best value per metric
def highlight_best(df):
    styled = pd.DataFrame("", index=df.index, columns=df.columns)
    for col in METRICS:
        if col in df.columns:
            max_idx = df[col].idxmax()
            styled.loc[max_idx, col] = (
                f"background-color:#D4EDDA; font-weight:bold; color:#155724"
            )
    # Highlight best model row
    for idx, row in df.iterrows():
        if row["Model"] == best_model_display:
            styled.loc[idx, "Model"] = (
                f"background-color:#E8F4F8; font-weight:bold"
            )
    return styled

st.dataframe(
    metrics_df.style.apply(highlight_best, axis=None),
    use_container_width=True, hide_index=True
)

st.markdown(
    "<div style='font-size:0.78rem;color:#6C757D;margin-top:0.3rem;'>"
    "Green cells = best value per metric. Highlighted row = selected best model from Phase 2."
    "</div>",
    unsafe_allow_html=True
)

divider()

# ── Grouped bar chart ──────────────────────────────────────────────────────────
section_header("Metric Comparison — Grouped Bar Chart")

tab_bar, tab_roc, tab_cm = st.tabs(["Bar Chart", "ROC Curves", "Confusion Matrices"])

with tab_bar:
    selected_metrics = st.multiselect(
        "Select metrics to display",
        options=METRICS, default=["ROC-AUC", "Recall", "F1", "MCC"]
    )

    if selected_metrics:
        n_metrics = len(selected_metrics)
        n_models  = len(metrics_df)
        x         = np.arange(n_metrics)
        width     = 0.8 / n_models

        palette = [COLOURS["primary"], COLOURS["secondary"],
                   COLOURS["accent"], COLOURS["high_risk"]]

        fig, ax = plt.subplots(figsize=(max(9, n_metrics * 2.2), 5))
        for i, (_, row) in enumerate(metrics_df.iterrows()):
            offset = (i - n_models / 2 + 0.5) * width
            vals   = [row[m] for m in selected_metrics]
            bars   = ax.bar(x + offset, vals, width * 0.92,
                            label=row["Model"],
                            color=palette[i % len(palette)],
                            alpha=0.85, edgecolor="white", linewidth=0.4)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.007,
                        f"{bar.get_height():.3f}",
                        ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(selected_metrics, fontsize=11)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Classifier Performance — Test Set",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10, loc="upper right")
        ax.axhline(0.75, color="gray", lw=0.8, linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown(
            "<div class='banner-info'>Dashed line at 0.75 = ROC-AUC target threshold.</div>",
            unsafe_allow_html=True
        )

with tab_roc:
    section_header("ROC Curves")
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = [COLOURS["primary"], COLOURS["secondary"],
               COLOURS["accent"], COLOURS["high_risk"]]
    for i, (display, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, lw=2, color=palette[i % len(palette)],
                label=f"{display}  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — All Classifiers", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with tab_cm:
    section_header("Confusion Matrices")
    cm_cols = st.columns(min(4, len(models)))
    for col, (name, model) in zip(cm_cols, models.items()):
        y_pred  = model.predict(X_test)
        cm      = confusion_matrix(y_test, y_pred)
        display = MODEL_DISPLAY_NAMES.get(name, name)
        with col:
            fig, ax = plt.subplots(figsize=(4, 3.5))
            disp = ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(display, fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

divider()

# ── Metric descriptions ────────────────────────────────────────────────────────
section_header("Metric Definitions")
for m, desc in METRIC_DESCRIPTIONS.items():
    st.markdown(f"""
    <div style="display:flex; gap:1rem; padding:0.45rem 0.6rem;
                border-bottom:1px solid {COLOURS['border']};">
        <div style="min-width:100px; font-weight:700;
                    color:{COLOURS['primary']}; font-size:0.87rem;">{m}</div>
        <div style="font-size:0.85rem; color:#444;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)
