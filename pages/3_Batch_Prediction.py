# =============================================================================
# pages/3_Batch_Prediction.py
# Batch CSV upload → scored output with risk bands and download
# =============================================================================

import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Batch Prediction · IACRSM",
                   page_icon="📋", layout="wide")

from utils.styling   import (inject_global_css, page_header, section_header,
                              banner, divider, metric_card, COLOURS)
from utils.predictor import (load_artefacts, predict_batch,
                              validate_batch_csv, batch_csv_template,
                              MODEL_DISPLAY_NAMES)

inject_global_css()

art = load_artefacts()
page_header("📋 Batch Prediction",
            "Upload a CSV of applicants — download scored results with risk bands.")
divider()

if not art.get("loaded"):
    banner(f"⚠️  Artefacts not loaded: {art.get('error','')}", kind="warn")
    st.stop()

feature_names     = art["feature_names"]
available_models  = art.get("models", {})
model_options     = list(available_models.keys())
display_options   = [MODEL_DISPLAY_NAMES.get(m, m) for m in model_options]

# ── Controls ───────────────────────────────────────────────────────────────────
ctrl_col, info_col = st.columns([2, 3])
with ctrl_col:
    selected_display = st.selectbox("Classifier", display_options)
    selected_key     = model_options[display_options.index(selected_display)]
    active_model     = available_models[selected_key]

with info_col:
    st.markdown("""
    <div class="banner-info" style="margin-top:1.8rem;">
    Upload a CSV whose columns match the model's feature schema.
    Download a blank template below to see the expected headers.
    </div>
    """, unsafe_allow_html=True)

# ── Template download ──────────────────────────────────────────────────────────
section_header("CSV Template")
st.markdown(
    "Download the blank template, fill in applicant data (one row per applicant), "
    "then upload below.",
    unsafe_allow_html=False
)
template_bytes = batch_csv_template(feature_names)
st.download_button(
    "⬇  Download CSV Template",
    data      = template_bytes,
    file_name = "credit_risk_batch_template.csv",
    mime      = "text/csv",
)

divider()

# ── File upload ────────────────────────────────────────────────────────────────
section_header("Upload Applicant CSV")
uploaded = st.file_uploader("Upload applicant CSV", type=["csv"],
                             label_visibility="collapsed")

# ── Demo mode: use X_test ─────────────────────────────────────────────────────
use_demo = st.checkbox(
    "Use test set (X_test) as demo — scores the 200 held-out applicants",
    value=False
)

if use_demo:
    batch_df = art["X_test"].copy()
    st.info(f"Demo mode: {len(batch_df)} applicants from X_test loaded.")
elif uploaded is not None:
    try:
        batch_df = pd.read_csv(uploaded)
        valid, msg = validate_batch_csv(batch_df, feature_names)
        if not valid:
            banner(f"❌  {msg}", kind="danger")
            st.stop()
        st.success(f"File loaded: {len(batch_df):,} applicants · {batch_df.shape[1]} columns")
    except Exception as e:
        banner(f"❌  Failed to read CSV: {e}", kind="danger")
        st.stop()
else:
    banner(
        "📤  Upload a CSV file above, or enable demo mode to score the test set.",
        kind="info"
    )
    st.stop()

# ── Run batch scoring ──────────────────────────────────────────────────────────
with st.spinner("Scoring applicants…"):
    results_df = predict_batch(active_model, batch_df[feature_names])

divider()
section_header("Batch Results Summary")

total      = len(results_df)
n_low      = (results_df["risk_band"] == "Low").sum()
n_moderate = (results_df["risk_band"] == "Moderate").sum()
n_high     = (results_df["risk_band"] == "High").sum()
avg_prob   = results_df["prob_default"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(metric_card("Total Applicants", f"{total:,}", "scored"), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Low Risk", f"{n_low:,}", f"{n_low/total:.1%}"), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Moderate Risk", f"{n_moderate:,}", f"{n_moderate/total:.1%}"), unsafe_allow_html=True)
with c4:
    st.markdown(metric_card("High Risk", f"{n_high:,}", f"{n_high/total:.1%}"), unsafe_allow_html=True)
with c5:
    st.markdown(metric_card("Avg Default Prob", f"{avg_prob:.3f}", "mean across batch"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Visualisations ─────────────────────────────────────────────────────────────
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    section_header("Risk Band Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    band_counts  = [n_low, n_moderate, n_high]
    band_labels  = ["Low", "Moderate", "High"]
    band_colours = [COLOURS["low_risk"], COLOURS["moderate_risk"], COLOURS["high_risk"]]
    wedges, texts, autotexts = ax.pie(
        band_counts, labels=band_labels, colors=band_colours,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Applicants by Risk Band", fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with viz_col2:
    section_header("Default Probability Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(results_df["prob_default"], bins=30, color=COLOURS["secondary"],
            alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(0.35, color=COLOURS["moderate_risk"],
               linestyle="--", lw=1.5, label="Low/Moderate (0.35)")
    ax.axvline(0.60, color=COLOURS["high_risk"],
               linestyle="--", lw=1.5, label="Moderate/High (0.60)")
    ax.set_xlabel("Predicted Default Probability", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Distribution of Default Scores", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

divider()

# ── Results table ──────────────────────────────────────────────────────────────
section_header("Scored Results Table")

filter_band = st.multiselect(
    "Filter by risk band",
    options=["Low", "Moderate", "High"],
    default=["Low", "Moderate", "High"]
)

filtered_df = results_df[results_df["risk_band"].isin(filter_band)].copy()

# Colour-highlight risk bands in output
def highlight_risk(row):
    c_map = {"Low": "background-color:#D4EDDA",
             "Moderate": "background-color:#FFF3CD",
             "High": "background-color:#F8D7DA"}
    return [c_map.get(row["risk_band"], "")] * len(row)

display_cols = ["prob_default", "predicted_class", "risk_band"]
st.dataframe(
    filtered_df[display_cols].style.apply(highlight_risk, axis=1),
    use_container_width=True,
    height=400
)

st.markdown(f"Showing **{len(filtered_df):,}** of **{total:,}** applicants.")

# ── Download scored CSV ────────────────────────────────────────────────────────
divider()
output_buf = io.BytesIO()
results_df.to_csv(output_buf, index=False)

st.download_button(
    "⬇  Download Full Scored Results (CSV)",
    data      = output_buf.getvalue(),
    file_name = f"scored_applicants_{selected_key.lower()}.csv",
    mime      = "text/csv",
)
