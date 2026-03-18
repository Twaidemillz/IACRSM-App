# =============================================================================
# pages/2_Single_Prediction.py
# Live single-applicant credit risk scoring
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Single Prediction · IACRSM",
                   page_icon="🔮", layout="wide")

from utils.styling   import (inject_global_css, page_header, section_header,
                              banner, divider, metric_card, risk_badge,
                              get_risk_level, COLOURS)
from utils.predictor import (load_artefacts, predict_single, build_input_df,
                              NUMERIC_FEATURES, CATEGORICAL_FEATURES,
                              MODEL_DISPLAY_NAMES)

inject_global_css()

art = load_artefacts()
page_header("🔮 Single Applicant Prediction",
            "Enter applicant details to generate a live credit risk score.")
divider()

if not art.get("loaded"):
    banner(f"Artefacts not loaded: {art.get('error','')}", kind="warn")
    st.stop()

scaler         = art["scaler"]
label_encoders = art["label_encoders"]
feature_names  = art["feature_names"]

available_models = art.get("models", {})
model_options    = list(available_models.keys())
display_options  = [MODEL_DISPLAY_NAMES.get(m, m) for m in model_options]

col_sel, col_info = st.columns([2, 3])
with col_sel:
    selected_display = st.selectbox("Select Classifier", display_options)
    selected_key     = model_options[display_options.index(selected_display)]
    active_model     = available_models[selected_key]
with col_info:
    st.markdown(f"""
    <div class="banner-info" style="margin-top:1.8rem;">
    Using <strong>{selected_display}</strong>.
    Best model from Phase 2: <strong>{art['best_model_name']}</strong>.
    </div>""", unsafe_allow_html=True)

divider()
section_header("Applicant Details")

form_values = {}

with st.form("prediction_form"):
    st.markdown("**Financial & Personal Attributes**")
    num_cols = st.columns(4)
    for i, (feat, meta) in enumerate(NUMERIC_FEATURES.items()):
        with num_cols[i % 4]:
            form_values[feat] = st.number_input(
                meta["label"], min_value=meta["min"], max_value=meta["max"],
                value=meta["default"], step=meta["step"], key=f"num_{feat}"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Account & Employment Status**")
    cat_cols = st.columns(4)
    for i, (feat, meta) in enumerate(CATEGORICAL_FEATURES.items()):
        with cat_cols[i % 4]:
            selected_opt = st.selectbox(
                meta["label"], options=meta["display"], key=f"cat_{feat}"
            )
            idx = meta["display"].index(selected_opt)
            form_values[feat] = meta["options"][idx]

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Score Applicant", use_container_width=True)

if submitted:
    divider()
    section_header("Prediction Result")

    input_df = build_input_df(form_values, feature_names, label_encoders, scaler)
    prob, pred, probs_all = predict_single(active_model, input_df)
    risk_label, risk_level = get_risk_level(prob)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Default Probability", f"{prob:.3f}",
                                "0 = no risk · 1 = certain default"), unsafe_allow_html=True)
    with c2:
        pred_label = "Default" if pred == 1 else "No Default"
        st.markdown(metric_card("Predicted Class", pred_label, ""), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Confidence (No Default)", f"{probs_all[0]:.3f}",
                                "P(class = 0)"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Confidence (Default)", f"{probs_all[1]:.3f}",
                                "P(class = 1)"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    colour_map = {"low": COLOURS["low_risk"],
                  "moderate": COLOURS["moderate_risk"],
                  "high": COLOURS["high_risk"]}
    bg_map = {"low": "#D4EDDA", "moderate": "#FFF3CD", "high": "#F8D7DA"}

    st.markdown(f"""
    <div style="background:{bg_map[risk_level]}; border-radius:12px;
                padding:1.2rem 1.6rem; display:flex; align-items:center; gap:1.2rem;">
        <div style="font-size:2.5rem;">
            {'✅' if risk_level=='low' else '⚠️' if risk_level=='moderate' else '🚨'}
        </div>
        <div>
            <div style="font-size:1.05rem; font-weight:700; color:{colour_map[risk_level]};">
                {risk_label}
            </div>
            <div style="font-size:0.88rem; color:#555; margin-top:0.2rem;">
                {'Default probability below 35% — applicant presents low credit risk.'
                 if risk_level=='low' else
                 'Default probability 35-60% — applicant warrants additional review.'
                 if risk_level=='moderate' else
                 'Default probability above 60% — high likelihood of default. Caution advised.'}
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    section_header("Probability Gauge")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 1.3))
    ax.barh(0, 0.35, color=COLOURS["low_risk"],      height=0.55, alpha=0.25)
    ax.barh(0, 0.25, color=COLOURS["moderate_risk"],  height=0.55, alpha=0.25, left=0.35)
    ax.barh(0, 0.40, color=COLOURS["high_risk"],      height=0.55, alpha=0.25, left=0.60)
    ax.barh(0, prob,  color=colour_map[risk_level],   height=0.55, alpha=0.9)
    ax.axvline(prob, color="black", lw=2)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Default Probability", fontsize=10)
    ax.text(0.175, 0.75, "Low",      ha="center", fontsize=9,
            color=COLOURS["low_risk"],      fontweight="bold",
            transform=ax.get_xaxis_transform())
    ax.text(0.475, 0.75, "Moderate", ha="center", fontsize=9,
            color=COLOURS["moderate_risk"],  fontweight="bold",
            transform=ax.get_xaxis_transform())
    ax.text(0.80,  0.75, "High",     ha="center", fontsize=9,
            color=COLOURS["high_risk"],      fontweight="bold",
            transform=ax.get_xaxis_transform())
    ax.set_title(f"Score: {prob:.4f}", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    divider()
    banner("Navigate to 5 — SHAP Explainability to see which features drove this prediction.",
           kind="info")
