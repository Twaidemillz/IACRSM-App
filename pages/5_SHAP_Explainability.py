# =============================================================================
# pages/5_SHAP_Explainability.py
# Global and individual-level SHAP explainability
# =============================================================================

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="SHAP Explainability · IACRSM",
                   page_icon="🔍", layout="wide")

from utils.styling    import (inject_global_css, page_header, section_header,
                               banner, divider, COLOURS, MODEL_DISPLAY_NAMES)
from utils.predictor  import load_artefacts
from utils.shap_utils import (get_explainer, compute_shap_values,
                               waterfall_plot, mean_abs_shap_bar,
                               SHAP_AVAILABLE)

inject_global_css()

art = load_artefacts()
page_header("🔍 SHAP Explainability",
            "Global feature importance and individual-level prediction explanations.")
divider()

if not art.get("loaded"):
    banner(f"⚠️  Artefacts not loaded: {art.get('error','')}", kind="warn")
    st.stop()

if not SHAP_AVAILABLE:
    banner(
        "⚠️  <code>shap</code> package not installed. "
        "Run <code>pip install shap</code> and restart the app.",
        kind="warn"
    )
    st.stop()

X_train       = art["X_train"]
X_test        = art["X_test"]
y_test        = art["y_test"]
feature_names = art["feature_names"]
models        = art.get("models", {})

model_options   = list(models.keys())
display_options = [MODEL_DISPLAY_NAMES.get(m, m) for m in model_options]

# ── Model selector ─────────────────────────────────────────────────────────────
col_sel, col_info = st.columns([2, 3])
with col_sel:
    selected_display = st.selectbox("Select Classifier", display_options)
    selected_key     = model_options[display_options.index(selected_display)]
    active_model     = models[selected_key]
with col_info:
    st.markdown(f"""
    <div class="banner-info" style="margin-top:1.8rem;">
    <strong>Explainer:</strong>
    {'TreeExplainer' if selected_key in ('XGBClassifier','RandomForestClassifier','DecisionTreeClassifier') else 'LinearExplainer'}
    — SHAP values represent the additive contribution of each feature
    to the predicted log-odds of default relative to the base rate.
    GDPR Article 22 aligned.
    </div>
    """, unsafe_allow_html=True)

# ── Build explainer ────────────────────────────────────────────────────────────
with st.spinner(f"Building SHAP explainer for {selected_display}…"):
    explainer = get_explainer(active_model, X_train, selected_key)

if explainer is None:
    banner("❌  Could not build SHAP explainer for this model.", kind="danger")
    st.stop()

divider()

tab_global, tab_individual = st.tabs(
    ["🌐 Global Importance", "🔎 Individual Prediction"]
)

# =============================================================================
# TAB 1: GLOBAL
# =============================================================================
with tab_global:
    section_header("Global Feature Importance (Test Set)")

    sample_size = st.slider(
        "Number of test samples to explain (larger = slower)",
        min_value=20, max_value=min(200, len(X_test)),
        value=min(100, len(X_test)), step=10
    )

    with st.spinner(f"Computing SHAP values for {sample_size} samples…"):
        X_sample   = X_test.sample(n=sample_size, random_state=42).reset_index(drop=True)
        shap_vals  = compute_shap_values(explainer, X_sample, selected_key)

    if shap_vals is None:
        banner("❌  SHAP computation failed.", kind="danger")
    else:
        top_n = st.slider("Features to display", min_value=5, max_value=30,
                          value=15, step=1)

        subtab_bar, subtab_bee = st.tabs(["Mean |SHAP| Bar", "Beeswarm Summary"])

        with subtab_bar:
            fig = mean_abs_shap_bar(shap_vals, feature_names, top_n=top_n)
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown(f"""
            <div style="font-size:0.82rem;color:{COLOURS['text_muted']};margin-top:0.5rem;">
            Mean absolute SHAP values across {sample_size} test samples.
            Longer bars = greater average impact on the default prediction.
            </div>
            """, unsafe_allow_html=True)

        with subtab_bee:
            try:
                import shap
                fig2, ax2 = plt.subplots(figsize=(10, max(5, top_n * 0.38)))
                shap.summary_plot(
                    shap_vals, X_sample,
                    max_display=top_n,
                    show=False,
                    plot_size=None
                )
                plt.title(
                    f"SHAP Summary — {selected_display} · {sample_size} samples",
                    fontsize=12, fontweight="bold"
                )
                plt.tight_layout()
                st.pyplot(plt.gcf(), use_container_width=True)
                plt.close("all")
                st.markdown(f"""
                <div style="font-size:0.82rem;color:{COLOURS['text_muted']};margin-top:0.5rem;">
                Each dot = one observation. Colour = feature value (red=high, blue=low).
                Position on x-axis = SHAP value (impact on model output).
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                banner(f"Beeswarm plot unavailable: {e}", kind="warn")

        # Feature importance table
        with st.expander("View numeric feature importance table"):
            mean_abs = np.abs(shap_vals).mean(axis=0)
            imp_df = pd.DataFrame({
                "Feature":        feature_names,
                "Mean |SHAP|":    np.round(mean_abs, 5),
                "Max |SHAP|":     np.round(np.abs(shap_vals).max(axis=0), 5),
                "Std SHAP":       np.round(shap_vals.std(axis=0), 5),
            }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
            st.dataframe(imp_df, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 2: INDIVIDUAL
# =============================================================================
with tab_individual:
    section_header("Individual Prediction Explanation")

    banner(
        "Select an applicant from the test set by index, or paste a row index "
        "to explain why the model produced that specific prediction.",
        kind="info"
    )

    applicant_idx = st.number_input(
        "Test set applicant index (0 – " + str(len(X_test) - 1) + ")",
        min_value=0, max_value=len(X_test) - 1, value=0, step=1
    )

    X_one     = X_test.iloc[[applicant_idx]]
    y_true    = int(y_test.iloc[applicant_idx])
    prob      = float(active_model.predict_proba(X_one)[0, 1])
    pred      = int(active_model.predict(X_one)[0])

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="label">True Label</div>'
            f'<div class="value" style="color:{"#D62728" if y_true==1 else "#2CA02C"}">'
            f'{"Default" if y_true==1 else "No Default"}</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="label">Predicted Class</div>'
            f'<div class="value" style="color:{"#D62728" if pred==1 else "#2CA02C"}">'
            f'{"Default" if pred==1 else "No Default"}</div></div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="label">Default Probability</div>'
            f'<div class="value">{prob:.4f}</div>'
            f'<div class="sub">{"✅ Correct" if pred==y_true else "❌ Incorrect"}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Compute individual SHAP
    with st.spinner("Computing individual SHAP values…"):
        shap_one = compute_shap_values(explainer, X_one, selected_key)

    if shap_one is None:
        banner("SHAP computation failed for this instance.", kind="danger")
    else:
        try:
            base_val = float(explainer.expected_value)
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[1]
        except Exception:
            base_val = 0.0

        top_n_ind = st.slider("Features to show in waterfall", 5, 20, 12, 1,
                              key="ind_top_n")
        fig = waterfall_plot(
            shap_one[0], feature_names, base_val, prob, top_n=top_n_ind
        )
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown(f"""
        <div style="font-size:0.82rem;color:{COLOURS['text_muted']};margin-top:0.5rem;">
        <strong>Red bars</strong> push the score toward default (increase risk).
        <strong>Green bars</strong> push toward no default (reduce risk).
        Base value ({base_val:.3f}) = model output with no features.
        Final prediction = base + sum of all SHAP values ≈ {prob:.3f}.
        </div>
        """, unsafe_allow_html=True)

        # Top contributing features table
        with st.expander("View all feature contributions for this applicant"):
            feature_vals = X_one.iloc[0].to_dict()
            contrib_df = pd.DataFrame({
                "Feature":       feature_names,
                "Feature Value": [round(float(feature_vals[f]), 4)
                                  if f in feature_vals else "—"
                                  for f in feature_names],
                "SHAP Value":    np.round(shap_one[0], 5),
                "Direction":     ["↑ Risk" if v > 0 else "↓ Risk"
                                  for v in shap_one[0]],
            }).sort_values("SHAP Value", key=abs, ascending=False).reset_index(drop=True)
            st.dataframe(contrib_df, use_container_width=True, hide_index=True)

divider()

banner(
    "ℹ️  SHAP (SHapley Additive exPlanations) values are computed using "
    "game-theoretic Shapley values, satisfying efficiency, symmetry, dummy, "
    "and additivity axioms. This approach satisfies GDPR Article 22 "
    "right-to-explanation requirements for automated credit decisions.",
    kind="info"
)
