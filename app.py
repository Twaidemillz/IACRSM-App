# =============================================================================
# app.py  — Entry point
# Intelligent Adaptive Credit Risk Scoring Model
# Streamlit Multi-Page Application
# =============================================================================
# Launch: streamlit run app.py
# =============================================================================

import streamlit as st

st.set_page_config(
    page_title  = "Credit Risk Scoring — IACRSM",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

from utils.styling import inject_global_css, page_header, banner, divider, COLOURS

inject_global_css()

# ── Sidebar brand ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <div style="font-size:2rem;">📊</div>
        <div style="font-size:1rem; font-weight:700; color:#FFFFFF;
                    letter-spacing:0.04em; line-height:1.3;">
            Credit Risk<br>Scoring Model
        </div>
        <div style="font-size:0.72rem; color:#AECBFA; margin-top:0.3rem;">
            IACRSM · German Credit Dataset
        </div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.15); margin:0.8rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.78rem; color:#AECBFA; padding:0 0.5rem;">
        <b style="color:#fff;">Navigation</b><br>
        Use the pages listed above to explore the app.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#AECBFA; padding:0 0.5rem;">
        <b style="color:#fff;">Quick Reference</b>
        <ul style="margin:0.4rem 0 0 1rem; padding:0; color:#AECBFA;">
            <li>ROC-AUC target ≥ 0.75</li>
            <li>Recall (default) co-primary</li>
            <li>PSI &lt; 0.10 → Stable</li>
            <li>PSI 0.10–0.25 → Moderate</li>
            <li>PSI &gt; 0.25 → Retrain</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <hr style="border-color:rgba(255,255,255,0.15);">
    <div style="font-size:0.68rem; color:#7B9FC4; text-align:center;">
        Dissertation Project · 2024<br>
        Logistic Regression · Decision Tree<br>
        Random Forest · XGBoost · SHAP
    </div>
    """, unsafe_allow_html=True)

# ── Home page content ─────────────────────────────────────────────────────────
page_header(
    "Intelligent Adaptive Credit Risk Scoring Model",
    "Machine Learning · SHAP Explainability · PSI-Based Adaptive Retraining",
    "RUFAI OLUWATOBI DAMILOLA · SCI/22/23/1044"
)

divider()

st.markdown("""
<div style="font-size:0.95rem; line-height:1.75; color:#212529; max-width:820px;">
This application is the interactive component of an undergraduate final year project
investigating an <strong>Intelligent Adaptive Credit Risk Scoring Model (IACRSM)</strong>
built on the German Credit Dataset. The system integrates four machine learning
classifiers, SHAP-based individual explainability, and a PSI-driven adaptive
retraining feedback mechanism aligned with GDPR Article 22 requirements.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Pipeline overview cards
col1, col2, col3, col4, col5 = st.columns(5)

pipeline_stages = [
    ("🔄", "Phase 1",  "Data Preparation",    "German Credit · Encoding · Class weights"),
    ("🤖", "Phase 2",  "Model Training",       "LR · DT · RF · XGBoost · RandomizedSearchCV"),
    ("📈", "Phase 3",  "Evaluation",           "ROC-AUC · Recall · F1 · MCC · Precision"),
    ("🔍", "Phase 4",  "SHAP Explainability",  "TreeExplainer · Waterfall · Summary plots"),
    ("🔁", "Phase 5",  "Adaptive Retraining",  "PSI monitoring · Drift detection · Versioning"),
]

for col, (icon, phase, title, desc) in zip(
        [col1, col2, col3, col4, col5], pipeline_stages):
    with col:
        st.markdown(f"""
        <div style="background:{COLOURS['bg_card']}; border:1px solid {COLOURS['border']};
                    border-radius:10px; padding:1rem 0.8rem; text-align:center;
                    box-shadow:0 2px 6px rgba(0,0,0,0.06); min-height:160px;">
            <div style="font-size:1.6rem;">{icon}</div>
            <div style="font-size:0.68rem; font-weight:700; color:{COLOURS['text_muted']};
                        text-transform:uppercase; letter-spacing:0.06em;
                        margin:0.3rem 0 0.2rem;">{phase}</div>
            <div style="font-size:0.85rem; font-weight:700;
                        color:{COLOURS['primary']}; margin-bottom:0.4rem;">{title}</div>
            <div style="font-size:0.75rem; color:{COLOURS['text_muted']};
                        line-height:1.4;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

divider()

# App pages directory
st.markdown(f'<div class="section-header">Application Pages</div>',
            unsafe_allow_html=True)

pages = [
    ("🏠", "1 — Home",              "This page. Project overview and pipeline summary."),
    ("🔮", "2 — Single Prediction", "Score an individual loan applicant with a live input form."),
    ("📋", "3 — Batch Prediction",  "Upload a CSV of applicants and download scored results."),
    ("📊", "4 — Model Comparison",  "Compare all four classifiers across six evaluation metrics."),
    ("🔍", "5 — SHAP Explainability","Global and individual-level SHAP feature importance."),
    ("📡", "6 — PSI Monitor",       "PSI drift detection dashboard with retraining trigger logic."),
]

for icon, name, desc in pages:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.8rem;
                padding:0.6rem 0.8rem; border-radius:8px;
                border:1px solid {COLOURS['border']};
                background:{COLOURS['bg_card']}; margin-bottom:0.5rem;">
        <div style="font-size:1.3rem; width:2rem; text-align:center;">{icon}</div>
        <div>
            <div style="font-weight:700; color:{COLOURS['primary']};
                        font-size:0.9rem;">{name}</div>
            <div style="font-size:0.82rem; color:{COLOURS['text_muted']};">{desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

divider()

# Setup notice
banner(
    "⚙️  <strong>Setup:</strong> Place your <code>artefacts/</code> folder "
    "(output of the five-phase pipeline) in the same directory as "
    "<code>app.py</code> before running the app. All pages load models "
    "and data from <code>artefacts/</code> automatically.",
    kind="info"
)

st.markdown(f"""
<div style="font-size:0.8rem; color:{COLOURS['text_muted']}; margin-top:1rem;">
    <strong>Required artefacts:</strong>
    <code>artefacts/X_train.csv</code> · <code>X_test.csv</code> ·
    <code>y_train.csv</code> · <code>y_test.csv</code> ·
    <code>best_model.pkl</code> · <code>best_model_name.pkl</code> ·
    <code>class_weights.pkl</code> ·
    <code>artefacts/models/</code> · <code>artefacts/psi/</code> ·
    <code>artefacts/versions/</code>
</div>
""", unsafe_allow_html=True)
