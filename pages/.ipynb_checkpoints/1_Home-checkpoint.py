# =============================================================================
# pages/1_Home.py
# Home — Project overview, pipeline diagram, artefact status check
# =============================================================================

import os
import streamlit as st

st.set_page_config(page_title="Home · IACRSM", page_icon="🏠", layout="wide")

from utils.styling  import inject_global_css, page_header, section_header, \
                           banner, divider, COLOURS, metric_card
from utils.predictor import load_artefacts

inject_global_css()

# ── Load artefacts ─────────────────────────────────────────────────────────────
art = load_artefacts()

page_header("🏠 Home", "Project Overview · Pipeline Summary · Artefact Status")
divider()

# ── Artefact status check ──────────────────────────────────────────────────────
section_header("Artefact Status")

if not art.get("loaded"):
    banner(
        f"⚠️  Artefacts not found. Error: {art.get('error', 'Unknown')}. "
        "Place your <code>artefacts/</code> folder next to <code>app.py</code> "
        "and reload.",
        kind="warn"
    )
else:
    X_train = art["X_train"]
    X_test  = art["X_test"]
    y_train = art["y_train"]
    y_test  = art["y_test"]

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        ("Train samples",   f"{len(X_train):,}",  "observations"),
        ("Test samples",    f"{len(X_test):,}",   "observations"),
        ("Features",        f"{X_train.shape[1]}", "encoded columns"),
        ("Default rate",    f"{y_train.mean():.1%}", "in training set"),
        ("Best model",      art["best_model_name"].replace("Classifier","").replace("Regression","LR"),
                            "from Phase 2"),
    ]
    for col, (lbl, val, sub) in zip([c1, c2, c3, c4, c5], cards):
        with col:
            st.markdown(metric_card(lbl, val, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    banner("✅  All artefacts loaded successfully. Navigate using the sidebar.", kind="info")

divider()

# ── Project abstract ───────────────────────────────────────────────────────────
section_header("Research Context")
st.markdown("""
<div style="font-size:0.93rem; line-height:1.78; color:#333; max-width:860px;">
Credit risk assessment is a cornerstone of financial system stability. In Nigeria,
non-performing loan (NPL) ratios have exceeded <strong>4%</strong> (CBN Financial
Stability Report, 2023), underscoring the need for more accurate, transparent, and
adaptive scoring models. Traditional rule-based scorecards fail to capture complex
non-linear patterns in applicant data and cannot adapt when population characteristics
shift over time — a phenomenon known as <em>concept drift</em>.
<br><br>
This study develops an <strong>Intelligent Adaptive Credit Risk Scoring Model (IACRSM)</strong>
that combines four machine learning classifiers, SHAP-based post-hoc explainability
compliant with GDPR Article 22, and a <strong>Population Stability Index (PSI)</strong>
feedback mechanism that automatically triggers model retraining when distributional
shift is detected (PSI &gt; 0.25).
</div>
""", unsafe_allow_html=True)

divider()

# ── Five-phase pipeline ────────────────────────────────────────────────────────
section_header("Five-Phase ML Pipeline")

phases = [
    ("1", "🔄", "Data Preparation",
     ["German Credit Dataset (1,000 observations, 20 features)",
      "Label encoding + one-hot encoding of categorical features",
      "Stratified 80/20 train-test split",
      "Class-weight computation for imbalanced target (70/30)"]),
    ("2", "🤖", "Model Training",
     ["Logistic Regression, Decision Tree, Random Forest, XGBoost",
      "RandomizedSearchCV with 5-fold StratifiedKFold (50 iterations each)",
      "ROC-AUC as primary optimisation metric",
      "Class weights injected to handle class imbalance"]),
    ("3", "📈", "Model Evaluation",
     ["Six-metric framework: ROC-AUC, Recall, Precision, F1, Accuracy, MCC",
      "ROC-AUC and Recall designated as co-primary metrics",
      "Confusion matrices, ROC curves, and calibration analysis",
      "Best model selected and serialised as best_model.pkl"]),
    ("4", "🔍", "SHAP Explainability",
     ["TreeExplainer for tree-based models; LinearExplainer for LR",
      "Global: mean |SHAP| feature importance + beeswarm summary plot",
      "Individual: waterfall chart per applicant prediction",
      "GDPR Article 22 compliance — right to explanation"]),
    ("5", "🔁", "Adaptive Retraining",
     ["PSI computed on predicted probability distributions",
      "Three drift scenarios: None, Moderate (+0.5σ), Severe (+1.5σ)",
      "Automatic retraining triggered at PSI > 0.25",
      "Semantic model versioning (v1.0.0 → v2.0.0) with audit registry"]),
]

for phase_num, icon, title, bullets in phases:
    with st.expander(f"Phase {phase_num} — {icon} {title}", expanded=False):
        for b in bullets:
            st.markdown(f"- {b}")

divider()

# ── Dataset summary ────────────────────────────────────────────────────────────
section_header("Dataset Summary — German Credit Dataset")
col_a, col_b = st.columns([1, 1])

with col_a:
    st.markdown("""
| Attribute          | Value                        |
|--------------------|------------------------------|
| Source             | UCI ML Repository            |
| Observations       | 1,000                        |
| Features (raw)     | 20                           |
| Target             | Credit risk (Good=0, Bad=1)  |
| Class distribution | 70% Good · 30% Bad           |
| Train split        | 800 observations             |
| Test split         | 200 observations             |
""")

with col_b:
    st.markdown("""
| Classifier              | Hyperparameter Search |
|-------------------------|-----------------------|
| Logistic Regression     | C, solver, penalty    |
| Decision Tree           | depth, min_samples    |
| Random Forest           | n_estimators, depth   |
| XGBoost                 | lr, depth, subsample  |
""")
    st.markdown("""
<div style="font-size:0.82rem; color:#6C757D; margin-top:0.5rem;">
All classifiers optimised via RandomizedSearchCV (50 iterations,
5-fold StratifiedKFold, ROC-AUC scoring).
</div>
""", unsafe_allow_html=True)

divider()

# ── Evaluation framework ───────────────────────────────────────────────────────
section_header("Evaluation Metrics Framework")

metrics = [
    ("ROC-AUC",   "Primary", "≥ 0.75",
     "Discriminative ability across all thresholds. Threshold-invariant ranking metric."),
    ("Recall",    "Co-primary", "Maximise",
     "True positive rate for defaults. Critical in credit: missed defaults are costly."),
    ("Precision", "Secondary", "—",
     "Proportion of predicted defaults that are actual defaults."),
    ("F1 Score",  "Secondary", "—",
     "Harmonic mean of Precision and Recall. Balances both error types."),
    ("Accuracy",  "Tertiary", "—",
     "Overall correct predictions. Less informative under class imbalance."),
    ("MCC",       "Tertiary", "—",
     "Matthews Correlation Coefficient. Robust single-value metric for imbalanced data."),
]

for name, priority, target, desc in metrics:
    p_colour = (COLOURS["primary"] if priority == "Primary"
                else COLOURS["secondary"] if priority == "Co-primary"
                else COLOURS["text_muted"])
    st.markdown(f"""
    <div style="display:flex; gap:1rem; padding:0.55rem 0.8rem;
                border-bottom:1px solid {COLOURS['border']}; align-items:flex-start;">
        <div style="min-width:110px; font-weight:700;
                    color:{COLOURS['primary']}; font-size:0.88rem;">{name}</div>
        <div style="min-width:90px; font-size:0.78rem; color:{p_colour};
                    font-weight:600; padding-top:0.1rem;">{priority}</div>
        <div style="min-width:70px; font-size:0.82rem;
                    color:{COLOURS['text_muted']}; padding-top:0.1rem;">{target}</div>
        <div style="font-size:0.85rem; color:#444; line-height:1.5;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)
