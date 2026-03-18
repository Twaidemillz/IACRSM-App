# =============================================================================
# utils/styling.py
# Shared CSS, colour constants, and UI helpers
# Intelligent Adaptive Credit Risk Scoring Model
# =============================================================================

import streamlit as st

# -----------------------------------------------------------------------------
# COLOUR PALETTE
# -----------------------------------------------------------------------------

COLOURS = {
    # Brand
    "primary":        "#1B3A6B",   # Deep navy
    "secondary":      "#2E86AB",   # Steel blue
    "accent":         "#F4A261",   # Warm amber

    # Risk bands
    "low_risk":       "#2CA02C",   # Green
    "moderate_risk":  "#FF7F0E",   # Orange
    "high_risk":      "#D62728",   # Red

    # PSI bands
    "psi_stable":     "#2CA02C",
    "psi_moderate":   "#FF7F0E",
    "psi_critical":   "#D62728",

    # Neutral
    "bg_card":        "#F8F9FA",
    "border":         "#DEE2E6",
    "text_dark":      "#212529",
    "text_muted":     "#6C757D",
}

# Risk band thresholds
RISK_THRESHOLDS = {
    "low":      (0.00, 0.35),
    "moderate": (0.35, 0.60),
    "high":     (0.60, 1.00),
}

# PSI band thresholds
PSI_THRESHOLDS = {
    "stable":   (0.00, 0.10),
    "moderate": (0.10, 0.25),
    "critical": (0.25, float("inf")),
}

# Model display names
MODEL_DISPLAY_NAMES = {
    "LogisticRegression":    "Logistic Regression",
    "DecisionTreeClassifier":"Decision Tree",
    "RandomForestClassifier":"Random Forest",
    "XGBClassifier":         "XGBoost",
}

# Metric descriptions for tooltips
METRIC_DESCRIPTIONS = {
    "ROC-AUC":   "Area under the ROC curve. Primary ranking metric (target ≥ 0.75).",
    "Recall":    "True positive rate — proportion of actual defaults correctly identified.",
    "Precision": "Of predicted defaults, proportion that were actual defaults.",
    "F1":        "Harmonic mean of Precision and Recall.",
    "Accuracy":  "Overall proportion of correct predictions.",
    "MCC":       "Matthews Correlation Coefficient — robust metric for imbalanced datasets.",
}


# -----------------------------------------------------------------------------
# GLOBAL CSS
# -----------------------------------------------------------------------------

def inject_global_css():
    """Inject shared CSS into every page."""
    st.markdown(f"""
    <style>

    /* ── Typography ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* ── Sidebar ─────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: {COLOURS["primary"]};
    }}
    [data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {{
        color: #FFFFFF !important;
    }}

    /* ── Metric cards ────────────────────────────────────────────── */
    .metric-card {{
        background: {COLOURS["bg_card"]};
        border: 1px solid {COLOURS["border"]};
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    .metric-card .label {{
        font-size: 0.78rem;
        font-weight: 600;
        color: {COLOURS["text_muted"]};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }}
    .metric-card .value {{
        font-size: 1.9rem;
        font-weight: 700;
        color: {COLOURS["primary"]};
        line-height: 1;
    }}
    .metric-card .sub {{
        font-size: 0.75rem;
        color: {COLOURS["text_muted"]};
        margin-top: 0.3rem;
    }}

    /* ── Risk badge ──────────────────────────────────────────────── */
    .risk-badge {{
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}
    .risk-low      {{ background:#D4EDDA; color:#155724; }}
    .risk-moderate {{ background:#FFF3CD; color:#856404; }}
    .risk-high     {{ background:#F8D7DA; color:#721C24; }}

    /* ── PSI badge ───────────────────────────────────────────────── */
    .psi-stable   {{ background:#D4EDDA; color:#155724; border-radius:6px;
                     padding:0.2rem 0.7rem; font-weight:600; font-size:0.82rem; }}
    .psi-moderate {{ background:#FFF3CD; color:#856404; border-radius:6px;
                     padding:0.2rem 0.7rem; font-weight:600; font-size:0.82rem; }}
    .psi-critical {{ background:#F8D7DA; color:#721C24; border-radius:6px;
                     padding:0.2rem 0.7rem; font-weight:600; font-size:0.82rem; }}

    /* ── Section headers ─────────────────────────────────────────── */
    .section-header {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {COLOURS["primary"]};
        border-left: 4px solid {COLOURS["secondary"]};
        padding-left: 0.75rem;
        margin: 1.5rem 0 0.8rem 0;
    }}

    /* ── Page title ──────────────────────────────────────────────── */
    .page-title {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {COLOURS["primary"]};
        margin-bottom: 0.2rem;
    }}
    .page-subtitle {{
        font-size: 0.92rem;
        color: {COLOURS["text_muted"]};
        margin-bottom: 1.5rem;
    }}

    /* ── Info / warning banners ──────────────────────────────────── */
    .banner-info {{
        background: #E8F4F8;
        border-left: 4px solid {COLOURS["secondary"]};
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        margin: 0.8rem 0;
    }}
    .banner-warn {{
        background: #FFF8E6;
        border-left: 4px solid {COLOURS["accent"]};
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        margin: 0.8rem 0;
    }}
    .banner-danger {{
        background: #FDE8E8;
        border-left: 4px solid {COLOURS["high_risk"]};
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        margin: 0.8rem 0;
    }}

    /* ── Divider ─────────────────────────────────────────────────── */
    .styled-divider {{
        border: none;
        border-top: 2px solid {COLOURS["border"]};
        margin: 1.5rem 0;
    }}

    /* ── Streamlit overrides ─────────────────────────────────────── */
    .stButton > button {{
        background: {COLOURS["primary"]};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.4rem;
        transition: background 0.2s;
    }}
    .stButton > button:hover {{
        background: {COLOURS["secondary"]};
        color: white;
    }}
    .stDownloadButton > button {{
        background: {COLOURS["secondary"]};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }}

    </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# REUSABLE HTML COMPONENTS
# -----------------------------------------------------------------------------

def metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {sub_html}
    </div>"""


def risk_badge(label: str, level: str) -> str:
    """level: 'low' | 'moderate' | 'high'"""
    return f'<span class="risk-badge risk-{level}">{label}</span>'


def psi_badge(band: str) -> str:
    """band: 'Stable' | 'Moderate' | 'Critical'"""
    cls = band.lower()
    return f'<span class="psi-{cls}">{band}</span>'


def section_header(text: str) -> None:
    st.markdown(f'<div class="section-header">{text}</div>',
                unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", author: str = "") -> None:
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>',
                    unsafe_allow_html=True)
    if author:
        parts = author.split("·")
        name = parts[0].strip()
        matric = parts[1].strip() if len(parts) > 1 else ""
        st.markdown(f"""
        <div style="margin-top:0.5rem; padding:0 0.2rem;">
            <p style="font-family:Georgia,serif; font-size:1.15rem; font-weight:700;
                      color:#1a1a2e; letter-spacing:0.05em; margin:0 0 8px 0;
                      line-height:1.3;">{name}</p>
            <p style="font-family:'Courier New',monospace; font-size:0.78rem;
                      color:#5a7fa8; letter-spacing:0.18em; margin:0;
                      border-top:1px solid rgba(90,127,168,0.25); padding-top:6px;
                      display:inline-block;">{matric}</p>
        </div>
        """, unsafe_allow_html=True)

def banner(text: str, kind: str = "info") -> None:
    """kind: 'info' | 'warn' | 'danger'"""
    st.markdown(f'<div class="banner-{kind}">{text}</div>',
                unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)


def get_risk_level(prob: float) -> tuple[str, str]:
    """Return (label, css_level) for a given probability."""
    if prob < RISK_THRESHOLDS["low"][1]:
        return "Low Risk", "low"
    elif prob < RISK_THRESHOLDS["moderate"][1]:
        return "Moderate Risk", "moderate"
    else:
        return "High Risk", "high"


def get_psi_band(psi: float) -> str:
    """Return PSI band label."""
    if psi < PSI_THRESHOLDS["stable"][1]:
        return "Stable"
    elif psi < PSI_THRESHOLDS["moderate"][1]:
        return "Moderate"
    else:
        return "Critical"
