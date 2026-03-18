# =============================================================================
# pages/6_PSI_Monitor.py
# PSI-based drift monitoring dashboard with retraining trigger logic
# Mirrors Phase 5 — interactive version with live drift simulation
# =============================================================================

import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="PSI Monitor · IACRSM",
                   page_icon="📡", layout="wide")

from utils.styling  import (inject_global_css, page_header, section_header,
                             banner, divider, metric_card, psi_badge, COLOURS)
from utils.predictor import load_artefacts, MODEL_DISPLAY_NAMES
from utils.psi_utils import (compute_psi_feature, compute_all_feature_psi,
                              compute_score_psi, simulate_drift,
                              plot_probability_distributions,
                              plot_psi_bar_scenarios, plot_feature_psi_chart,
                              plot_psi_gauge, interpret_psi)

inject_global_css()

art = load_artefacts()
page_header("📡 PSI Drift Monitor",
            "Population Stability Index — concept drift detection and adaptive retraining logic.")
divider()

if not art.get("loaded"):
    banner(f"⚠️  Artefacts not loaded: {art.get('error','')}", kind="warn")
    st.stop()

X_train = art["X_train"]
X_test  = art["X_test"]
y_test  = art["y_test"]

available_models  = art.get("models", {})
model_options     = list(available_models.keys())
display_options   = [MODEL_DISPLAY_NAMES.get(m, m) for m in model_options]

# ── PSI Reference ──────────────────────────────────────────────────────────────
section_header("PSI Threshold Reference")
c1, c2, c3 = st.columns(3)
threshold_cards = [
    ("< 0.10",    "Stable",   COLOURS["low_risk"],
     "#D4EDDA", "No action required. Model distribution is stable."),
    ("0.10 – 0.25","Moderate", COLOURS["moderate_risk"],
     "#FFF3CD", "Monitor closely. Investigate feature shifts."),
    ("> 0.25",    "Critical", COLOURS["high_risk"],
     "#F8D7DA", "Trigger retraining. Significant concept drift detected."),
]
for col, (rng, band, text_col, bg, desc) in zip([c1, c2, c3], threshold_cards):
    with col:
        st.markdown(f"""
        <div style="background:{bg}; border-radius:10px; padding:1rem 1.2rem;
                    text-align:center; border:1px solid {text_col}30;">
            <div style="font-size:1.4rem; font-weight:800;
                        color:{text_col};">{rng}</div>
            <div style="font-weight:700; color:{text_col};
                        font-size:0.95rem; margin:0.2rem 0;">{band}</div>
            <div style="font-size:0.8rem; color:#555;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

divider()

# ── Controls ───────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 2])
with col_left:
    selected_display = st.selectbox("Classifier for PSI computation", display_options)
    selected_key     = model_options[display_options.index(selected_display)]
    active_model     = available_models[selected_key]

with col_right:
    n_bins = st.slider("PSI bins", min_value=5, max_value=20, value=10, step=1)
    st.markdown(
        "<div style='font-size:0.8rem;color:#6C757D;margin-top:0.2rem;'>"
        "More bins = finer granularity. PSI is robust to bin count changes.</div>",
        unsafe_allow_html=True
    )

divider()

# ── Generate drift scenarios ───────────────────────────────────────────────────
section_header("Drift Scenario Simulation")
st.markdown("""
<div style="font-size:0.88rem; color:#444; max-width:780px; margin-bottom:1rem;">
Concept drift is simulated by perturbing numeric feature distributions in X_test.
<strong>Moderate drift</strong>: mean shifted +0.5σ per feature.
<strong>Severe drift</strong>: mean shifted +1.5σ per feature.
PSI is then computed on predicted probability distributions (reference = X_train).
</div>
""", unsafe_allow_html=True)

with st.spinner("Simulating drift scenarios and computing PSI…"):
    X_no_drift       = simulate_drift(X_test, "none")
    X_moderate_drift = simulate_drift(X_test, "moderate")
    X_severe_drift   = simulate_drift(X_test, "severe")

    scenarios = {
        "No Drift":       X_no_drift,
        "Moderate Drift": X_moderate_drift,
        "Severe Drift":   X_severe_drift,
    }

    psi_results = {}
    for name, X_scen in scenarios.items():
        psi, table, ref_probs, cur_probs = compute_score_psi(
            active_model, X_train, X_scen, bins=n_bins
        )
        band, action = interpret_psi(psi)
        psi_results[name] = {
            "PSI":       round(psi, 4),
            "Band":      band,
            "Action":    action,
            "ref_probs": ref_probs,
            "cur_probs": cur_probs,
        }

# ── PSI Summary metrics ────────────────────────────────────────────────────────
section_header("PSI Summary — All Scenarios")
psi_cols = st.columns(3)
colours_map = {
    "Stable":   (COLOURS["low_risk"],      "#D4EDDA"),
    "Moderate": (COLOURS["moderate_risk"],  "#FFF3CD"),
    "Critical": (COLOURS["high_risk"],      "#F8D7DA"),
}

for col, (scenario_name, res) in zip(psi_cols, psi_results.items()):
    text_col, bg = colours_map[res["Band"]]
    with col:
        st.markdown(f"""
        <div style="background:{bg}; border-radius:10px; padding:1.1rem 1.3rem;
                    text-align:center; border:1px solid {text_col}40;">
            <div style="font-size:0.75rem; font-weight:700; color:{text_col};
                        text-transform:uppercase; letter-spacing:0.06em;">
                {scenario_name}</div>
            <div style="font-size:2.2rem; font-weight:800;
                        color:{text_col}; line-height:1.1;">{res['PSI']:.4f}</div>
            <div style="font-weight:700; color:{text_col};
                        font-size:0.88rem;">{res['Band']}</div>
            <div style="font-size:0.75rem; color:#555; margin-top:0.3rem;">
                {res['Action']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── PSI Bar chart ──────────────────────────────────────────────────────────────
col_bar, col_dist = st.columns(2)

with col_bar:
    section_header("PSI Comparison")
    fig = plot_psi_bar_scenarios({k: v["PSI"] for k, v in psi_results.items()})
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_dist:
    section_header("Probability Distributions")
    dist_scenario = st.selectbox(
        "View scenario", list(psi_results.keys()),
        key="dist_scenario_select"
    )
    res = psi_results[dist_scenario]
    fig = plot_probability_distributions(
        res["ref_probs"], res["cur_probs"],
        dist_scenario, res["PSI"]
    )
    st.pyplot(fig, use_container_width=True)
    plt.close()

divider()

# ── Feature-level PSI ──────────────────────────────────────────────────────────
section_header("Feature-Level PSI Analysis")

feat_scenario = st.selectbox(
    "Compute feature PSI for scenario",
    list(scenarios.keys()),
    index=2,  # default: Severe Drift
    key="feat_psi_scenario"
)

with st.spinner(f"Computing per-feature PSI for {feat_scenario}…"):
    X_sel = scenarios[feat_scenario]
    feature_psi_df = compute_all_feature_psi(X_train, X_sel, bins=n_bins)

top_feat_n = st.slider("Features to display", 5, min(30, len(feature_psi_df)),
                        15, 1, key="feat_psi_n")

feat_col, table_col = st.columns([3, 2])

with feat_col:
    fig = plot_feature_psi_chart(feature_psi_df, top_n=top_feat_n)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with table_col:
    n_stable   = (feature_psi_df["Band"] == "Stable").sum()
    n_moderate = (feature_psi_df["Band"] == "Moderate").sum()
    n_critical = (feature_psi_df["Band"] == "Critical").sum()

    st.markdown(f"""
    <div style="margin-bottom:1rem;">
    <div style="font-size:0.85rem; font-weight:600; color:{COLOURS['primary']};
                margin-bottom:0.5rem;">Band Summary</div>
    <div style="display:flex; gap:0.6rem; margin-bottom:0.8rem;">
        <div style="background:#D4EDDA;border-radius:8px;padding:0.5rem 0.8rem;text-align:center;flex:1;">
            <div style="font-size:1.3rem;font-weight:700;color:#155724;">{n_stable}</div>
            <div style="font-size:0.72rem;color:#155724;font-weight:600;">Stable</div>
        </div>
        <div style="background:#FFF3CD;border-radius:8px;padding:0.5rem 0.8rem;text-align:center;flex:1;">
            <div style="font-size:1.3rem;font-weight:700;color:#856404;">{n_moderate}</div>
            <div style="font-size:0.72rem;color:#856404;font-weight:600;">Moderate</div>
        </div>
        <div style="background:#F8D7DA;border-radius:8px;padding:0.5rem 0.8rem;text-align:center;flex:1;">
            <div style="font-size:1.3rem;font-weight:700;color:#721C24;">{n_critical}</div>
            <div style="font-size:0.72rem;color:#721C24;font-weight:600;">Critical</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        feature_psi_df.head(top_feat_n),
        use_container_width=True, hide_index=True, height=320
    )

divider()

# ── Retraining trigger logic ───────────────────────────────────────────────────
section_header("Adaptive Retraining Trigger Logic")

severe_psi  = psi_results["Severe Drift"]["PSI"]
band, action = interpret_psi(severe_psi)

if severe_psi >= 0.25:
    st.markdown(f"""
    <div style="background:#F8D7DA; border:2px solid {COLOURS['high_risk']};
                border-radius:12px; padding:1.2rem 1.6rem; margin-bottom:1rem;">
        <div style="display:flex; align-items:center; gap:1rem;">
            <div style="font-size:2rem;">🚨</div>
            <div>
                <div style="font-size:1rem; font-weight:700;
                            color:{COLOURS['high_risk']};">
                    Retraining Triggered — PSI = {severe_psi:.4f} &gt; 0.25
                </div>
                <div style="font-size:0.88rem; color:#721C24; margin-top:0.3rem;">
                    The Severe Drift scenario has exceeded the critical PSI threshold.
                    In a production pipeline, this would automatically initiate model
                    retraining on the combined training + drifted data, followed by
                    semantic version increment (v1.0.0 → v2.0.0).
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="background:#D4EDDA; border-radius:12px; padding:1rem 1.4rem;">
        ✅ PSI = {severe_psi:.4f} — below critical threshold. No retraining required.
    </div>
    """, unsafe_allow_html=True)

# ── Retraining pseudocode ──────────────────────────────────────────────────────
with st.expander("View retraining trigger pseudocode"):
    st.code("""
# PSI-based adaptive retraining logic (Phase 5)

psi, _, ref_probs, cur_probs = compute_score_psi(model, X_train, X_incoming)
band, action = interpret_psi(psi)

if psi >= 0.25:                          # Critical threshold exceeded
    X_retrain = pd.concat([X_train, X_incoming])
    y_retrain = pd.concat([y_train, y_incoming])

    retrained_model = XGBClassifier(scale_pos_weight=xgb_spw)
    search = RandomizedSearchCV(retrained_model, param_dist,
                                n_iter=50, scoring='roc_auc', cv=5)
    search.fit(X_retrain, y_retrain)

    # Semantic versioning
    version_registry["v2.0.0"] = {
        "trigger":      f"PSI = {psi:.4f} > 0.25",
        "timestamp":    datetime.now().isoformat(),
        "PSI_at_train": psi,
        "metrics":      evaluate(search.best_estimator_, X_test, y_test),
    }
    joblib.dump(search.best_estimator_, "artefacts/best_model.pkl")
    """, language="python")

divider()

# ── Version registry ───────────────────────────────────────────────────────────
section_header("Model Version Registry")

version_registry = art.get("version_registry", {})

if version_registry:
    for ver, info in version_registry.items():
        band_v, _ = interpret_psi(info.get("PSI_at_train") or 0)
        with st.expander(f"Version {ver} — {info.get('trigger','')[:60]}"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Timestamp:** {info.get('timestamp','—')}")
                st.markdown(f"**Trigger:** {info.get('trigger','—')}")
                if info.get("PSI_at_train"):
                    st.markdown(f"**PSI at training:** {info['PSI_at_train']:.4f}")
                st.markdown(f"**Model file:** `{info.get('model_file','—')}`")
            with col_b:
                metrics = info.get("metrics", {})
                if metrics:
                    st.markdown("**Evaluation metrics:**")
                    for m, v in metrics.items():
                        st.markdown(f"- {m}: **{v}**")
else:
    # Try loading from disk
    ver_path = "artefacts/versions/version_registry.json"
    if os.path.exists(ver_path):
        with open(ver_path) as f:
            registry = json.load(f)
        st.json(registry)
    else:
        banner(
            "Version registry not found. Run Phase 5 pipeline to generate "
            "<code>artefacts/versions/version_registry.json</code>.",
            kind="warn"
        )

divider()

# ── Monitoring log ─────────────────────────────────────────────────────────────
section_header("PSI Monitoring Log")

monitoring_log = art.get("monitoring_log")

if monitoring_log is not None:
    st.dataframe(monitoring_log, use_container_width=True, hide_index=True)
else:
    # Build live log from current run
    log_rows = []
    for name, res in psi_results.items():
        log_rows.append({
            "Scenario":  name,
            "PSI":       res["PSI"],
            "Band":      res["Band"],
            "Action":    res["Action"],
            "Retrained": "Yes" if res["PSI"] >= 0.25 else "No",
        })
    live_log = pd.DataFrame(log_rows)
    st.dataframe(live_log, use_container_width=True, hide_index=True)

    banner(
        "ℹ️  Showing live-computed log. Run the Phase 5 pipeline to persist "
        "this log to <code>artefacts/psi/monitoring_log.csv</code>.",
        kind="info"
    )

divider()
banner(
    "📘  <strong>PSI Reference:</strong> Population Stability Index measures the "
    "shift between two distributions. In credit scoring, it compares the "
    "reference (training) predicted probability distribution against the current "
    "incoming data distribution. A PSI &gt; 0.25 indicates the model is no longer "
    "representative of the current population and retraining is required.",
    kind="info"
)
