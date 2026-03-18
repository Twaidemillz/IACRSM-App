# =============================================================================
# utils/psi_utils.py
# PSI computation, drift simulation, and monitoring visualisations
# Mirrors Phase 5 logic for interactive use in the Streamlit app
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.styling import COLOURS, get_psi_band


# -----------------------------------------------------------------------------
# CORE PSI FUNCTIONS
# -----------------------------------------------------------------------------

def compute_psi_feature(expected: np.ndarray,
                        actual: np.ndarray,
                        bins: int = 10) -> tuple[float, pd.DataFrame]:
    """
    Compute PSI for a single feature or probability distribution.

    Parameters
    ----------
    expected : reference distribution (training)
    actual   : current distribution (new / drifted data)
    bins     : number of quantile bins

    Returns
    -------
    psi   : float
    table : pd.DataFrame with bin-level breakdown
    """
    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual,   bins=breakpoints)

    eps     = 1e-6
    exp_pct = (exp_counts / len(expected)) + eps
    act_pct = (act_counts / len(actual))   + eps

    psi_bins = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    psi      = float(psi_bins.sum())

    table = pd.DataFrame({
        "Bin":          range(1, len(psi_bins) + 1),
        "Expected_%":   np.round(exp_pct * 100, 2),
        "Actual_%":     np.round(act_pct * 100, 2),
        "PSI_contrib":  np.round(psi_bins, 6),
    })
    return psi, table


def compute_all_feature_psi(X_ref: pd.DataFrame,
                             X_cur: pd.DataFrame,
                             bins: int = 10) -> pd.DataFrame:
    """
    Compute PSI for every feature column between two DataFrames.
    Returns a sorted DataFrame with Feature, PSI, Band columns.
    """
    records = []
    for col in X_ref.columns:
        psi, _ = compute_psi_feature(X_ref[col].values,
                                     X_cur[col].values, bins)
        records.append({"Feature": col, "PSI": round(psi, 4)})

    df       = pd.DataFrame(records).sort_values("PSI", ascending=False)
    df["Band"] = df["PSI"].apply(get_psi_band)
    return df.reset_index(drop=True)


def compute_score_psi(model,
                      X_ref: pd.DataFrame,
                      X_cur: pd.DataFrame,
                      bins: int = 10) -> tuple[float, pd.DataFrame]:
    """
    PSI on predicted probability distributions.
    Reference = model on X_ref; Current = model on X_cur.
    """
    ref_probs = model.predict_proba(X_ref)[:, 1]
    cur_probs = model.predict_proba(X_cur)[:, 1]
    psi, table = compute_psi_feature(ref_probs, cur_probs, bins)
    return psi, table, ref_probs, cur_probs


# -----------------------------------------------------------------------------
# DRIFT SIMULATION
# -----------------------------------------------------------------------------

def simulate_drift(X: pd.DataFrame, drift_level: str = "moderate",
                   seed: int = 42) -> pd.DataFrame:
    """
    Simulate concept drift by perturbing numeric feature distributions.

    drift_level : 'none' | 'moderate' | 'severe'
    """
    rng      = np.random.default_rng(seed)
    X_out    = X.copy()
    num_cols = X_out.select_dtypes(include=[np.number]).columns

    if drift_level == "none":
        return X_out

    for col in num_cols:
        std  = X_out[col].std()
        if drift_level == "moderate":
            X_out[col] = (X_out[col]
                          + 0.5 * std
                          + rng.normal(0, 0.1 * std, len(X_out)))
        elif drift_level == "severe":
            X_out[col] = (X_out[col]
                          + 1.5 * std
                          + rng.normal(0, 0.3 * std, len(X_out)))

    return X_out


# -----------------------------------------------------------------------------
# VISUALISATIONS
# -----------------------------------------------------------------------------

def plot_probability_distributions(ref_probs: np.ndarray,
                                   cur_probs: np.ndarray,
                                   scenario_name: str,
                                   psi: float) -> plt.Figure:
    """
    Overlapping histogram of reference vs current probability distributions.
    """
    band = get_psi_band(psi)
    colour_map = {"Stable": COLOURS["psi_stable"],
                  "Moderate": COLOURS["psi_moderate"],
                  "Critical": COLOURS["psi_critical"]}

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ref_probs, bins=25, alpha=0.65,
            color=COLOURS["secondary"], label="Reference (Train)")
    ax.hist(cur_probs, bins=25, alpha=0.65,
            color=colour_map[band], label=f"Current ({scenario_name})")
    ax.axvline(0.5, color="black", linestyle="--", lw=1, alpha=0.7,
               label="Decision boundary (0.5)")
    ax.set_xlabel("Predicted Default Probability", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(
        f"{scenario_name} — PSI = {psi:.4f}  [{band}]",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def plot_psi_gauge(psi: float, title: str = "PSI") -> plt.Figure:
    """
    Simple colour-coded bar gauge for a single PSI value.
    """
    band = get_psi_band(psi)
    colour_map = {"Stable": COLOURS["psi_stable"],
                  "Moderate": COLOURS["psi_moderate"],
                  "Critical": COLOURS["psi_critical"]}

    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh(0, min(psi, 0.5), color=colour_map[band],
            height=0.6, alpha=0.85)
    ax.axvline(0.10, color=COLOURS["psi_moderate"],
               linestyle="--", lw=1.2, label="0.10")
    ax.axvline(0.25, color=COLOURS["psi_critical"],
               linestyle="--", lw=1.2, label="0.25")
    ax.set_xlim(0, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("PSI Value", fontsize=9)
    ax.set_title(f"{title}: {psi:.4f}  [{band}]",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, title="Thresholds", title_fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_feature_psi_chart(feature_psi_df: pd.DataFrame,
                           top_n: int = 15) -> plt.Figure:
    """
    Horizontal bar chart of per-feature PSI values with colour bands.
    """
    df = feature_psi_df.head(top_n).copy()
    colour_map = {"Stable": COLOURS["psi_stable"],
                  "Moderate": COLOURS["psi_moderate"],
                  "Critical": COLOURS["psi_critical"]}
    bar_colours = df["Band"].map(colour_map)
    feats = [f[:35] for f in df["Feature"]]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.38)))
    ax.barh(feats[::-1], df["PSI"].values[::-1],
            color=bar_colours.values[::-1], alpha=0.85,
            edgecolor="white", linewidth=0.4)
    ax.axvline(0.10, color=COLOURS["psi_moderate"],
               linestyle="--", lw=1.4, label="Moderate (0.10)")
    ax.axvline(0.25, color=COLOURS["psi_critical"],
               linestyle="--", lw=1.4, label="Critical (0.25)")

    legend_patches = [
        mpatches.Patch(color=COLOURS["psi_stable"],   label="Stable"),
        mpatches.Patch(color=COLOURS["psi_moderate"],  label="Moderate"),
        mpatches.Patch(color=COLOURS["psi_critical"],  label="Critical"),
    ]
    ax.legend(handles=legend_patches, fontsize=9)
    ax.set_xlabel("PSI Value", fontsize=11)
    ax.set_title("Feature-Level PSI", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_psi_bar_scenarios(psi_dict: dict) -> plt.Figure:
    """
    Bar chart comparing PSI across multiple drift scenarios.
    psi_dict: {"No Drift": 0.01, "Moderate Drift": 0.15, "Severe Drift": 0.42}
    """
    names  = list(psi_dict.keys())
    values = list(psi_dict.values())
    colour_map = {"Stable": COLOURS["psi_stable"],
                  "Moderate": COLOURS["psi_moderate"],
                  "Critical": COLOURS["psi_critical"]}
    bar_colours = [colour_map[get_psi_band(v)] for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, values, color=bar_colours, alpha=0.85,
                  edgecolor="black", linewidth=0.4, width=0.45)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.axhline(0.10, color=COLOURS["psi_moderate"],
               linestyle="--", lw=1.4, label="Moderate threshold (0.10)")
    ax.axhline(0.25, color=COLOURS["psi_critical"],
               linestyle="--", lw=1.4, label="Critical threshold (0.25)")
    ax.set_ylabel("PSI Value", fontsize=11)
    ax.set_title("Population Stability Index — Drift Scenarios",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(values) * 1.3 if values else 0.5)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# ACTION MAPPING
# -----------------------------------------------------------------------------

def interpret_psi(psi: float) -> tuple[str, str]:
    """Return (band, recommended_action)."""
    if psi < 0.10:
        return "Stable",   "No action required — model is stable."
    elif psi < 0.25:
        return "Moderate", "Monitor closely — investigate feature shifts."
    else:
        return "Critical", "Trigger retraining — significant concept drift detected."
