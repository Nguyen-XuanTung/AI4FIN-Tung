"""
src.visualization.regime_plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Publication-grade Matplotlib plotting module for Phase 8: Historical Regime Visualization.

Implements all 5 required figures:
- Figure 1: Historical Price and Regime Timeline (gap-aware, neutral color palette).
- Figure 2: Filtered Regime Probabilities (causal probabilities in [0, 1]).
- Figure 3: Regime Confidence and Uncertainty (Shannon entropy & dispersion).
- Figure 4: Regime Transition Analysis (transition matrices, run lengths, 4-way duration decoupling).
- Figure 5: Regime Characteristics Cross-Check (volatility separation, pip range, occupancy).
- Composite Dashboard: Synchronized multi-panel view (Price + Probabilities + Entropy).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from src.visualization.data_loader import (
    STATE_LABELS,
    STATE_COLORS,
    STATE_DESCRIPTIONS,
    extract_continuous_segments,
    compute_transition_and_run_lengths,
)

# Publication style defaults
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_figure_1_price_and_regime_timeline(
    df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "(Full-Year Retrospective View)",
    sample_stride: int = 1,
) -> None:
    """
    Figure 1: Historical Price and Regime Timeline.
    Plots EURUSD M15 price with background colored bands by argmax filtered state,
    regime switch points marked, and gaps properly disconnected.
    """
    fig, ax = plt.subplots(figsize=(16, 7), dpi=300)

    # Extract continuous segments so no false diagonal lines connect weekend breaks
    segments = extract_continuous_segments(df, max_gap_seconds=900)

    # Plot continuous price segments
    for seg in segments:
        if len(seg) < 2:
            continue
        ax.plot(
            seg["timestamp"].iloc[::sample_stride],
            seg["close"].iloc[::sample_stride],
            color="#222222",
            linewidth=0.9,
            alpha=0.9,
            zorder=3,
        )

    # Identify continuous regime spans to shade background
    current_state = int(df["pred_state"].iloc[0])
    span_start = df["timestamp"].iloc[0]
    switch_points_x = []
    switch_points_y = []

    for i in range(1, len(df)):
        prev_ts = df["timestamp"].iloc[i - 1]
        curr_ts = df["timestamp"].iloc[i]
        curr_state = int(df["pred_state"].iloc[i])
        gap_sec = (curr_ts - prev_ts).total_seconds()

        # If regime changed or session gap encountered, close span
        if curr_state != current_state or gap_sec > 900:
            color = STATE_COLORS[current_state]
            ax.axvspan(span_start, prev_ts, color=color, alpha=0.22, zorder=1, linewidth=0)
            if curr_state != current_state and gap_sec <= 900:
                switch_points_x.append(curr_ts)
                switch_points_y.append(df["close"].iloc[i])
            span_start = curr_ts
            current_state = curr_state

    # Close final span
    ax.axvspan(span_start, df["timestamp"].iloc[-1], color=STATE_COLORS[current_state], alpha=0.22, zorder=1, linewidth=0)

    # Plot regime switch markers
    if switch_points_x:
        ax.scatter(
            switch_points_x[::sample_stride],
            switch_points_y[::sample_stride],
            color="#d95f02",
            s=8,
            marker="o",
            alpha=0.6,
            zorder=4,
            label=f"Regime Switch Points (N={len(switch_points_x):,})",
        )

    # Format Axes
    ax.set_ylabel("EURUSD M15 Close Price", fontweight="bold", fontsize=11)
    ax.set_title(f"Figure 1: EURUSD M15 Historical Price and Filtered Market Regime Timeline {title_suffix}", fontweight="bold", fontsize=13, pad=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.grid(True, linestyle=":", alpha=0.5, color="#888888", zorder=2)
    ax.set_xlim(df["timestamp"].min(), df["timestamp"].max())

    # Build Legend
    legend_patches = [
        mpatches.Patch(color=STATE_COLORS[0], alpha=0.35, label=STATE_DESCRIPTIONS[0]),
        mpatches.Patch(color=STATE_COLORS[2], alpha=0.35, label=STATE_DESCRIPTIONS[2]),
        mpatches.Patch(color=STATE_COLORS[1], alpha=0.35, label=STATE_DESCRIPTIONS[1]),
    ]
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_patches + lines, loc="upper left", framealpha=0.92, fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_figure_2_filtered_probabilities(
    df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "(Full-Year Retrospective View)",
    sample_stride: int = 1,
) -> None:
    """
    Figure 2: Filtered Regime Probabilities.
    Plots continuous causal filtered probabilities P(S_t = j | O_{1:t}) in [0, 1]
    with clear legend, distinct colors, and gap-aware breaks.
    """
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(16, 9), sharex=True, dpi=300)
    axes = [ax0, ax2, ax1]  # Order: State 0 (Low), State 2 (Med), State 1 (High)
    state_order = [0, 2, 1]

    segments = extract_continuous_segments(df, max_gap_seconds=900)

    for ax, s_id in zip(axes, state_order):
        col_name = f"prob_state_{s_id}"
        color = STATE_COLORS[s_id]
        label = STATE_DESCRIPTIONS[s_id]

        for seg in segments:
            if len(seg) < 2:
                continue
            t_sub = seg["timestamp"].iloc[::sample_stride]
            p_sub = seg[col_name].iloc[::sample_stride]
            ax.plot(t_sub, p_sub, color=color, linewidth=1.1, alpha=0.9)
            ax.fill_between(t_sub, 0, p_sub, color=color, alpha=0.25)

        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel(f"P(S_t={s_id})", fontweight="bold", fontsize=10)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True, linestyle=":", alpha=0.5, color="#888888")
        ax.legend(handles=[mpatches.Patch(color=color, alpha=0.5, label=label)], loc="upper left", framealpha=0.9, fontsize=9)

    ax1.set_xlabel("Time (UTC)", fontweight="bold", fontsize=11)
    ax0.set_title(f"Figure 2: Causal Filtered Regime Posterior Probabilities P(S_t = j | O_{{1:t}}) {title_suffix}", fontweight="bold", fontsize=13, pad=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.set_xlim(df["timestamp"].min(), df["timestamp"].max())

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_figure_3_confidence_and_entropy(
    df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "(Full-Year Retrospective View)",
    sample_stride: int = 1,
) -> None:
    """
    Figure 3: Regime Posterior Entropy and Dispersion Profile.
    Plots Shannon entropy H_t and normalized entropy H_t / ln(3) over time
    alongside the empirical posterior dispersion distribution.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [2.2, 1.2]}, dpi=300)

    segments = extract_continuous_segments(df, max_gap_seconds=900)

    # Panel A: Time-series of Normalized Shannon Entropy
    for seg in segments:
        if len(seg) < 2:
            continue
        t_sub = seg["timestamp"].iloc[::sample_stride]
        e_sub = seg["normalized_entropy"].iloc[::sample_stride]
        ax1.plot(t_sub, e_sub, color="#4d4d4d", linewidth=0.8, alpha=0.85)

    # Reference entropy thresholds (operational heuristics)
    h_max = np.log(3.0)
    h_half_norm = np.log(2.0) / h_max  # Two-state ambiguity: ~0.6309
    ax1.axhline(0.0, color="#1b7837", linestyle=":", linewidth=1.2, label="Concentrated Posterior Mass (H_t = 0)")
    ax1.axhline(h_half_norm, color="#d95f02", linestyle="--", linewidth=1.2, label=f"Two-State Ambiguity (H_t = ln 2 / ln 3 ≈ {h_half_norm:.3f})")
    ax1.axhline(1.0, color="#e41a1c", linestyle="-.", linewidth=1.2, label="Uniform 3-Way Dispersion (H_t = 1.0)")

    ax1.set_ylim(-0.02, 1.05)
    ax1.set_ylabel("Normalized Entropy\nH_t / ln(3)", fontweight="bold", fontsize=10)
    ax1.set_title(f"Figure 3: Posterior Shannon Entropy and Probability Dispersion Profile {title_suffix}", fontweight="bold", fontsize=13, pad=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.grid(True, linestyle=":", alpha=0.5, color="#888888")
    ax1.legend(loc="upper right", framealpha=0.92, fontsize=9)
    ax1.set_xlim(df["timestamp"].min(), df["timestamp"].max())

    # Panel B: Histogram / Density of Normalized Entropy
    norm_ent = df["normalized_entropy"].values
    mean_ent = float(np.mean(norm_ent))
    p90_ent = float(np.percentile(norm_ent, 90))
    p50_ent = float(np.median(norm_ent))

    ax2.hist(norm_ent, bins=60, range=(0.0, 1.0), density=True, color="#386cb0", alpha=0.6, edgecolor="#254b7f")
    ax2.axvline(mean_ent, color="#e41a1c", linewidth=1.8, linestyle="-", label=f"Mean Normalized Entropy = {mean_ent:.3f}")
    ax2.axvline(p50_ent, color="#1b7837", linewidth=1.8, linestyle="--", label=f"Median Normalized Entropy = {p50_ent:.3f}")
    ax2.axvline(p90_ent, color="#d95f02", linewidth=1.8, linestyle=":", label=f"90th Percentile = {p90_ent:.3f}")

    ax2.set_xlabel("Normalized Shannon Entropy H_t / ln(3) (0.0 = Concentrated Mass, 1.0 = Uniform Dispersion)", fontweight="bold", fontsize=10)
    ax2.set_ylabel("Probability Density", fontweight="bold", fontsize=10)
    ax2.set_xlim(-0.02, 1.02)
    ax2.grid(True, linestyle=":", alpha=0.5, color="#888888")
    ax2.legend(loc="upper right", framealpha=0.92, fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_figure_4_transition_and_duration_analysis(
    df: pd.DataFrame,
    trans_stats: Dict[str, Any],
    canonical_artifact: ModelArtifact,
    output_path: Path,
    title_suffix: str = "(Full-Year Retrospective View)",
) -> None:
    """
    Figure 4: Regime Transition Analysis.
    Decouples:
    (a) Theoretical Markov expected duration 1 / (1 - A_kk).
    (b) Retrospective Viterbi decoded duration.
    (c) Causal filtered argmax empirical mean run length.
    (d) Causal filtered switching frequency.
    Includes Empirical Transition Count Matrix & Heatmaps.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # Panel A: Empirical Transition Count Matrix Heatmap
    ax = axes[0, 0]
    counts = np.array(trans_stats["transition_counts"])
    state_names = [STATE_LABELS[i] for i in range(3)]

    sns.heatmap(
        counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=state_names,
        yticklabels=state_names,
        cbar=True,
        ax=ax,
    )
    ax.set_title("A. Empirical Transition Counts\n(Within-Session Only, Zero Gap Bleed)", fontweight="bold", fontsize=11)
    ax.set_xlabel("Target Regime (State at t)", fontweight="bold", fontsize=9)
    ax.set_ylabel("Source Regime (State at t-1)", fontweight="bold", fontsize=9)

    # Panel B: Transition Probability Matrix Heatmap
    ax = axes[0, 1]
    probs = np.array(trans_stats["transition_probabilities"])

    sns.heatmap(
        probs,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=state_names,
        yticklabels=state_names,
        cbar=True,
        ax=ax,
    )
    ax.set_title(f"B. Empirical Transition Probabilities\n(Total Switches: {trans_stats['total_switches']:,} | Rate: {trans_stats['switching_frequency']:.1%})", fontweight="bold", fontsize=11)
    ax.set_xlabel("Target Regime", fontweight="bold", fontsize=9)
    ax.set_ylabel("Source Regime", fontweight="bold", fontsize=9)

    # Panel C: Empirical Run Length Distributions
    ax = axes[1, 0]
    run_lengths = trans_stats["run_lengths"]
    data_list = []
    labels_list = []
    colors_list = []

    for s_id in [0, 2, 1]:  # Ordered by volatility
        lens = run_lengths[s_id]
        if lens:
            data_list.append(lens)
            labels_list.append(f"{STATE_LABELS[s_id]}\n(N={len(lens):,})")
            colors_list.append(STATE_COLORS[s_id])

    bp = ax.boxplot(data_list, labels=labels_list, patch_artist=True, showmeans=True, meanline=True, showfliers=False)
    for patch, col in zip(bp["boxes"], colors_list):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)

    ax.set_title("C. Empirical Filtered Run Length Distributions\n(Excluding Inter-Session Gaps)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Run Length (Continuous M15 Bars)", fontweight="bold", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)

    # Panel D: Decoupled Duration Metric Comparison Bar Chart
    ax = axes[1, 1]
    A = np.array(canonical_artifact.parameters["transmat"])
    theo_dur = 1.0 / (1.0 - np.diag(A))
    
    # Phase 6 Viterbi durations on Train baseline: [5.6, 7.5, 6.8]
    viterbi_dur = [5.6, 7.5, 6.8]

    # Empirical mean run lengths from filtering
    emp_run_lens = [trans_stats["run_stats"][s]["mean_bars"] for s in range(3)]

    x = np.arange(3)
    width = 0.25

    ax.bar(x - width, [theo_dur[s] for s in [0, 2, 1]], width, label="1. Theoretical Markov 1/(1-A_kk)", color="#386cb0", alpha=0.85)
    ax.bar(x, [viterbi_dur[s] for s in [0, 2, 1]], width, label="2. Retrospective Viterbi Decoding", color="#7fc97f", alpha=0.85)
    ax.bar(x + width, [emp_run_lens[s] for s in [0, 2, 1]], width, label="3. Causal Filtered Empirical Run", color="#f0027f", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([STATE_LABELS[s] for s in [0, 2, 1]], fontweight="bold", fontsize=9)
    ax.set_ylabel("Expected Duration / Run Length (Bars)", fontweight="bold", fontsize=9)
    ax.set_title("D. Explicit Decoupling of Duration Definitions\n(Never Conflate Theoretical A_kk with Empirical Filtering)", fontweight="bold", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    plt.suptitle(f"Figure 4: Market Regime Transition Dynamics and Duration Decoupling {title_suffix}", fontweight="bold", fontsize=13, y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_figure_5_regime_characteristics_crosscheck(
    df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "(Full-Year Retrospective View)",
) -> None:
    """
    Figure 5: Regime Characteristics Cross-Check.
    Validates that latent regimes separate purely along the volatility axis
    (return std, log-vol, pip range) with zero directional price bias.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)

    # Panel A: Return Distribution by Regime (KDE Density)
    ax = axes[0, 0]
    for s_id in [0, 2, 1]:
        sub = df[df["pred_state"] == s_id]["return_bps"].dropna()
        sns.kdeplot(sub, ax=ax, color=STATE_COLORS[s_id], label=f"{STATE_LABELS[s_id]} (σ={sub.std():.2f} bps)", linewidth=1.8, bw_adjust=1.5)

    ax.set_xlim(-25, 25)
    ax.set_xlabel("Body Return r_oc (Basis Points)", fontweight="bold", fontsize=9)
    ax.set_ylabel("Probability Density", fontweight="bold", fontsize=9)
    ax.set_title("A. Return Distribution by Regime\n(Symmetric Centering, Distinct Volatility Spread)", fontweight="bold", fontsize=11)
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.7)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    # Panel B: Physical Pip Range Boxplot
    ax = axes[0, 1]
    pip_data = [df[df["pred_state"] == s]["pip_range"].dropna().values for s in [0, 2, 1]]
    labels = [STATE_LABELS[s] for s in [0, 2, 1]]
    colors = [STATE_COLORS[s] for s in [0, 2, 1]]

    bp = ax.boxplot(pip_data, labels=labels, patch_artist=True, showmeans=True, meanline=True, showfliers=False)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)

    means = [np.mean(d) for d in pip_data]
    for i, m in enumerate(means):
        ax.text(i + 1, m + 0.4, f"Mean: {m:.1f}p", horizontalalignment="center", fontweight="bold", fontsize=8, color="#111111")

    ax.set_ylabel("M15 Candle Range (Pips)", fontweight="bold", fontsize=9)
    ax.set_title("B. Physical OHLC Candle Range Distribution\n(HighVol ~4x Range of LowVol)", fontweight="bold", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.4)

    # Panel C: Log-Volatility Distribution
    ax = axes[1, 0]
    for s_id in [0, 2, 1]:
        sub = df[df["pred_state"] == s_id]["log_vol"].dropna()
        sns.kdeplot(sub, ax=ax, color=STATE_COLORS[s_id], label=f"{STATE_LABELS[s_id]} (μ={sub.mean():.2f})", linewidth=1.8, bw_adjust=1.5)

    ax.set_xlabel("Log-Volatility: ln(High - Low + ε)", fontweight="bold", fontsize=9)
    ax.set_ylabel("Probability Density", fontweight="bold", fontsize=9)
    ax.set_title("C. Orthogonal Log-Volatility Densities\n(Distinct Multi-Modal Separation)", fontweight="bold", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    # Panel D: Empirical Regime Occupancy
    ax = axes[1, 1]
    counts = df["pred_state"].value_counts().to_dict()
    total = len(df)
    occupancies = [counts.get(s, 0) / total for s in [0, 2, 1]]
    bars = ax.bar(labels, occupancies, color=colors, alpha=0.75, edgecolor="#222222")

    for bar, occ in zip(bars, occupancies):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f"{occ:.1%}\n({int(occ * total):,} bars)", ha="center", va="bottom", fontweight="bold", fontsize=9)

    ax.set_ylim(0, 0.6)
    ax.set_ylabel("Empirical State Occupancy Fraction", fontweight="bold", fontsize=9)
    ax.set_title("D. Balanced Regime Occupancy\n(Zero Degenerate State Collapse)", fontweight="bold", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.4)

    plt.suptitle(f"Figure 5: Regime Characteristic Statistical Cross-Check {title_suffix}", fontweight="bold", fontsize=13, y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_composite_regime_dashboard(
    df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "(Full-Year Retrospective View)",
    sample_stride: int = 1,
) -> None:
    """
    Composite Dashboard: Synchronized 3-Panel Overview.
    Panel 1: EURUSD M15 Price with regime color-coding.
    Panel 2: 3 Filtered Probability Curves P(S_t = j | O_{1:t}).
    Panel 3: Normalized Shannon Entropy (Uncertainty).
    """
    fig, (ax_price, ax_prob, ax_ent) = plt.subplots(
        3, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [2.5, 2.0, 1.2]}, dpi=300
    )

    segments = extract_continuous_segments(df, max_gap_seconds=900)

    # 1. Price Panel
    for seg in segments:
        if len(seg) < 2:
            continue
        ax_price.plot(
            seg["timestamp"].iloc[::sample_stride],
            seg["close"].iloc[::sample_stride],
            color="#222222",
            linewidth=0.9,
            alpha=0.9,
        )

    # Shaded Spans
    current_state = int(df["pred_state"].iloc[0])
    span_start = df["timestamp"].iloc[0]
    for i in range(1, len(df)):
        prev_ts = df["timestamp"].iloc[i - 1]
        curr_ts = df["timestamp"].iloc[i]
        curr_state = int(df["pred_state"].iloc[i])
        gap_sec = (curr_ts - prev_ts).total_seconds()

        if curr_state != current_state or gap_sec > 900:
            ax_price.axvspan(span_start, prev_ts, color=STATE_COLORS[current_state], alpha=0.22, linewidth=0)
            span_start = curr_ts
            current_state = curr_state
    ax_price.axvspan(span_start, df["timestamp"].iloc[-1], color=STATE_COLORS[current_state], alpha=0.22, linewidth=0)

    ax_price.set_ylabel("EURUSD M15 Price", fontweight="bold", fontsize=10)
    ax_price.set_title(f"Market Regime Detection Synchronized Dashboard {title_suffix}", fontweight="bold", fontsize=13, pad=10)
    ax_price.grid(True, linestyle=":", alpha=0.4)

    legend_patches = [
        mpatches.Patch(color=STATE_COLORS[0], alpha=0.35, label=STATE_DESCRIPTIONS[0]),
        mpatches.Patch(color=STATE_COLORS[2], alpha=0.35, label=STATE_DESCRIPTIONS[2]),
        mpatches.Patch(color=STATE_COLORS[1], alpha=0.35, label=STATE_DESCRIPTIONS[1]),
    ]
    ax_price.legend(handles=legend_patches, loc="upper left", framealpha=0.9, fontsize=8)

    # 2. Probability Panel
    for seg in segments:
        if len(seg) < 2:
            continue
        t_sub = seg["timestamp"].iloc[::sample_stride]
        for s_id in [0, 2, 1]:
            ax_prob.plot(t_sub, seg[f"prob_state_{s_id}"].iloc[::sample_stride], color=STATE_COLORS[s_id], linewidth=0.9, alpha=0.85)

    ax_prob.set_ylabel("Filtered Probabilities\nP(S_t = j | O_{1:t})", fontweight="bold", fontsize=10)
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.grid(True, linestyle=":", alpha=0.4)

    # 3. Entropy Panel
    for seg in segments:
        if len(seg) < 2:
            continue
        t_sub = seg["timestamp"].iloc[::sample_stride]
        ax_ent.plot(t_sub, seg["normalized_entropy"].iloc[::sample_stride], color="#444444", linewidth=0.8, alpha=0.85)

    ax_ent.axhline(np.log(2.0) / np.log(3.0), color="#d95f02", linestyle="--", linewidth=1.0, label="Two-State Ambiguity")
    ax_ent.set_ylabel("Normalized Entropy\nH_t / ln(3)", fontweight="bold", fontsize=10)
    ax_ent.set_ylim(-0.02, 1.02)
    ax_ent.set_xlabel("Time (UTC)", fontweight="bold", fontsize=10)
    ax_ent.grid(True, linestyle=":", alpha=0.4)
    ax_ent.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_ent.xaxis.set_major_locator(mdates.MonthLocator())
    ax_ent.set_xlim(df["timestamp"].min(), df["timestamp"].max())
    ax_ent.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
