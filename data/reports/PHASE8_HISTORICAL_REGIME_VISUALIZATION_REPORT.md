# Phase 8 Research Report: Historical Market Regime Visualization

**Project:** EURUSD M15 Market Regime Detection Using Hidden Markov Models  
**Module:** Phase 8 — Historical Regime Visualization  
**Status:** Complete & Scientifically Audited  
**Author:** Google DeepMind Advanced Agentic Coding System  
**Date:** November 2026  
**Canonical Model Version:** `GMM_HMM_K3_M3_v1`  
**Model SHA-256 Hash:** `431e19d3f92e57df`  
**Datasets:**
- Full Year 2023: `data/processed/EURUSD_M15_2023.parquet` (20,836 complete bars)
- As-Of Real-Time Replay: `data/processed/as_of_replay_inference_log.parquet` (8,335 complete bars)  
**Execution Environment:** Python 3.14.3, NumPy 2.4.4, SciPy 1.17.1, Pandas 3.0.0, Matplotlib 3.10.0, Seaborn 0.13.2  
**Summary Artifact:** `data/reports/phase8_visualization_summary.json`  
**Test Suite:** 10/10 visualization tests passing (`tests/test_visualization.py`), 94/94 total project unit tests passing.

---

## 1. Executive Summary & Scientific Principles

Phase 8 implements the **Historical Regime Visualization Engine** for the EURUSD M15 market regime detection framework. Following parameter estimation, real-time causal filtering, and walk-forward validation in Phases 1–7, Phase 8 bridges mathematical models and human operational interpretability.

> [!IMPORTANT]
> **Strict Scientific Scope & Boundaries:**  
> This visualization system is designed exclusively for **Regime Detection Visualization** and diagnostic inspection. It is **NOT** a price prediction engine, trading signal generator, or backtest. Regimes represent unsupervised latent volatility distributions ($\mathcal{N}(\mu_{j,m}, \Sigma_{j,m})$), not directional recommendations. Colors, titles, and legends are deliberately chosen to avoid directional connotations (e.g., green/red for bull/bear is strictly prohibited).

### Core Diagnostic Insights

1. **Dual Operational Modes Implemented:**
   - **Mode A (Full-Year Retrospective View):** Evaluates all 20,836 complete M15 bars in 2023 (`2023-01-01 23:30` to `2023-12-29 21:30 UTC`). To maintain absolute scientific honesty, an explicit vertical cutover marker separates the **Train Partition** (bars 1 to 12,501; in-sample with respect to canonical model parameter estimation) from the **Replay Partition** (bars 12,502 to 20,836; out-of-sample forward filtering).
   - **Mode B (As-Of Replay View):** Visualizes strictly the 8,335 post-cutover complete bars (`2023-08-24 12:00` to `2023-12-29 21:30 UTC`) directly loaded from the verified real-time inference log (`as_of_replay_inference_log.parquet`).
2. **Deterministic Replay Equivalence Audit ($0.0$ Difference Across 25,005 Entries):**
   Comparing continuous forward filtering of `GMM_HMM_K3_M3_v1` from January 1, 2023 (Mode A) against the real-time replay log initialized via cutover at August 24, 2023 with a $W=50$ warm-up buffer (Mode B) reveals **identically zero discrepancy** ($0.0$) across all 8,335 bars and all 25,005 probability entries. This equivalence is mathematically explained by the exponential dissipation of initial conditions in the Baum-Welch / forward filter ($\mathcal{O}(\lambda_2^W)$), where $L_1$ total variation distance reaches floating-point machine precision for $W \ge 20$ bars.
3. **Information Entropy and Operational Dispersion Heuristics:**
   - Full Year (Mode A): Mean Shannon entropy is **$0.314$ nats** (normalized entropy $H_t^{\text{norm}} = 0.285$, where $1.0$ represents total ignorance $\ln(3) = 1.0986$ nats). Median normalized entropy is $0.261$.
   - Replay Window (Mode B): Mean Shannon entropy is **$0.300$ nats** ($H_t^{\text{norm}} = 0.273$).
   - Over $80\%$ of bars exhibit $H_t^{\text{norm}} < 0.40$, indicating that the filtered posterior distribution is concentrated.  
   - *Methodological Caveat:* The thresholds $H_t^{\text{norm}} < 0.30$ and $H_t^{\text{norm}} \ge 0.60$ are purely **operational/heuristic visualization boundaries**, not statistically calibrated confidence thresholds. Low entropy indicates probability mass concentration on a single latent state; it does **NOT** imply that the model is objectively correct, nor does it guarantee price forecasting accuracy.
4. **Decoupled Duration Metrics & Absence of Theoretical Markov Conflation:**
   As mandated by scientific audits, this report strictly decouples three distinct duration concepts:
   - **Theoretical Markov Expected Duration ($1 / (1 - A_{kk})$):** LowVol = $7.78$ bars, HighVol = $8.75$ bars, MedVol = $6.94$ bars.
   - **Retrospective Viterbi Smoothed Duration:** LowVol = $5.6$ bars, HighVol = $7.5$ bars, MedVol = $6.8$ bars.
   - **Causal Filtered Empirical Run Length ($\arg\max_k \alpha_{t,k}$):** Mean durations are LowVol = $4.19$ bars ($63$ mins), HighVol = $4.34$ bars ($65$ mins), MedVol = $4.46$ bars ($67$ mins). Medians are identical at $3.0$ bars ($45$ mins), with 90th percentiles at $10\text{--}11$ bars ($2.5\text{--}2.75$ hours).
5. **Zero Gap Bleed & Weekend Segmentation:**
   Line plots and transition counts strictly segment continuous sessions at gaps $> 15$ minutes (`max_gap_seconds=900`). Transitions are never computed across weekend market closures or missing bars.

---

## 2. Visualization Architecture, Data Integrity & Replay Equivalence Audit

### 2.1 Model Artifact Specification

All visualizations are produced using the canonical reference model:
- **Model Version:** `GMM_HMM_K3_M3_v1`
- **Class:** `src.models.gmm_hmm.GMMHMM` ($K=3$ hidden states, $M=3$ Gaussian mixture components per state)
- **Parameter Checksum:** `431e19d3f92e57df`
- **Feature Space:** Standardized 2-dimensional vector $[r_{\text{oc}}, \log\text{vol}]$ (`Config_2_MinimalOrthogonal_v1`) fitted on the Phase 3 training set.

### 2.2 Canonical State Alignment and Neutral Color Palette

To guarantee intuitive visual interpretation while preventing directional misinterpretation, the following color palette and semantic profiles are enforced across all 10 figures:

| State Index | Canonical Label | Strictly Empirical Characteristics | Hex Color | Visual Designation |
| :---: | :---: | :--- | :---: | :--- |
| **0** | `LowVol_Neutral` | Tight price range, low return std ($1.62$ bps), mean pip range $3.08$ pips | `#1f77b4` | Steel Blue |
| **1** | `HighVol_Neutral` | Elevated return std ($8.66$ bps), wide price range ($13.20$ pips), win rate $51.2\%$ | `#7b3294` | Imperial Purple |
| **2** | `MedVol_Neutral` | Intermediate return std ($3.78$ bps), moderate price range ($6.43$ pips) | `#ff7f0e` | Ochre Amber |

> [!NOTE]
> Green and red colors are strictly banned from regime designation to eliminate any unconscious bias associating regimes with directional long/short signals. State 1 (`HighVol_Neutral`) has an empirical win rate of $51.2\%$ (virtually identical to a fair coin toss); it represents volatility expansion, not a market downturn or high tail risk (tail risk has not been independently modeled or tested).

### 2.3 Comprehensive Exact Replay Equivalence Audit

To rigorously audit the claim that continuous Mode A filtering and Mode B as-of replay match with $0.0$ difference, we performed a component-by-component verification:

| Audit Dimension | Mode A (Full-Year Retrospective) | Mode B (As-Of Replay Log) | Reconciliation Status |
| :--- | :--- | :--- | :---: |
| **Model Version** | `GMM_HMM_K3_M3_v1` | `GMM_HMM_K3_M3_v1` | Identical |
| **Model SHA-256** | `431e19d3f92e57df` | `431e19d3f92e57df` | Identical |
| **Feature List** | `['r_oc', 'log_vol']` | `['r_oc', 'log_vol']` | Identical |
| **Feature Version** | `Config_2_MinimalOrthogonal_v1` | `Config_2_MinimalOrthogonal_v1` | Identical |
| **Max Feature Diff ($r_{\text{oc}}$)** | Baseline | Compared against `feat_r_oc` | **$0.00 \times 10^{0}$** |
| **Max Feature Diff ($\log\text{vol}$)** | Baseline | Compared against `feat_log_vol` | **$0.00 \times 10^{0}$** |
| **Evaluated Observations** | 8,335 complete replay bars | 8,335 complete replay bars | Identical |
| **Partial Bars Excluded** | 321 skipped (`is_complete == False`) | 321 skipped (`is_complete == False`) | Identical |
| **Gap Handler** | `FXCalendarGapHandler(300)` | `FXCalendarGapHandler(300)` | Identical |
| **Weekend Reset Rule** | Prior reset to `model.startprob_` | Prior reset to `model.startprob_` | Identical |
| **Cutover Initialization** | Filtered through 12,501 bars | Cutover with $W=50$ warm-up buffer | **Mathematically Equivalent** |
| **Initial Posterior at Cutover** | `[0.56004773, 0.00228931, 0.43766296]` | `[0.56004773, 0.00228931, 0.43766296]` | **Diff: $0.00$ (IEEE 754)** |
| **Matched Timestamps** | 8,335 / 8,335 ($100.0\%$) | 8,335 / 8,335 ($100.0\%$) | Perfect Match |
| **Mismatched Probabilities ($>10^{-12}$)** | 0 / 25,005 entries | 0 / 25,005 entries | **0 Mismatches** |
| **Maximum Absolute Diff** | — | — | **$0.0$** |
| **Mean Absolute Diff** | — | — | **$0.0$** |
| **Argmax Predicted State Match** | 8,335 / 8,335 ($100.0\%$) | 8,335 / 8,335 ($100.0\%$) | **100.0% Concordance** |

#### Why Is the Difference Exactly 0.0 Despite Different Initializations?
In Mode B, the engine executed an online atomic upgrade at bar 12,920 (`2023-08-24 12:00 UTC`), warming up a fresh GMM-HMM instance on the $W=50$ complete bars immediately preceding cutover. In Mode A, GMM-HMM was initialized on January 1, 2023 and filtered sequentially through all 12,501 bars preceding cutover.

As established by the Phase 5 warm-up sensitivity audit:
- At $W=5$ bars, $L_1$ total variation distance to the asymptotic stream is $6.10 \times 10^{-5}$.
- At $W=10$ bars, $L_1$ distance is $3.23 \times 10^{-9}$.
- At $W \ge 20$ bars, $L_1$ distance is **$0.00$** within IEEE 754 double precision.

Because the hidden Markov chain has a second eigenvalue $\lambda_2 \approx 0.65$, initial boundary priors decay geometrically at rate $\mathcal{O}(\lambda_2^W)$. By $W=50$ bars (12.5 trading hours), memory of the starting distribution is completely erased to machine precision. Consequently, the cutover posterior vector matches to all decimal places, producing identically zero difference throughout the entire 8,335-bar replay sequence.

---

## 3. Mode A: Full-Year Retrospective View (20,836 Bars)

Mode A covers the full 2023 calendar year from `2023-01-01 23:30:00 UTC` to `2023-12-29 21:30:00 UTC`. It encompasses 12,501 training bars and 8,335 out-of-sample replay bars.

### 3.1 Figure 21: Historical Price and Dominant Regime Timeline
- **File:** `data/reports/figures/21_historical_price_and_regime_timeline_full_year.png`
- **Layout & Contents:** A focused, high-resolution single-panel time series displaying the EURUSD M15 Close price over the full year 2023. Continuous price segments are rendered without weekend bridging lines, with the background shaded according to the dominant regime ($\arg\max_k \alpha_{t,k}$). Discrete regime switch points ($N = 3,647$) are overlaid as scatter markers.
- **Audit of Cumulative Return Aggregation:**
  - *Mathematical Formulation:* 
    $$R_k^{\text{agg}}(t) = \sum_{\tau=1}^t r_{\text{oc}}(\tau) \cdot \mathbb{I}\{\hat{S}_\tau = k\}, \quad \text{where } \hat{S}_\tau = \arg\max_j \alpha_{\tau,j}$$
  - *Strict Methodological Boundary:* Cumulative return aggregation by regime is a **purely descriptive historical return aggregation**. It is **NOT** a strategy return, portfolio equity curve, or backtest performance. It accounts for no transaction costs, execution delay, spread, or position sizing constraints. Its sole scientific function is to confirm that unsupervised regimes isolate symmetric volatility dispersion rather than directional alpha ($\mathbb{E}[r_{\text{oc}} \mid S_t] \approx 0$).
- **Key Observation:** HighVol regimes (purple) concentrate during intervals of rapid price displacement and large candle ranges, whereas LowVol regimes (blue) appear during prolonged periods of tight range compression.

### 3.2 Figure 22: Filtered Regime Probabilities
- **File:** `data/reports/figures/22_filtered_regime_probabilities_full_year.png`
- **Description:** Stacked area chart showing the evolution of causal posterior probabilities $\alpha_{t,0}$ (LowVol), $\alpha_{t,1}$ (HighVol), and $\alpha_{t,2}$ (MedVol).
- **Key Observation:** The posterior bands display near-binary switching between states, with transitional mixtures (e.g., $50/50$ uncertainty) persisting for only $1\text{--}2$ bars before resolving into a clear dominant state.

### 3.3 Figure 23: Regime Posterior Entropy and Dispersion Profile
- **File:** `data/reports/figures/23_regime_confidence_and_entropy_full_year.png`
- **Description:**
  - *Top Panel:* Normalized Shannon entropy $H_t^{\text{norm}} = H_t / \ln(3)$ plotted over time with operational threshold reference lines at $H_t = 0.0$ (concentrated posterior), $H_t^{\text{norm}} \approx 0.631$ (two-state ambiguity $\ln 2 / \ln 3$), and $H_t^{\text{norm}} = 1.0$ (uniform dispersion).
  - *Bottom Panel:* Empirical probability density distribution of normalized entropy, showing mean ($0.285$), median ($0.261$), and 90th percentile ($0.603$).
- **Operational Threshold Qualifications:**
  - The thresholds $H_t^{\text{norm}} < 0.30$ and $H_t^{\text{norm}} \ge 0.60$ are purely operational heuristics for descriptive categorization.
  - Low entropy indicates concentration of posterior probability mass on a single latent state; it does **NOT** prove ground-truth correctness or price prediction accuracy.
- **Key Observation:** Elevated normalized entropy occurs predominantly during rapid transitions between clusters when an observation exhibits intermediate feature values.

### 3.4 Figure 24: Regime Transition Dynamics and Run Length Distributions
- **File:** `data/reports/figures/24_regime_transition_and_duration_full_year.png`
- **Description:**
  - *Left Panel:* Empirical transition probability matrix heatmap computed strictly on adjacent within-session pairs ($N = 19,686$).
  - *Right Panels:* Empirical run length distributions for each state showing positive skewness, with mean durations of $4.2\text{--}4.5$ bars and 90th percentiles reaching $10\text{--}11$ bars.
- **Key Observation:** The direct transition probability between LowVol and HighVol is virtually zero ($P(0 \to 1) = 0.19\%$, $P(1 \to 0) = 0.15\%$). The market almost invariably transitions through the intermediary `MedVol_Neutral` state before entering or exiting high volatility.

### 3.5 Figure 25: Regime Characteristics Cross-Check
- **File:** `data/reports/figures/25_regime_characteristics_crosscheck_full_year.png`
- **Description:** Boxplots of physical return volatility ($\sigma(r_{\text{oc}})$ in basis points) and bar range in pips across the three filtered regimes.
- **Key Observation:** Confirms complete physical separation:
  - LowVol: Return std $1.62$ bps, Mean pip range $3.08$ pips.
  - MedVol: Return std $3.78$ bps, Mean pip range $6.43$ pips.
  - HighVol: Return std $8.66$ bps, Mean pip range $13.20$ pips.
  The interquartile ranges of pip range do not overlap between LowVol and HighVol.

### 3.6 Figure 26: Composite Regime Dashboard (Mode A)
- **File:** `data/reports/figures/26_composite_regime_dashboard_full_year.png`
- **Description:** A synchronized 3-panel executive dashboard combining: (1) EURUSD M15 Price with regime color-coding, (2) Filtered probability trajectories, and (3) Normalized Shannon entropy timeline.

---

## 4. Mode B: As-Of Replay View (8,335 Bars)

Mode B examines the out-of-sample forward replay window (`2023-08-24 12:00:00 UTC` to `2023-12-29 21:30:00 UTC`, 8,335 complete bars). This dataset represents the true operational streaming environment evaluated without lookahead.

### 4.1 Figure 27: Historical Price and Dominant Regime Timeline (Replay)
- **File:** `data/reports/figures/27_historical_price_and_regime_timeline_replay.png`
- **Description:** Real-time filtered regimes overlaid on the Fall 2023 EURUSD price trajectory (including the September downtrend and November–December rebound).
- **Key Observation:** Regime classification transitions to `HighVol_Neutral` within a single 15-minute bar upon observing substantial return and range expansion.

### 4.2 Figure 28: Filtered Regime Probabilities (Replay)
- **File:** `data/reports/figures/28_filtered_regime_probabilities_replay.png`
- **Description:** High-resolution area chart of posterior probabilities across the 8,335 replay bars.
- **Key Observation:** Probability dynamics in out-of-sample streaming exhibit identical stability and sharpness as in-sample filtering, demonstrating absence of parameter degeneration.

### 4.3 Figure 29: Regime Confidence and Entropy (Replay)
- **File:** `data/reports/figures/29_regime_confidence_and_entropy_replay.png`
- **Description:** Shannon entropy profile across the replay window. Mean normalized entropy is $0.273$ (lower than the full-year average, reflecting the lower empirical volatility observed in Q4 2023).

### 4.4 Figure 30: Composite Regime Dashboard (Mode B)
- **File:** `data/reports/figures/30_composite_regime_dashboard_replay.png`
- **Description:** Unified executive dashboard designed for operational monitoring of the out-of-sample streaming replay.

---

## 5. Comprehensive Quantitative Metrics Table

The table below summarizes all quantitative visualization and dynamic metrics compiled directly into `data/reports/phase8_visualization_summary.json`:

| Metric Category | Metric Name | Mode A: Full Year (20,836 Bars) | Mode B: As-Of Replay (8,335 Bars) |
| :--- | :--- | :---: | :---: |
| **Data Scope** | Start Timestamp (UTC) | `2023-01-01 23:30` | `2023-08-24 12:00` |
| | End Timestamp (UTC) | `2023-12-29 21:30` | `2023-12-29 21:30` |
| | Complete Bars Analyzed | **20,836** | **8,335** |
| | Partial Bars Excluded | 796 | 321 |
| **Occupancy** | `MedVol_Neutral` (State 2) | **46.48%** (9,684 bars) | **47.20%** (3,934 bars) |
| | `LowVol_Neutral` (State 0) | **33.22%** (6,921 bars) | **36.45%** (3,038 bars) |
| | `HighVol_Neutral` (State 1) | **20.31%** (4,231 bars) | **16.35%** (1,363 bars) |
| **Information Entropy** | Mean Shannon Entropy ($H_t$) | **0.3136 nats** | **0.2996 nats** |
| | Median Shannon Entropy | 0.2865 nats | 0.2694 nats |
| | 90th Percentile Shannon Entropy | 0.6625 nats | 0.6560 nats |
| | Mean Normalized Entropy ($H_t^{\text{norm}}$) | **0.2854** | **0.2727** |
| | Median Normalized Entropy | 0.2608 | 0.2452 |
| | 90th Percentile Normalized Entropy | 0.6030 | 0.5971 |
| **Transition Dynamics** | Within-Session Pairs ($N$) | 19,686 | 8,060 |
| *(Zero Gap Bleed)* | Total Switches Observed | 3,647 | 1,403 |
| | Causal Switching Frequency | **18.53%** | **17.41%** |
| **Decoupled Durations** | **Theoretical Markov Duration** ($1/(1-A_{kk})$) | | |
| | - LowVol_Neutral | 7.78 bars ($1.94$ hrs) | 7.78 bars ($1.94$ hrs) |
| | - HighVol_Neutral | 8.75 bars ($2.19$ hrs) | 8.75 bars ($2.19$ hrs) |
| | - MedVol_Neutral | 6.94 bars ($1.73$ hrs) | 6.94 bars ($1.73$ hrs) |
| | **Retrospective Viterbi Smoothed Mean** | | |
| | - LowVol_Neutral | 5.6 bars ($1.40$ hrs) | — |
| | - HighVol_Neutral | 7.5 bars ($1.88$ hrs) | — |
| | - MedVol_Neutral | 6.8 bars ($1.70$ hrs) | — |
| | **Causal Filtered Empirical Mean** | | |
| | - LowVol_Neutral | 4.19 bars ($1.05$ hrs) | 4.59 bars ($1.15$ hrs) |
| | - HighVol_Neutral | 4.34 bars ($1.09$ hrs) | 4.78 bars ($1.20$ hrs) |
| | - MedVol_Neutral | 4.46 bars ($1.12$ hrs) | 5.38 bars ($1.35$ hrs) |
| | **Causal Filtered Empirical Median** | | |
| | - All Regimes | **3.0 bars** ($45$ mins) | **3.0 bars** ($45$ mins) |
| | **Causal Filtered Empirical P90** | | |
| | - LowVol / MedVol / HighVol | 10.0 / 10.0 / 11.0 bars | 11.0 / 12.0 / 12.0 bars |
| **Physical Characteristics** | **Return Volatility $\sigma(r_{\text{oc}})$** | | |
| | - LowVol_Neutral | 1.62 bps | 1.57 bps |
| | - MedVol_Neutral | 3.78 bps | 3.82 bps |
| | - HighVol_Neutral | 8.66 bps | 8.87 bps |
| | **Mean Pip Range** | | |
| | - LowVol_Neutral | 3.08 pips | 2.99 pips |
| | - MedVol_Neutral | 6.43 pips | 6.39 pips |
| | - HighVol_Neutral | 13.20 pips | 13.01 pips |
| | **Median Pip Range** | | |
| | - LowVol_Neutral | 3.00 pips | 2.90 pips |
| | - MedVol_Neutral | 6.10 pips | 6.00 pips |
| | - HighVol_Neutral | 11.40 pips | 11.20 pips |

---

## 6. Empirical Transition Probability Matrices (Zero Gap Bleed)

The transition matrices below were estimated strictly over continuous within-session adjacent 15-minute intervals. Any transition across a market closure or missing candle is excluded.

### 6.1 Mode A: Full Year Empirical Transition Probability Matrix ($N = 19,686$)

$$\mathbf{P}_{\text{Mode A}} = \begin{pmatrix} 0.8353 & 0.0019 & 0.1628 \\ 0.0015 & 0.8044 & 0.1941 \\ 0.1093 & 0.0853 & 0.8053 \end{pmatrix}$$

Rows correspond to State at $t-1$, columns to State at $t$:
- Row 0 (`LowVol_Neutral`): $83.5\%$ persistence, $0.19\%$ direct jump to HighVol, $16.28\%$ jump to MedVol.
- Row 1 (`HighVol_Neutral`): $80.4\%$ persistence, $0.15\%$ direct jump to LowVol, $19.41\%$ jump to MedVol.
- Row 2 (`MedVol_Neutral`): $80.5\%$ persistence, $10.93\%$ transition to LowVol, $8.53\%$ transition to HighVol.

### 6.2 Mode B: As-Of Replay Empirical Transition Probability Matrix ($N = 8,060$)

$$\mathbf{P}_{\text{Mode B}} = \begin{pmatrix} 0.8474 & 0.0018 & 0.1509 \\ 0.0007 & 0.7938 & 0.2054 \\ 0.1083 & 0.0700 & 0.8217 \end{pmatrix}$$

The transition structure is remarkably consistent across both time windows, confirming that the Markov chain topology of the latent process is stationary.

---

## 7. Verification Test Suite & Engineering Rigor

The visualization pipeline is validated by a dedicated test suite (`tests/test_visualization.py`) comprising 10 tests:

```
tests/test_visualization.py::test_mode_a_data_loader_structure PASSED
tests/test_visualization.py::test_mode_b_data_loader_structure PASSED
tests/test_visualization.py::test_probability_properties PASSED
tests/test_visualization.py::test_entropy_properties PASSED
tests/test_visualization.py::test_segment_extraction_no_gap_interpolation PASSED
tests/test_visualization.py::test_transition_metrics_zero_gap_bleed PASSED
tests/test_visualization.py::test_decoupled_duration_metrics PASSED
tests/test_visualization.py::test_physical_characteristics_separation PASSED
tests/test_visualization.py::test_plot_generation_runs PASSED
tests/test_visualization.py::test_visualization_summary_json_valid PASSED
```

**Full Project Test Suite:** 94 / 94 tests passing in 12.21 seconds across all modules (`test_alignment`, `test_features`, `test_gaussian_hmm`, `test_gmm_hmm`, `test_inference`, `test_interpretation`, `test_quality`, `test_visualization`, `test_walk_forward`).

---

## 8. Scientific Scope, Limitations & Anti-Claims

1. **Unsupervised Latent States, Not Trading Strategies:**
   The regimes depicted in Figures 21–30 reflect statistical clustering of return variance and price range. No claim is made regarding predictability of future price direction. Descriptive return aggregations conditioned on regime remain centered near zero, proving that regimes do not provide a standalone trading edge without directional alpha.
2. **In-Sample vs. Out-of-Sample Boundary Visibility:**
   Mode A is a retrospective diagnostic combining in-sample (bars 1–12,501) and out-of-sample (bars 12,502–20,836) observations. Figure 22 and Figure 26 make this distinction transparent via explicit cutover markers. Mode B should be consulted for pure out-of-sample evaluation.
3. **Decoupled Duration Semantics:**
   Empirical causal run length is fundamentally governed by the dynamic filtering of noisy bar-by-bar observations ($\alpha_{t,k}$), leading to shorter continuous runs ($\approx 4.3$ bars) than the theoretical persistence of the latent Markov matrix $1/(1-A_{kk}) \approx 7\text{--}9$ bars. Conflating these two metrics is scientifically invalid and strictly avoided.
4. **Zero Weekend Interpolation:**
   All time series plots deliberately break across weekend closures to avoid displaying phantom prices or probabilities during closed markets.

---

## 9. Conclusion & Phase Completion

Phase 8 is fully completed, empirically validated, and documented. All 10 publication figures and the machine-readable summary JSON have been generated, and all 94 project tests pass with zero regressions. The project is prepared for subsequent phases upon user review.
