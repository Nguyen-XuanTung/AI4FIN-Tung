# Phase 7 Experimental Design: Out-of-Sample Walk-Forward Validation & Cross-Regime Robustness

**Project:** EURUSD M15 Market Regime Detection Using Hidden Markov Models  
**Module:** Phase 7 — Walk-Forward Evaluation & Cross-Regime Robustness Analysis  
**Status:** Methodologically Refined Experimental Blueprint (Pre-Implementation)  
**Author:** Google DeepMind Advanced Agentic Coding System  
**Date:** November 2026  

---

## 1. Executive Summary & Core Research Questions

The primary objective of this research is estimating causal filtered market regime probabilities:
$$P(S_t = j \mid O_{1:t})$$
without forward-looking data leakage or subjective directional forcing.

Phases 1 through 6 established:
- High-integrity M15 candle aggregation, calendar gap handling, and orthogonal feature engineering ($r_{\text{oc}}$ body return and $\log\text{vol}$ range).
- Baseline Gaussian HMM ($K=3, M=1$) and GMM-HMM architectures ($K=3, M=2$ and $K=3, M=3$).
- Streaming inference engine with FX session calendar gap propagation ($\mathbf{A}^n$ within open sessions, `model.startprob_` on weekend closures), partial-bar skipping, and immutable versioning.
- In-sample regime interpretation via multi-sequence forward-backward smoothing, law of total variance moment aggregation, and Hungarian label alignment to a canonical reference profile (`GMM_HMM_K3_M3_v1`).

Phase 3 evaluated a single static chronological split (Train: Jan–Aug 2023, Val: Aug–Oct 2023, Test: Oct–Dec 2023). However, financial time series exhibit non-stationarity, shifting central bank policy regimes, and episodic volatility clustering. A single static split cannot establish whether regime topologies and filtering accuracy persist over time.

### Core Research Questions (RQs):
1. **RQ1 (Temporal Stability across Chronological Windows):**
   Do estimated transition matrices $\mathbf{A}$, emission moments $(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$, and stationary distributions $\boldsymbol{\pi}$ persist across successive chronological training windows, or do they suffer rapid parameter decay? Does an expanding training window dampen parameter variance compared to a fixed-width rolling window?
2. **RQ2 (Architectural Generalization on Out-of-Sample Observations):**
   How do `Gaussian_HMM_K3_M1`, `GMM_HMM_K3_M2`, and `GMM_HMM_K3_M3` compare in sequential out-of-sample predictive log-likelihood? Does the higher capacity of $M=3$ overfit unseen folds, or does its ability to model fat tails and skewed emissions provide superior out-of-sample predictive density compared to $M=2$ and $M=1$?
3. **RQ3 (Controlled Parameter & Feature Sensitivity):**
   - How does the variance-relative covariance regularization floor ($\text{cov\_reg} \in \{10^{-4}, 10^{-3}, 10^{-2}\}$) affect numerical stability and out-of-sample likelihood in GMM-HMM?
   - How does joint feature-representation and diurnal seasonal adjustment via `DiurnalVolatilityScaler` ($[r_{\text{oc}}, \text{range\_zscore}]$) compare against the baseline orthogonal representation ($[r_{\text{oc}}, \log\text{vol}]$)?
4. **RQ4 (Regime Semantic Persistence & Filtering Dynamics):**
   Under Hungarian bipartite alignment against the canonical reference profile, do state identities (Low Volatility Neutral, Medium Volatility Neutral, High Volatility Neutral) remain distinct and monotonically ordered by volatility? How do out-of-sample causal filtered switching rates and regime run lengths behave across quiet versus turbulent market periods?

---

## 2. Review and Elimination of Arbitrary Acceptance Thresholds

The previous draft proposed scalar thresholds ($c_{\text{match}} \le 1.50$, $\Delta_{\text{global}} \ge 0.50$, and switching rate between $15\%$ and $25\%$). These arbitrary cutoffs **must not be treated as established pass/fail acceptance criteria**. Below is the formal review and methodological refinement for each metric:

### 2.1 Matching Cost
- **What it measures:** The normalized Euclidean distance $\|\tilde{\mathbf{v}}_{\text{candidate}} - \tilde{\mathbf{v}}_{\text{reference}}\|_2$ in 5D standardized profile space ($[\mu_r, \sigma_r, \mu_{vol}, \sigma_{vol}, P(r > 0)]$) between candidate state profiles and canonical reference profiles.
- **Statistical basis:** **None.** The value $1.50$ is not a calibrated critical value or $\chi^2_5$ quantile; it is an uncalibrated scalar heuristic.
- **Classification:** **Descriptive / Diagnostic Metric.**
- **Refined Evaluation Approach:**
  * Report continuous empirical distribution statistics across folds: $\min_f c_f$, $\text{median}_f c_f$, $\max_f c_f$.
  * Verify **topological consistency**: Require that the volatility ordering across aligned states remains strictly monotonic:
    $$\sigma(r \mid \text{LowVol}) < \sigma(r \mid \text{MedVol}) < \sigma(r \mid \text{HighVol})$$
  * Maintain $d_{\max} = 2.50$ strictly as an **operational rejection ceiling** (representing an RMS normalized feature deviation of $\approx 1.12$ standard deviations across dimensions, established in Phase 6) to flag uncharacteristic or degenerate topologies as `UNMATCHED_ANOMALY` without discarding the entire fold.

### 2.2 Global Assignment Gap (Murty Second-Best Margin)
- **What it measures:** The difference in total assignment cost between the optimal Hungarian matching and the second-best globally valid permutation: $\Delta_{\text{global}} = c_{\text{2nd-best}} - c_{\text{best}}$.
- **Statistical basis:** **None.** While Murty's algorithm is mathematically exact for ranking assignments, the scalar margin $0.50$ is not a calibrated hypothesis test threshold.
- **Classification:** **Diagnostic Margin.**
- **Refined Evaluation Approach:**
  * Report the actual numerical gap $\Delta_{\text{global}}$ and candidate-level competitor margins $\Delta_{\text{cand}, k} = c_{k, \text{2nd-best-ref}} - c_{k, \text{best-ref}}$ for each fold.
  * If a candidate-level margin falls below $\delta_{\text{ambiguity}} = 0.20$, flag the state as `AMBIGUOUS_ALIGNMENT` and diagnose whether two latent states are merging in that training window (e.g. Medium Volatility collapsing into Low Volatility). Do not invalidate the experiment based on an arbitrary $0.50$ global threshold.

### 2.3 Causal Switching Frequency
- **What it measures:** The fraction of adjacent bars where the causal filtered argmax state changes: $\frac{1}{T-1} \sum_{t=2}^T \mathbb{I}(\arg\max_k \alpha_t(k) \neq \arg\max_k \alpha_{t-1}(k))$.
- **Statistical basis:** **None.** The range $[15\%, 25\%]$ was simply an empirical observation from 2023 Train and Replay data. Financial market dynamics fluctuate: during quiet summer consolidation, switching rates may naturally fall below $10\%$, while during major macro news events (CPI, NFP, rate decisions), switching rates may exceed $30\%$. Mandating a narrow 15%–25% band confuses an empirical observation with a normative validity criterion.
- **Classification:** **Descriptive Behavioral Diagnostic.**
- **Refined Evaluation Approach:**
  * Interpret switching frequency jointly with:
    1. **Model-implied Markov geometric duration:** $E[T_k] = 1 / (1 - A_{kk})$.
    2. **Empirical filtered run lengths:** Mean, median, and 90th percentile run lengths in bars and hours.
    3. **Filtered probability Shannon entropy:**
       $$H_t = -\sum_{k=1}^K \alpha_t(k) \log \alpha_t(k)$$
       If switching frequency spikes in a fold, inspect $H_t$: does the increase reflect decisive transitions between confident states ($\alpha \approx 1$), or rapid chattering due to posterior ambiguity ($H_t \approx \log 3 \approx 1.10$ with $\alpha \approx [0.33, 0.33, 0.33]$)?

---

## 3. Verified Walk-Forward Fold Architecture (2023 In-Dataset Benchmark)

### 3.1 Methodological Grounding & Structural Definitions

To prevent conceptual confusion, the following definitions are strictly maintained:
- **Expanding-Window Training:** Each successive fold $f$ trains on all historical observations from the start of the dataset ($t_0$) up to the fold's training cutoff: $\mathcal{D}_{\text{train}}^{(f)} = [t_0, T_{\text{train}}^{(f)}]$. The training sets are strictly nested: $\mathcal{D}_{\text{train}}^{(1)} \subset \mathcal{D}_{\text{train}}^{(2)} \subset \dots \subset \mathcal{D}_{\text{train}}^{(F)}$. Parameter estimates across folds are serially dependent and must not be treated as independent estimates.
- **Sequential Validation Window:** A dedicated non-overlapping window $[T_{\text{train}}^{(f)} + 1, T_{\text{val}}^{(f)}]$ immediately following training, used strictly for hyperparameter sensitivity assessment (e.g. comparing regularization floors) before evaluating the test window.
- **Sequential Out-of-Sample Evaluation Window:** The test period $[T_{\text{val}}^{(f)} + 1, T_{\text{test}}^{(f)}]$ evaluated strictly forward in time using frozen parameters.
- **Historical Reuse of Observations in Later Folds:** An observation tested out-of-sample in Fold 1 ($T_{\text{test}}^{(1)} = \text{August 2023}$) becomes part of the training set in Fold 3 ($T_{\text{train}}^{(3)} = \text{Jan–Aug 2023}$). This reflects realistic operational retraining, but reinforces that cross-fold evaluations are sequential diagnostics, not independent cross-validation splits.
- **Retrospective Diagnostic Benchmark Status:** The entire 2023 dataset was inspected during Phases 1–6 (feature engineering, model selection, GMM-HMM tuning, and Phase 5 real-time replay). Therefore, these 2023 folds constitute a **reproducible retrospective benchmark of the walk-forward mechanism**, not an uncontaminated forward validation.

---

### 3.2 Exact Verified Fold Boundaries & Counts

The 2023 dataset contains **21,632 total M15 bars**, of which exactly **20,836 are complete M15 candles** (`is_complete == True`, 15 active M1 ticks) and **796 are partial candles** (`is_complete == False`).

> [!IMPORTANT]
> **Complete-Bar Isolation:**
> In accordance with Phase 2, Phase 3, and Phase 5 findings, partial candles have mechanically compressed price ranges. All partial candles are **100% excluded** from training, validation, and test likelihood evaluation.

The table below presents the verified timestamps (UTC to the minute) and exact complete-bar counts verified directly against `data/processed/EURUSD_M15_2023.parquet`:

| Fold ID | Training Window (UTC) | Complete Train Bars | Validation Window (UTC) | Complete Val Bars | Test Window (UTC) | Complete Test Bars |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `2023-01-01 23:30` to `2023-06-30 21:45` | **9,472** | `2023-07-02 23:00` to `2023-07-31 23:30` | **1,414** | `2023-08-01 00:00` to `2023-08-31 23:30` | **2,120** |
| **Fold 2** | `2023-01-01 23:30` to `2023-07-31 23:30` | **10,886** | `2023-08-01 00:00` to `2023-08-31 23:30` | **2,120** | `2023-09-01 00:00` to `2023-09-29 21:45` | **1,887** |
| **Fold 3** | `2023-01-01 23:30` to `2023-08-31 23:30` | **13,006** | `2023-09-01 00:00` to `2023-09-29 21:45` | **1,887** | `2023-10-01 22:00` to `2023-10-31 23:45` | **2,056** |
| **Fold 4** | `2023-01-01 23:30` to `2023-09-29 21:45` | **14,893** | `2023-10-01 22:00` to `2023-10-31 23:45` | **2,056** | `2023-11-01 00:00` to `2023-11-30 23:45` | **2,027** |
| **Fold 5** | `2023-01-01 23:30` to `2023-10-31 23:45` | **16,949** | `2023-11-01 00:00` to `2023-11-30 23:45` | **2,027** | `2023-12-01 00:00` to `2023-12-29 21:30` | **1,860** |

#### Window Integrity Audit:
1. **Zero Accidental Overlap:**
   - Between M6 end (`2023-06-30 21:45`) and M7 start (`2023-07-02 23:00`): Normal weekend closure. Zero overlapping bars.
   - Between M7 end (`2023-07-31 23:30`) and M8 start (`2023-08-01 00:00`): Contiguous session boundary. Zero overlapping bars.
   - Between M9 end (`2023-09-29 21:45`) and M10 start (`2023-10-01 22:00`): Normal weekend closure. Zero overlapping bars.
   - Similar verification across all fold partitions confirms strictly non-overlapping subsets.
2. **Window Adequacy:**
   - Minimum training length: **9,472 complete bars** (~6 months, ~130 trading days). Ample sample size to fit 3 states and full covariance matrices without parameter degeneracy.
   - Validation windows: **1,414 to 2,120 complete bars** (~4 weeks of 24h trading). Ample length to evaluate out-of-sample likelihood and tune regularization.
   - Test windows: **1,860 to 2,120 complete bars** (~20–22 trading days). Sufficient statistical power to compute empirical occupancy and per-bar predictive densities.
3. **Multi-Sequence Gap Handling:**
   Within every window, trading sessions are segmented into continuous sequences using `identify_continuous_sequences()` from `src/models/splits.py`. Weekend closures reset sequence boundaries; EM and forward filtering never propagate transitions across market breaks.

---

## 4. Posterior Initialization at Train → Validation and Validation → Test Boundaries

A critical methodological requirement of out-of-sample walk-forward evaluation is explicitly specifying how the belief state $\boldsymbol{\alpha}$ is initialized and propagated across partition boundaries: from Training to Validation, and from Validation to Test.

### 4.1 Evaluation Policies: Policy A (Continuous) vs. Policy B (Independent)

#### Policy A — Continuous Forward Evaluation (PRIMARY):
Policy A simulates genuine continuous production deployment:
1. **Fit Parameters on Train:** The model is fitted strictly on the complete observations of the training window $\mathcal{D}_{\text{train}}^{(f)}$.
2. **Filter Train In-Sample:** The frozen model sequentially filters the training window to establish causal belief dynamics.
3. **Extract Terminal Train Belief:** The posterior distribution at the final complete training bar, $\boldsymbol{\alpha}_{T_{\text{train}}}$, is obtained.
4. **Propagate Across Train $\to$ Val Boundary:** The calendar gap between the last training bar $t_{\text{train\_last}}$ and the first validation bar $t_{\text{val\_first}}$ is classified via `FXCalendarGapHandler`:
   - If within an open FX trading session (`requires_reset == False`):
     $$\boldsymbol{\alpha}_{\text{prior}} = \boldsymbol{\alpha}_{T_{\text{train}}} \mathbf{A}^n, \quad n = \frac{\Delta t}{15\text{ min}}$$
   - If crossing a genuine market closure (`requires_reset == True`, e.g. Friday close to Sunday open):
     $$\boldsymbol{\alpha}_{\text{prior}} = \text{model.startprob\_}$$
     *(Note: Strictly `model.startprob_`, the initial state distribution from model fitting, NEVER the asymptotic stationary distribution $\boldsymbol{\pi}$).*
5. **Update with First Validation Observation:**
   $$\boldsymbol{\alpha}_{1} \propto \boldsymbol{\alpha}_{\text{prior}} \odot \mathbf{b}(O_{\text{val\_first}})$$
6. **Filter Validation Sequentially:** Validation bars are processed sequentially with frozen parameters.
7. **Propagate Across Val $\to$ Test Boundary:** The terminal validation posterior $\boldsymbol{\alpha}_{T_{\text{val}}}$ is carried into the test window following the exact same `FXCalendarGapHandler` rules: $\mathbf{A}^n$ during open sessions, and resetting to `model.startprob_` upon weekend market closure.

#### Policy B — Independent Benchmark Sequence (DIAGNOSTIC):
- Both Validation and Test windows are evaluated as isolated sequences starting unconditionally from:
  $$\boldsymbol{\alpha}_{\text{prior}} = \text{model.startprob\_}$$
- **Rationale:** Policy B evaluates the test window independent of validation history, serving as a diagnostic benchmark to quantify whether carrying forward the continuous posterior from validation has any measurable or enduring effect on out-of-sample predictive log-likelihood.

---

### 4.2 Exact Empirical Classification of Train → Validation Boundaries

Applying `FXCalendarGapHandler` to the exact boundaries between `df_train` and `df_val` across all 5 folds produces the following verified transition classifications:

| Fold ID | Last Train Bar ($t_{\text{train\_last}}$) | First Val Bar ($t_{\text{val\_first}}$) | Elapsed Delta | Gap Classification | Markov Transitions ($n$) | Reset Mandated? | Policy A Propagation Rule |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `2023-06-30 21:45` | `2023-07-02 23:00` | 49h 15m (177,300s) | `market_closure` | $n = 197$ | **True** | **Prior resets to `model.startprob_`** (Weekend closure Fri $\to$ Sun) |
| **Fold 2** | `2023-07-31 23:30` | `2023-08-01 00:00` | 30 min (1,800s) | `active_missing` | $n = 2$ | False | $\boldsymbol{\alpha}_{\text{prior}} = \boldsymbol{\alpha}_{T_{\text{train}}} \mathbf{A}^2$ (Bar 23:45 was partial/skipped) |
| **Fold 3** | `2023-08-31 23:30` | `2023-09-01 00:00` | 30 min (1,800s) | `active_missing` | $n = 2$ | False | $\boldsymbol{\alpha}_{\text{prior}} = \boldsymbol{\alpha}_{T_{\text{train}}} \mathbf{A}^2$ (Bar 23:45 was partial/skipped) |
| **Fold 4** | `2023-09-29 21:45` | `2023-10-01 22:00` | 48h 15m (173,700s) | `market_closure` | $n = 193$ | **True** | **Prior resets to `model.startprob_`** (Weekend closure Fri $\to$ Sun) |
| **Fold 5** | `2023-10-31 23:45` | `2023-11-01 00:00` | 15 min (900s) | `normal` | $n = 1$ | False | $\boldsymbol{\alpha}_{\text{prior}} = \boldsymbol{\alpha}_{T_{\text{train}}} \mathbf{A}^1$ (Normal consecutive step) |

---

### 4.3 Exact Empirical Classification of Validation → Test Boundaries

Applying `FXCalendarGapHandler` to the exact boundaries between `df_val` and `df_test` across all 5 folds produces the following verified transition classifications:

| Fold ID | Last Val Bar ($t_{\text{val\_last}}$) | First Test Bar ($t_{\text{test\_first}}$) | Elapsed Delta | Gap Classification | Markov Transitions ($n$) | Reset Mandated? | Policy A Propagation Rule |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `2023-07-31 23:30` | `2023-08-01 00:00` | 30 min (1,800s) | `active_missing` | $n = 2$ | False | $\boldsymbol{\alpha}_{\text{prior}} = \boldsymbol{\alpha}_{T_{\text{val}}} \mathbf{A}^2$ (Bar 23:45 was partial/skipped) |
| **Fold 2** | `2023-08-31 23:30` | `2023-09-01 00:00` | 30 min (1,800s) | `active_missing` | $n = 2$ | False | $\boldsymbol{\alpha}_{\text{prior}} = \boldsymbol{\alpha}_{T_{\text{val}}} \mathbf{A}^2$ (Bar 23:45 was partial/skipped) |
| **Fold 3** | `2023-09-29 21:45` | `2023-10-01 22:00` | 48h 15m (173,700s) | `market_closure` | $n = 193$ | **True** | **Prior resets to `model.startprob_`** (Weekend closure Fri $\to$ Sun) |
| **Fold 4** | `2023-10-31 23:45` | `2023-11-01 00:00` | 15 min (900s) | `normal` | $n = 1$ | False | $\boldsymbol{\alpha}_{\text{prior}} = \boldsymbol{\alpha}_{T_{\text{val}}} \mathbf{A}^1$ (Normal consecutive step) |
| **Fold 5** | `2023-11-30 23:45` | `2023-12-01 00:00` | 15 min (900s) | `normal` | $n = 1$ | False | $\boldsymbol{\alpha}_{\text{prior}} = \boldsymbol{\alpha}_{T_{\text{val}}} \mathbf{A}^1$ (Normal consecutive step) |

#### Required Unit Tests for Boundary Implementation:
When Phase 7 is implemented, `tests/test_walk_forward.py` must verify:
1. `test_boundary_policy_a_active_gap_propagation`: Verifies $\boldsymbol{\alpha}_{\text{prior}} = \boldsymbol{\alpha} \mathbf{A}^2$ when $n=2$ across open month boundaries (Train $\to$ Val Folds 2 & 3; Val $\to$ Test Folds 1 & 2).
2. `test_boundary_policy_a_weekend_reset`: Verifies prior resets strictly to `model.startprob_` when crossing Friday close to Sunday open (Train $\to$ Val Folds 1 & 4; Val $\to$ Test Fold 3).
3. `test_boundary_policy_a_normal_consecutive`: Verifies $\boldsymbol{\alpha}_{\text{prior}} = \boldsymbol{\alpha} \mathbf{A}^1$ when $n=1$ (Train $\to$ Val Fold 5; Val $\to$ Test Folds 4 & 5).
4. `test_boundary_policy_b_fresh_reset`: Verifies independent sequence starts from `model.startprob_` across all windows.

---

## 5. Reassessment of Multi-Year Historical Data Strategy

### 5.1 Codebase Audit Findings & Timestamp Assumption Verification
- **Data Ingestion Agnosticism:** Inspection of `src/data/acquire.py` and `src/data/resample.py` confirms that the data acquisition pipeline is fully parameterized by year (`--year`).
- **Unverified Data-Source Assumption (HistData Fixed UTC-5):**
  * In `src/data/acquire.py` (lines 18–20), the code assumes: `# HistData timestamps are in Eastern Standard Time (EST) all year round (no DST shifts). EST = UTC - 5 hours.`
  * **Critical Audit Finding:** While this is a common convention cited in retail forex forums, **no authoritative vendor specification or data dictionary is stored or linked within this repository**.
  * **Classification:** It is explicitly categorized as an **unverified data-source assumption**.
  * **Risks of DST Mismatch:**
    1. *Session Classification:* If HistData actually shifts to EDT (UTC-4) in summer, defined FX sessions (Asian, London, NY Overlap) will be misaligned by 60 minutes between March and November.
    2. *Gap Handler False Positives:* A 1-hour shift around the March and November DST changeovers could cause Sunday openings (17:00 NY time) to be misclassified as active-session missing bars or irregular intervals.
    3. *Diurnal Smearing:* If true local trading activity shifts by 1 hour while timestamps do not (or vice versa), the 96 diurnal buckets will blur the London/NY liquidity peaks between summer and winter.
  * **Mandatory Pre-Acquisition Verification Step:** Before downloading 2022 or 2024 data, inspect the raw tick timestamps of the 2023 dataset around the March 12, 2023 (US DST start) and November 5, 2023 (US DST end) weekends to empirically confirm whether Sunday FX market opening quotes consistently appear at 17:00 EST / 22:00 UTC or shift to 21:00 UTC.

### 5.2 Status of 2024 Data: Qualified Holdout Statement
- A full search of the repository confirms that no 2024 historical market data exists on disk.
- **Formally Qualified Statement:**
  > "2024 is an intended untouched holdout, subject to confirmation that it has not been inspected or used in prior experiments outside this repository."
- Likewise, **2022–2024 is NOT yet a validated multi-year dataset**; it remains an architectural proposal pending empirical verification.
- **Strict Rule:** **No data for 2022 or 2024 will be downloaded during this design phase.**

---

## 6. Controlled Sensitivity Experiments & Reconstructed Training Budget

### 6.1 Actual Meaning and Scale of Covariance Regularization
Inspection of `src/models/gaussian_hmm.py` and `src/models/gmm_hmm.py` reveals that `cov_reg` is **not an unscaled additive constant**. The implementation scales `cov_reg` by empirical feature variance:
$$\text{cov\_reg\_diag} = \max(\text{Var}(X), 10^{-12}) \times \text{cov\_reg}$$
$$\boldsymbol{\Sigma}_{\text{regularized}} = \boldsymbol{\Sigma}_{\text{empirical}} + \text{diag}(\text{cov\_reg\_diag})$$
`cov_reg` represents a **relative fraction of feature variance**:
- `cov_reg = 1e-4`: Adds $0.01\%$ of empirical variance along the diagonal (Gaussian HMM baseline).
- `cov_reg = 1e-3`: Adds $0.1\%$ of empirical variance along the diagonal (audited GMM-HMM baseline).
- `cov_reg = 1e-2`: Adds $1.0\%$ of empirical variance along the diagonal.

---

### 6.2 Clarification of the Diurnal Feature Experiment
The comparison between:
- Feature Set A: $[r_{\text{oc}}, \log\text{vol}]$
- Feature Set B: $[r_{\text{oc}}, \text{range\_zscore}]$

changes **both the seasonal adjustment AND the distributional scaling/representation of the feature** (logarithmic scale vs. median/MAD standardized z-score).  
Therefore:
- This is explicitly designated as **Feature-Representation / Diurnal-Adjustment Sensitivity**, NOT a pure seasonality ablation.
- `DiurnalVolatilityScaler` is fitted **strictly on `df_train^{(f)}`** of each fold and applied to validation/test without look-ahead.

---

### 6.3 Reconstructed Experiment Matrix & Model Reuse Map

To prevent double-counting and eliminate redundant computation, the training runs are structured with complete model reuse:

```
+---------------------------------------------------------------------------------------------------+
| 1. Primary Architecture Comparison (RQ2)                                                         |
|    - Gaussian HMM K3-M1 (cov_reg=1e-4, S=5, [r_oc, log_vol]): 5 folds                            | -> 5 fits
|    - GMM-HMM K3-M2     (cov_reg=1e-3, S=5, [r_oc, log_vol]): 5 folds                            | -> 5 fits
|    - GMM-HMM K3-M3     (cov_reg=1e-3, S=5, [r_oc, log_vol]): 5 folds                            | -> 5 fits (ANCHOR BASELINE)
|    Subtotal: 15 unique fits                                                                       |
+---------------------------------------------------------------------------------------------------+
                                                  |
                     REUSED AS ANCHOR BASELINE FOR SENSITIVITY EXPERIMENTS
                                                  |
+---------------------------------------------------------------------------------------------------+
| 2. Covariance Regularization Sensitivity (RQ3)                                                    |
|    - GMM-HMM K3-M3 with cov_reg=1e-4: 5 folds                                                     | -> 5 fits
|    - GMM-HMM K3-M3 with cov_reg=1e-2: 5 folds                                                     | -> 5 fits
|    - (cov_reg=1e-3 is REUSED from Anchor Baseline Model 3: 0 new fits)                            |
|    Subtotal: 10 unique fits                                                                       |
+---------------------------------------------------------------------------------------------------+
                                                  |
+---------------------------------------------------------------------------------------------------+
| 3. Feature-Representation / Diurnal Sensitivity (RQ3)                                             |
|    - GMM-HMM K3-M3 with [r_oc, range_zscore] (cov_reg=1e-3, S=5): 5 folds                         | -> 5 fits
|    - ([r_oc, log_vol] is REUSED from Anchor Baseline Model 3: 0 new fits)                         |
|    Subtotal: 5 unique fits                                                                        |
+---------------------------------------------------------------------------------------------------+
                                                  |
+---------------------------------------------------------------------------------------------------+
| 4. EM Restart Sensitivity (RQ3) — Codebase Audit & Methodological Clarification                   |
|    - Codebase & Artifact Audit Findings:                                                          |
|      1. `src/models/gaussian_hmm.py` (lines 356-385) and `src/models/gmm_hmm.py` (lines 412-426) |
|         only retain the single winning model parameters in memory (`best_log_likelihood`).       |
|         Intermediate non-winning restart candidate scores and random seeds are overwritten and    |
|         discarded; neither model serializes intermediate restart history to JSON.                 |
|      2. The earlier Phase 4 audit benchmark (`scripts/audit_phase4.py`, reported in               |
|         `PHASE4_GMM_HMM_REPORT.md` Section 6.1 and `phase4_scientific_audit_results.json`)        |
|         evaluated 5 independent seeds with SINGLE-RESTART fits (N_restarts = 1) without multi-    |
|         sequence segmentation, reporting validation predictive LL standard deviations:           |
|         Gaussian HMM (sigma = 0.3 nats), GMM K3-M2 (sigma = 2.8 nats), GMM K3-M3 (sigma = 79.7). |
|      3. The previous draft statement claiming "Phase 4 demonstrated that Baum-Welch EM with       |
|         S >= 5 random restarts consistently converges to within numerical tolerance               |
|         (< 0.05 nats dispersion)" was an unsupported assertion and is formally RETRACTED.        |
|         No multi-seed benchmark evaluating best-of-5 restart stability was executed in Phase 4.   |
|    - Operational Decision for Phase 7:                                                            |
|      * Standardize all Phase 7 fits on S=5 random restarts with fixed base seed `random_state=42`.|
|      * Clarification: S=5 is an operational baseline to dampen initialization variance across     |
|        folds and establish a uniform computational ceiling, NOT an empirically proven optimal     |
|        restart count.                                                                             |
|      * Explicit Scope Disclosure: Systematic restart-count sensitivity (evaluating S in {1,3,5,10})|
|        remains an UNTESTED dimension in Phase 7 to prevent computational budget inflation          |
|        (avoiding 10 additional model training fits).                                              |
|    Subtotal: 0 unique fits                                                                        |
+---------------------------------------------------------------------------------------------------+
```

#### Exact Total:
$$\text{Total Unique Model Training Fits} = 15 + 10 + 5 + 0 = \mathbf{30 \text{ unique fits}}$$
$$\text{Total Evaluated Configurations} = 30 \text{ fits} + 10 \text{ reused fits} = \mathbf{40 \text{ evaluations}}$$

---

### 6.4 Evidence-Based Runtime Estimate (Benchmarked on Representative Fold 1)

#### Execution Hardware & Software Environment:
- **Operating System:** Windows 11 Home Single Language (AMD64)
- **Processor:** AMD Ryzen (16 logical cores)
- **Python Runtime:** Python 3.14.3
- **Core Scientific Libraries:** NumPy 2.4.4, SciPy 1.17.1, Pandas 3.0.0

#### Single-Fit Benchmarks on Fold 1 (9,472 complete bars, 692 segments):
- `GaussianHMM` (1 fit, $S=1$ restart): 3 repeated runs yielded **10.67s** (56 iters), **8.85s** (54 iters), **7.98s** (54 iters) $\implies$ **Mean: 9.17s $\pm$ 1.12s**
- `GMM-HMM K3-M2` (1 fit, $S=1$ restart): **19.25s** (58 iters)
- `GMM-HMM K3-M3` (1 fit, $S=1$ restart): **16.64s** (44 iters)

#### Projected Runtime Calculation:
With $S=5$ random restarts, accounting for sample size expansion across folds (Fold 1 has 9,472 bars; Fold 5 has 16,949 bars; average is ~13,000 bars $\approx 1.37\times$ Fold 1):
- 5 Gaussian HMM fits: $5 \times (9.17\text{s} \times 5 \times 1.37) \approx 314\text{s} \approx \mathbf{5.2\text{ minutes}}$
- 5 GMM-HMM K3-M2 fits: $5 \times (19.25\text{s} \times 5 \times 1.37) \approx 659\text{s} \approx \mathbf{11.0\text{ minutes}}$
- 5 GMM-HMM K3-M3 anchor fits: $5 \times (16.64\text{s} \times 5 \times 1.37) \approx 570\text{s} \approx \mathbf{9.5\text{ minutes}}$
- 10 GMM-HMM K3-M3 regularization fits: $10 \times (16.64\text{s} \times 5 \times 1.37) \approx 1,140\text{s} \approx \mathbf{19.0\text{ minutes}}$
- 5 GMM-HMM K3-M3 diurnal feature fits: $5 \times (16.64\text{s} \times 5 \times 1.37) \approx 570\text{s} \approx \mathbf{9.5\text{ minutes}}$

$$\mathbf{\text{Total Projected Evidence-Based Runtime: } \approx 3,253\text{ seconds} \approx 54\text{ minutes}}$$
*(Projected at approximately 50 to 60 minutes on a single CPU core. Note: This is an evidence-based projection subject to EM iteration counts and machine load, not a fixed constant).*

---

### 6.5 Deterministic Random Seed Schedule & Reproducibility Protocol

To ensure 100% bitwise reproducibility and eliminate arbitrary initialization drift across folds, all model training adheres to a deterministic seed specification:

#### 1. Assigned Random Seed Schedule:
- **Base Random State:** `random_state = 42` is assigned to all 30 model fits.
- **Cross-Fold Seed Schedule Reuse:** The identical base seed schedule is reused across all 5 folds and across all model architectures.
  * *Methodological Rationale:* Each fold $f$ trains on a distinct, expanded historical observation dataset $\mathcal{D}_{\text{train}}^{(f)}$ ($N \in \{9472, 10886, 13006, 14893, 16949\}$ bars). Reusing the exact same initialization RNG schedule isolates parameter divergence $(\mathbf{A}, \boldsymbol{\mu}, \boldsymbol{\Sigma})$ strictly to changing empirical market data, rather than introducing confounding seed perturbations across chronological windows.
- **Per-Restart Seed Generation (Codebase Formula):**
  Within each fit, the $S=5$ restarts are seeded deterministically according to the existing model implementations:
  * `GaussianHMM`: $\text{seed}_r = \text{base\_seed} + r \times 101 \implies [42, 143, 244, 345, 446]$ for $r \in \{0, 1, 2, 3, 4\}$.
  * `GMMHMM`: $\text{seed}_r = \text{base\_seed} + r \times 107 \implies [42, 149, 256, 363, 470]$ for $r \in \{0, 1, 2, 3, 4\}$.

#### 2. Mandatory Reproducibility Metadata Recorded per Fit:
For every fitted model artifact, `scripts/run_phase7_tier1.py` must serialize:
- `fold_id`: Fold identifier (1 to 5).
- `model_name`: Architecture string (e.g. `Gaussian_HMM_K3_M1`, `GMM_HMM_K3_M2`, `GMM_HMM_K3_M3`).
- `base_random_state`: Integer base seed (`42`).
- `n_restarts`: Number of restarts executed (`5`).
- `restart_seed_schedule`: Explicit list of seeds evaluated.
- `parameter_sha256`: SHA-256 parameter hash computed via `ModelArtifact.compute_parameter_hash`.
- `best_log_likelihood`: Final in-sample log-likelihood of the selected restart.
- `n_iter_done`: Number of EM iterations completed by the winning restart.
- `converged`: Boolean EM convergence flag.
- `training_time_seconds`: Wall-clock execution time in seconds.

---

## 7. Refined Statistical Evaluation & Diebold-Mariano Protocol

### 7.1 Primary Metric: Per-Bar Out-of-Sample Predictive Log-Likelihood

For each test sequence $O_{1:T_{\text{test}}}$, the causal one-step-ahead predictive density is:
$$\bar{\ell}_{\text{oos}} = \frac{1}{T_{\text{test}}} \sum_{t=1}^{T_{\text{test}}} \log P(O_t \mid O_{1:t-1})$$
$$P(O_t \mid O_{1:t-1}) = \sum_{j=1}^K \left( \sum_{i=1}^K \alpha_{t-1}(i) A_{ij} \right) b_j(O_t)$$

This is strictly causal: it evaluates the likelihood of the incoming observation $O_t$ conditional strictly on past observations $O_{1:t-1}$ and frozen model parameters.

### 7.2 Reporting Fold Variability Without Claiming Independent Samples

Because expanding-window folds have nested, overlapping training periods, cross-fold metrics **cannot be treated as independent and identically distributed (i.i.d.) observations**.

Therefore:
- Report fold-level performance as a **chronological trajectory**: $\bar{\ell}_{\text{oos}}^{(1)}, \bar{\ell}_{\text{oos}}^{(2)}, \dots, \bar{\ell}_{\text{oos}}^{(5)}$.
- Report cross-fold summary statistics (mean, median, min, max) strictly as **descriptive performance profiles across historical periods**.
- Do not compute naive $t$-tests across fold averages, which would violate independence assumptions.

### 7.3 Within-Fold Sequential Score Comparison (Diebold-Mariano with HAC Lag Sensitivity)

To evaluate whether the predictive score difference between Model A and Model B is statistically distinguishable within an individual test fold ($T_{\text{test}} \approx 1,860$ to $2,120$ bars):
$$d_t = \log P_{M_A}(O_t \mid O_{1:t-1}) - \log P_{M_B}(O_t \mid O_{1:t-1})$$
$$\bar{d} = \frac{1}{T_{\text{test}}} \sum_{t=1}^{T_{\text{test}}} d_t$$

#### 1. Test Direction and Null Hypothesis:
- **Loss Differential Definition:** $d_t$ is defined as the log-likelihood differential where higher values indicate superior one-step-ahead predictive density.
- **Null Hypothesis ($H_0$):** $E[d_t] = 0$ (Model A and Model B have identical predictive density accuracy).
- **Alternative Hypothesis ($H_1$):**
  * Two-sided: $E[d_t] \neq 0$ (predictive accuracy differs significantly).
  * Directional Interpretation: A positive sample mean $\bar{d} > 0$ and $\text{DM}_{\text{HLN}} > 0$ indicates Model A outperforms Model B. A negative value indicates Model B outperforms Model A.
- **Decision Rule:** For a two-tailed significance level $\alpha = 0.05$, reject $H_0$ if $|\text{DM}_{\text{HLN}}| > 1.96$ ($p < 0.05$).

#### 2. Exact Newey-West HAC Lag Convention & Estimator:
The long-run variance of the loss differential $\hat{\sigma}_{\text{HAC}}^2$ is estimated using the Newey-West (1987) heteroskedasticity and autocorrelation consistent (HAC) estimator with a triangular Bartlett kernel:
$$\hat{\sigma}_{\text{HAC}}^2 = \hat{\gamma}_0 + 2 \sum_{l=1}^L \left( 1 - \frac{l}{L + 1} \right) \hat{\gamma}_l$$
where the sample autocovariance at lag $l$ is:
$$\hat{\gamma}_l = \frac{1}{T_{\text{test}}} \sum_{t=l+1}^{T_{\text{test}}} (d_t - \bar{d})(d_{t-l} - \bar{d})$$
- **Triangular Bartlett Kernel Weight:** $w_l = 1 - \frac{l}{L + 1}$ guarantees positive semi-definiteness ($\hat{\sigma}_{\text{HAC}}^2 > 0$).
- **Predefined Truncation Lag Grid ($L$):** To prevent post-hoc lag shopping, results are reported across three fixed horizons:
  * $L = 4$ lags (1 hour of M15 bars, captures immediate high-frequency microstructure persistence).
  * $L = 8$ lags (2 hours, calibrated to mean regime duration $\approx 7\text{--}9$ bars).
  * $L = 16$ lags (4 hours, half a major trading session).

#### 3. Harvey-Leybourne-Newbold (HLN 1997) Finite-Sample Correction:
For one-step-ahead ($h=1$) forecasts on sample size $T_{\text{test}}$, the HLN-adjusted statistic is:
$$\text{DM}_{\text{HLN}} = \frac{\bar{d}}{\sqrt{\hat{\sigma}_{\text{HAC}}^2 / T_{\text{test}}}} \times \sqrt{\frac{T_{\text{test}} - 1}{T_{\text{test}}}}$$
Under $H_0$, $\text{DM}_{\text{HLN}}$ is asymptotically distributed as standard normal $\mathcal{N}(0, 1)$ for large $T_{\text{test}} \approx 2,000$.

#### 4. Multiple Model Comparisons Handling (Holm-Bonferroni FWER Adjustment):
With 3 primary architectures, each test fold entails $\binom{3}{2} = 3$ pairwise comparisons:
1. GMM-HMM K3-M3 vs. Gaussian HMM K3-M1
2. GMM-HMM K3-M2 vs. Gaussian HMM K3-M1
3. GMM-HMM K3-M3 vs. GMM-HMM K3-M2

To control the Family-Wise Error Rate (FWER) across the $m=3$ simultaneous hypothesis tests within each fold:
- Sort raw two-sided $p$-values in ascending order: $p_{(1)} \le p_{(2)} \le p_{(3)}$.
- Compute Holm-Bonferroni step-down adjusted $p$-values:
  $$p_{\text{adj}(i)} = \min\left(1, \max_{j \le i} (m - j + 1) p_{(j)}\right)$$
- Report both raw and Holm-adjusted $p$-values in the final walk-forward diagnostic tables.

#### 5. Multi-Criteria Model Selection Protocol (Never p-values alone):
Model selection will **NEVER depend solely on statistical significance or $p$-values**. DM test statistics provide local evidence of loss differential distinction within individual windows, but financial regime models must be arbitrated holistically. A candidate model is declared superior only if it satisfies all four criteria:
1. **Predictive Density Supremacy:** Consistently higher per-bar predictive log-likelihood ($\bar{\ell}_{\text{oos}}^{(f)}$) across all or most historical folds.
2. **Topological Monotonicity:** Strict volatility ordering under Hungarian bipartite alignment:
   $$\sigma(r \mid \text{LowVol}) < \sigma(r \mid \text{MedVol}) < \sigma(r \mid \text{HighVol})$$
3. **State Health & Non-Collapse:** Balanced state occupancy across all aligned regimes ($\min_k \text{occupancy}_k \ge 5\%$).
4. **Complexity Parsimony:** If GMM K3-M3 ($p=59$) achieves only marginally higher predictive density than GMM K3-M2 ($p=41$) without statistically distinguishable or economically meaningful gains, the more parsimonious model is favored.

---

## 8. Preserved Methodological Safeguards

The following core safeguards are strictly preserved without modification:
1. **Strict Chronological Splits:** All folds strictly respect arrow of time (no shuffling, no k-fold cross-validation).
2. **Partial-Candle Exclusion:** All 796 incomplete M15 candles are completely filtered out.
3. **Multi-Sequence Boundary Reset:** Continuous sequences segmented by `identify_continuous_sequences()` reset filter propagation across market closures.
4. **Train-Only State Alignment:** Hungarian profile mapping calculated strictly on in-sample smoothed training characteristics.
5. **Frozen Parameters:** No online parameter updates during validation or test evaluations.
6. **Immutable Model Fingerprinting:** All models saved with SHA-256 parameter hashes.

---

## 9. Recommended Implementation Sequence

Upon user approval of this experimental design:
1. **Module Implementation:** Create `src/evaluation/walk_forward.py` implementing:
   - Chronological fold partitioning with complete-bar filtering.
   - Policy A (continuous propagation) and Policy B (independent diagnostic) boundary handling.
   - Out-of-sample forward filter evaluation loop.
   - Diebold-Mariano test with Newey-West HAC lags (4, 8, 16) and HLN correction.
2. **Unit Testing:** Create `tests/test_walk_forward.py` verifying fold indexing, non-overlap, boundary propagation ($n=1, 2, 193$), and zero data leakage.
3. **Tier 1 Benchmark Execution:** Create `scripts/run_phase7_tier1.py` and execute the 30 unique fits across 2023 data.
4. **Research Report & Visualization:** Generate Figure 19 (OOS predictive likelihood trajectories) and Figure 20 (Regime parameter drift heatmaps), compiling `data/reports/PHASE7_WALKFORWARD_REPORT.md`.
5. **Tier 2 Multi-Year Verification & Acquisition (Optional):** Pre-verify HistData 2023 DST transition timestamps before acquiring 2022 and 2024 data.

---

*Methodologically audited and refined by Google DeepMind Advanced Agentic Coding System.*  
*Awaiting explicit user authorization prior to implementation.*
