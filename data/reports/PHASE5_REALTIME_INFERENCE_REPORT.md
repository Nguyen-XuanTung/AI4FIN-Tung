# Phase 5: Real-Time Inference, Model Versioning, and Retraining Management Report

**Project:** EURUSD M15 Market Regime Detection Using Hidden Markov Models  
**Date:** 2026-11-03 15:22:11 UTC  
**Author / Model Engine:** Google DeepMind Advanced Agentic Coding Pair  
**Primary Baseline Model:** `Gaussian_HMM_K3_M1_v1`  
**Deployed Champion Model:** `GMM_HMM_K3_M3_v1`  

---

## 1. Executive Summary & Audit Resolution
Phase 5 operationalizes the EURUSD M15 regime detection framework into a causal, real-time sequential inference engine. The system estimates filtered latent regime probabilities:
$$P(S_t = j \mid O_{1:t})$$
one completed M15 candle at a time, strictly preventing future information leakage ($O_{t+1:T}$) and enforcing data integrity across market closures, unobserved gaps, model cutover transitions, and incomplete bars.

### 1.1 Methodological Distinction: Parameter Lookahead, Causal Replay, and Real-Time Deployment
A rigorous scientific audit of the sequential inference pipeline establishes clear distinctions between causal filtering, parameter estimation boundaries, and physical execution semantics:
1. **In-Sample Filtered Probabilities (`in_sample_filtered`):**
   - Applied to the 12,501 complete candles of the training partition (2023-01-01 to 2023-08-24 11:45:00 UTC).
   - Although forward filtering processes sequentially ($P(S_t \mid O_{1:t}, \Theta)$), model parameters $\Theta$ were estimated over the full training period via the Baum-Welch algorithm.
   - Consequently, $\Theta$ embeds future summary statistics relative to intraday timestamps $t < T_{\text{train}}$. This constitutes **model parameter lookahead**.
   - These records are strictly categorized as `inference_type='in_sample_filtered'` and are never represented as live `as_of` predictions.
2. **Historical Causal Replay (`as_of_replay`):**
   - Applied to the 8,335 complete candles of the out-of-sample validation and test partitions (2023-08-24 12:00:00 UTC to 2023-12-29 21:45:00 UTC).
   - Model parameters $\Theta_{\text{GMM}}$ were estimated strictly on training data ending at 2023-08-24 11:45:00 UTC and frozen prior to the cutover.
   - Zero parameter lookahead: $\Theta$ was finalized before any out-of-sample observation arrived.
   - Zero observation lookahead: sequential recursive updates receive strictly one completed M15 candle at a time.
   - Categorized as `inference_type='as_of_replay'` to distinguish historical simulation from live production execution at 2023 wall-clock time.
3. **Genuine Live As-Of Predictions (`as_of`):**
   - Reserved exclusively for operational live production deployment where models are frozen before live market ticks occur and inference executes concurrently with live candle closes.
4. **Skipped Partial-Candle Audit Records (`skipped_audit`):**
   - Exactly 796 incomplete M15 bars (419 pre-cutover, 377 post-cutover) where emission likelihood calculation is bypassed to prevent structural distortion from compressed candle ranges.
5. **Retrospective Evaluation Stream (`retrospective`):**
   - Offline counterfactual or sensitivity runs (50 records) executed alongside the primary stream without index or file collisions.

### 1.2 Resolution of the Five Key Audit Questions
A thorough code-path and data-integrity audit investigated whether incomplete partial candles (`is_complete == False`) contaminated filtering updates:
1. **Were partial bars passed to `RealTimeInferenceEngine.step()`?**  
   *Audit Finding:* In initial streaming runs, raw data rows were fed sequentially without checking `is_complete`.  
   *Resolution:* Implemented explicit `is_complete` argument in `RealTimeInferenceEngine.step()`. All 796 partial bars are recognized and routed through a strict skipping path.
2. **Were emission likelihoods calculated for partial bars?**  
   *Audit Finding:* In the unconstrained script, `compute_single_log_emission(x_arr)` was invoked on incomplete candles with mechanically compressed returns and ranges.  
   *Resolution:* Emission computation is completely bypassed when `is_complete is False`. Zero emission likelihood calculations occur for partial bars.
3. **Were partial-bar probabilities written as normal predictions?**  
   *Audit Finding:* Partial bars were previously stored in the valid prediction stream.  
   *Resolution:* Segregated the streams. Complete candles are stored under their respective causal inference types (`in_sample_filtered` or `as_of_replay`). The 796 partial candles are logged with `inference_type='skipped_audit'` (`status='skipped_partial'`, `is_valid_prediction=False`).
4. **Did the engine update its posterior using partial-bar observations?**  
   *Audit Finding:* Bayes measurement updates previously altered the internal belief state using compressed partial candle features.  
   *Resolution:* The engine's belief state $\boldsymbol{\alpha}_t$ and `last_timestamp` are strictly preserved at the last valid complete observation. When the next complete candle arrives, Markov transitions $\mathbf{A}^n$ propagate across the unobserved interval.
5. **Were partial bars included in warm-up buffers?**  
   *Audit Finding:* Warm-up buffers previously sliced raw trailing rows from `df`.  
   *Resolution:* Warm-up buffers strictly filter to complete bars only (`df[df['is_complete'] == True]`). Exactly 50 complete M15 candles precede the cutover point.

### 1.3 Comprehensive Record Count Reconciliation
| Dataset Partition / Category | Raw Rows | Complete Candles (`is_complete=True`) | Partial Candles (`is_complete=False`) | Causal Inference Type | Active Model Version | Parameter Lookahead? |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **Pre-Cutover (Training Partition)** | 12,920 | 12,501 | 419 | `in_sample_filtered` (12,501)<br>`skipped_audit` (419) | `Gaussian_HMM_K3_M1_v1` | **Yes** (fitted on full train split) |
| **Post-Cutover (Validation + Test)** | 8,712 | 8,335 | 377 | `as_of_replay` (8,335)<br>`skipped_audit` (377) | `GMM_HMM_K3_M3_v1` | **No** (frozen prior to cutover) |
| **Full 2023 Sequence Total** | **21,632** | **20,836** | **796** | — | — | — |
| **Retrospective Coexistence Audit** | — | — | — | `retrospective` (50) | `GMM_HMM_K3_M3_v1` | Offline audit |
| **Total JSONL Log Entries** | — | — | — | **21,682** | — | — |

### Columnar Parquet Exports (Strict Partitioning):
- `data/processed/in_sample_inference_log.parquet`: Exactly **12,501 rows** (100% complete candles, in-sample causal filtering conditional on training parameters).
- `data/processed/as_of_replay_inference_log.parquet`: Exactly **8,335 rows** (100% complete candles, out-of-sample historical causal replay with frozen model).
- `data/processed/as_of_inference_log.parquet`: Exactly **8,335 rows** (legacy alias for out-of-sample replay stream).
- `data/processed/skipped_audit_log.parquet`: Exactly **796 rows** (100% partial bars skipped from emission update, preserving raw OHLC metadata without fabrication).
- `data/processed/full_candle_audit_log.parquet`: Exactly **21,632 rows** (chronological stream of all 2023 bars, 20,836 valid + 796 skipped, matching source dataset 1-to-1).
- `data/processed/retrospective_inference_log.parquet`: Exactly **50 rows** (isolated offline evaluation stream).

### Performance: Mean Latency 156.6 us (6,384 bars/sec), Median 141.2 us, p99 327.9 us.

---

## 2. Mathematical Formulations & Sequential Recursive Filtering
### 2.1 Forward Filter Recursion
For an incoming observation $\mathbf{x}_t = [r_{OC, t}, \log\text{vol}_t]^T$, the causal filtered state probability is updated recursively from the previous posterior $\boldsymbol{\alpha}_{t-1}$:
1. **Time Propagation (Prior):**
   $$\boldsymbol{\alpha}_{t\mid t-1} = \boldsymbol{\alpha}_{t-1} \mathbf{A}$$
2. **Predictive Likelihood:**
   $$P(\mathbf{x}_t \mid O_{1:t-1}) = \sum_{j=1}^K \alpha_{t\mid t-1}(j) \, b_j(\mathbf{x}_t)$$
3. **Bayes Measurement Update (Posterior):**
   $$\alpha_t(j) = \frac{\alpha_{t\mid t-1}(j) \, b_j(\mathbf{x}_t)}{\sum_{k=1}^K \alpha_{t\mid t-1}(k) \, b_k(\mathbf{x}_t)}$$
All computations are executed in log-space using stable `logsumexp` operations to prevent numerical underflow.

### 2.2 Emission Densities: Gaussian vs. GMM-HMM
- **Gaussian HMM (M=1):**
  $$b_j(\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_t \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)$$
- **GMM-HMM (M components):**
  $$b_j(\mathbf{x}_t) = \sum_{m=1}^M c_{jm} \, \mathcal{N}(\mathbf{x}_t \mid \boldsymbol{\mu}_{jm}, \boldsymbol{\Sigma}_{jm}), \quad \sum_{m=1}^M c_{jm} = 1$$

---

## 3. FX Trading Calendar & Data Gap Policy
Unlike continuous synthetic time series, EURUSD M15 data contains real-world gaps caused by market closures, holidays, broker rollover pauses, and intermittent data dropouts. Classifying gaps solely by duration produces severe state distortion (e.g. treating a 48-hour weekend closure identically to a 48-hour active outage).

### 3.1 HistData Source Convention & Timezone Standardization
- **Source Timezone:** HistData ASCII files are timestamped in Eastern Standard Time (EST) all year round without Daylight Saving Time shifts (fixed UTC-5).
- **Conversion:** Adding 5 hours produces true continuous UTC timestamps without artificial 1-hour forward or backward leaps.
- **Summer/Winter Trading Hours:** In UTC, the New York Friday close occurs at 21:00 UTC during US DST and 22:00 UTC during US standard time. Sunday opening occurs at 21:00 UTC (DST) or 22:00 UTC (standard). The calendar handler accommodates both transitions cleanly without ambiguity.

### 3.2 Partial-Bar Skipping & Transition Propagation
Incomplete M15 candles have mechanically compressed returns and ranges because trading terminates before the full 15-minute interval completes. In Phases 2 and 3, incomplete bars were excluded from primary HMM training and evaluation to prevent structural distortion. In Phase 5 real-time sequential inference, this principle is strictly enforced:
1. **Skipping Incomplete Bars:** When a bar has `is_complete == False`, no emission likelihood is computed, no measurement update occurs, and the belief state $\boldsymbol{\alpha}$ is preserved at the last valid complete observation.
2. **Propagating Across Skipped Intervals:** When the next complete candle arrives at $t_{\text{curr}}$, the time delta $\Delta t = t_{\text{curr}} - t_{\text{last\_complete}}$ is evaluated:
   - If the interval is within open market hours ($m$ unobserved / skipped bars), exactly $n = m + 1 = \Delta t / 15\text{ min}$ Markov transitions occur:
     $$\boldsymbol{\alpha}_{\text{prior}} = \boldsymbol{\alpha}_{\text{last\_complete}} \mathbf{A}^n$$
   - If the interval crosses a weekend closure or holiday, intraday memory decays and prior resets to the initial-state distribution $\boldsymbol{\pi}$ (`model.startprob_`).
   - The Bayes measurement update is then computed using only the valid complete candle observation.

### 3.3 Empirical Gap Taxonomy (Consecutive Complete Bars)
| Gap Classification | Trigger Condition | Markov Transition Steps $n$ | Empirical Count (2023) | State Propagation Policy |
| :--- | :--- | :---: | :---: | :--- |
| **Normal Bar** | $\Delta t = 15$ min | $n = 1$ | 19,686 | Standard 1-step propagation: $\boldsymbol{\alpha}_{\text{prior}} = \boldsymbol{\alpha}_{t-1} \mathbf{A}$. |
| **Active Missing Bars** | $\Delta t \in [30, 300]$ min during open market | $n = \frac{\Delta t}{15} = \text{missing\_bars} + 1$ | 1,093 | Exponentiated transition matrix: $\boldsymbol{\alpha}_{\text{prior}} = \boldsymbol{\alpha}_{t-1} \mathbf{A}^n$. Bayes emission update applied on arrival. |
| **Market Closure** | Friday close $\rightarrow$ Sunday open, Christmas, New Year | $n = \text{unspecified}$ | 56 | Intraday volatility memory decays. Prior resets to initial-state distribution $\boldsymbol{\pi}$ (`model.startprob_`). |
| **Uncertain / Irregular** | Non-monotonic, non-multiples of 15 min, or gap $> 300$ min | $n = 0$ | 0 | Strict mode raises `UncertainGapError`. All 20,836 complete bars classified cleanly. |

### 3.4 Proof of Active-Session Missing Bar Propagation
If $m$ intermediate bars are unobserved between $t$ and $t+m+1$, the marginal transition probability over the hidden Markov chain is:
$$P(S_{t+m+1} = j \mid S_t = i) = [\mathbf{A}^{m+1}]_{ij}$$
Hence, with $m = \text{missing\_bars}$, exactly $n = m+1$ transitions occur before the arrival of the next observation $\mathbf{x}_{t+m+1}$:
$$\boldsymbol{\alpha}_{\text{prior}} = \boldsymbol{\alpha}_t \mathbf{A}^{m+1}$$
Because no measurement arrived during the gap, no intermediate emission update is computed. As $n$ increases, $\mathbf{A}^n$ naturally diffuses toward the asymptotic stationary distribution $\boldsymbol{\pi}_{\infty}$ satisfying $\boldsymbol{\pi}_{\infty} \mathbf{A} = \boldsymbol{\pi}_{\infty}$, reflecting increasing uncertainty. We emphasize that the initial-state distribution $\boldsymbol{\pi}$ (`model.startprob_`) is generally distinct from the asymptotic stationary distribution $\boldsymbol{\pi}_{\infty}$ (i.e. $\boldsymbol{\pi} \mathbf{A} \ne \boldsymbol{\pi}$).

---

## 4. Retraining Management & Warm-Up Buffer Sensitivity
### 4.1 The Two-Tiered Validation Architecture
Candidate models submitted for deployment must pass strict automated validation:
- **Tier 1: Hard Validation Gates (Block Deployment)**
  1. Parameter finiteness: $\boldsymbol{\pi}, \mathbf{A}, \mathbf{c}, \boldsymbol{\mu}, \boldsymbol{\Sigma}$ contain no NaNs or Infs.
  2. Probability normalization: $\sum_j \pi_j = 1$, $\sum_k A_{jk} = 1$, $\sum_m c_{jm} = 1$, all values in $[0, 1]$.
  3. Positive-definite covariance: all eigenvalues $\lambda_k \ge 10^{-9}$, condition numbers $\kappa < 10^8$.
  4. Feature compatibility: exact match of feature names (`['r_oc', 'log_vol']`) and feature version (`Config_2_MinimalOrthogonal_v1`).
  5. Serialization round-trip: byte/checksum invariance after JSON serialization.
- **Tier 2: Diagnostic Warnings (Informative Only)**
  * Minimum soft state occupancy: flagged if $< 5\%$.
  * Argmax switching frequency: flagged if $> 35\%$.
  * Covariance conditioning: flagged if condition number $> 10^7$.

### 4.2 Multi-Window Warm-Up Buffer Sensitivity Sweep
When migrating active live inference from $\Theta_{\text{old}}$ to $\Theta_{\text{new}}$, the new model cannot reuse the posterior vector of the old model because the latent state space is model-specific. To initialize the new filter, a contiguous buffer of $W$ recent historical observations is filtered exclusively through $\Theta_{\text{new}}$, starting from its initial distribution $\boldsymbol{\pi}_{\text{new}}$.

To determine the required buffer length $W^*$, we evaluated the $L_1$ total variation distance against an asymptotic reference of 500 bars across **5 representative market regimes**:
$$\Delta_{L_1}(W) = \frac{1}{2} \sum_{j=1}^K |\alpha_W(j) - \alpha_{\infty}(j)|$$

| Market Window / Regime | Target Timestamp (UTC) | $W^* (L_1 < 10^{-4})$ | $W^* (L_1 < 10^{-6})$ | $L_1$ at $W=50$ |
| :--- | :---: | :---: | :---: | :---: |
| **Train/Val Boundary (2023-08-24 12:00 UTC)** | `2023-08-24 11:45:00+00:00` | 5 bars | 10 bars | 0.00e+00 |
| **Low-Volatility Quiet Session (2023-02-14 00:00 UTC)** | `2023-02-14 04:45:00+00:00` | 5 bars | 10 bars | 0.00e+00 |
| **High-Volatility Turbulence (2023-03-24 14:00 UTC)** | `2023-03-24 01:45:00+00:00` | 5 bars | 10 bars | 0.00e+00 |
| **Post-Weekend Gap Opening (2023-05-15 00:00 UTC)** | `2023-05-16 03:45:00+00:00` | 5 bars | 10 bars | 0.00e+00 |
| **Year-End Trending / Liquidity Fade (2023-12-06 12:00 UTC)** | `2023-12-06 06:30:00+00:00` | 5 bars | 5 bars | 0.00e+00 |

### 4.3 Detailed Sensitivity Curve (Train/Val Cutover Window)
| Buffer Length $W$ | GMM-HMM (K=3, M=3) $L_1$ Error | Gaussian HMM (K=3, M=1) $L_1$ Error | Argmax State Agreement |
| :---: | :---: | :---: | :---: |
|   5 |   6.10e-05 |   7.27e-05 | Match |
|  10 |   3.23e-09 |   1.59e-08 | Match |
|  15 |   8.88e-12 |   3.75e-12 | Match |
|  20 |   0.00e+00 |   1.26e-17 | Match |
|  25 |   0.00e+00 |   0.00e+00 | Match |
|  30 |   0.00e+00 |   0.00e+00 | Match |
|  35 |   0.00e+00 |   0.00e+00 | Match |
|  40 |   0.00e+00 |   0.00e+00 | Match |
|  50 |   0.00e+00 |   0.00e+00 | Match |
|  60 |   0.00e+00 |   0.00e+00 | Match |
|  75 |   0.00e+00 |   0.00e+00 | Match |
| 100 |   0.00e+00 |   0.00e+00 | Match |
| 150 |   0.00e+00 |   0.00e+00 | Match |
| 200 |   0.00e+00 |   0.00e+00 | Match |

> [!IMPORTANT] Empirical Sensitivity vs. Universal Mathematical Guarantees  
> In hidden Markov models, contraction of the filter's initial distribution is governed by the Dobrushin contraction coefficient of the transition matrix $\mathbf{A}$, while observation likelihoods accelerate contraction along specific sample paths. The empirical finding that $W=50$ bars yields $L_1 < 10^{-6}$ across all evaluated empirical windows on EURUSD M15 2023 is an **empirical property of the fitted models**, not an unconditional universal mathematical guarantee across all possible degenerate observation regimes or arbitrary transition matrices. A conservative buffer of $W=50$ bars (12.5 trading hours) is adopted as the operational heuristic.

---

## 5. Model Cutover Alignment & Chronological Integrity
The simulation streamed **21,632 bars** across the entire 2023 EURUSD M15 sequence.
- **Baseline Model (`Gaussian_HMM_K3_M1_v1`):** Active for bars 0 to 12,919 (2023-08-24 11:45:00 UTC).
- **Model Upgrade (`GMM_HMM_K3_M3_v1`):** Deployed at bar **12,920 (2023-08-24 12:00:00 UTC)**.
- **Champion Model (`GMM_HMM_K3_M3_v1`):** Active for bars 12,920 to 21,631 (8,712 bars).

### Reconciliation of Split Indexing:
- In Phase 3, the training split was defined as 12,501 bars on the **complete-candle dataset** (`filter_incomplete=True`).
- In the raw unfiltered sequence (21,632 bars), exactly 419 partial candles preceded the boundary. Hence, the 12,501st complete candle is located at row index **12,919 (2023-08-24 11:45:00 UTC)**.
- The first out-of-sample bar is row index **12,920 (2023-08-24 12:00:00 UTC)**, matching the exact start of the Validation partition.
- Because the champion model was fitted on data terminating at 2023-08-24 11:45:00 UTC, and the warm-up buffer strictly draws from observations preceding 2023-08-24 12:00:00 UTC, **the deployed model uses zero future information and has zero overlap between deployment observations and training observations**.

### 5.1 Single-Bar Latency Benchmark Profile
| Metric | Latency (Microseconds, $\mu\text{s}$) | Fraction of 15-Min Bar ($900\text{s}$) |
| :--- | :---: | :---: |
| **Mean Latency** | 156.63 $\mu\text{s}$ | $0.0000078\%$ |
| **Median Latency (p50)** | 141.25 $\mu\text{s}$ | $0.0000070\%$ |
| **90th Percentile (p90)** | 226.20 $\mu\text{s}$ | $0.0000096\%$ |
| **95th Percentile (p95)** | 256.73 $\mu\text{s}$ | $0.0000111\%$ |
| **99th Percentile (p99)** | 327.87 $\mu\text{s}$ | $0.0000164\%$ |
| **Maximum Latency** | 1243.20 $\mu\text{s}$ | $0.0003889\%$ |
| **Average Throughput** | **6,384 bars/sec** | — |

---

## 6. Immutable Storage Architecture & Data Schema
### 6.1 Immutability & Provenance Guarantees
Real-time and causal regime predictions are historical observations that cannot be altered or overwritten. The storage engine enforces:
1. **Append-Only JSON Lines Log (`data/processed/as_of_inference_log.jsonl`):** Each completed bar is written as an immutable JSON record.
2. **Deterministic Record Hashing:** Record ID is computed via SHA256 over `(timestamp_utc, model_version, inference_type, run_id)`.
3. **Duplicate Rejection Guardrails:** Attempting to log duplicate timestamps within `in_sample_filtered`, `as_of_replay`, `as_of`, or `skipped_audit` raises `DuplicateRecordError`.
4. **Stream Segregation:** Clear semantic partitioning across in-sample training causal filtering, out-of-sample replay, live as-of, and retrospective audit runs.
5. **Columnar Parquet Exports & Analytical Audit Streams:**
   * `data/processed/in_sample_inference_log.parquet`: Exactly **12,501 rows** (100% complete candles, in-sample causal filtering).
   * `data/processed/as_of_replay_inference_log.parquet`: Exactly **8,335 rows** (100% complete candles, out-of-sample historical causal replay).
   * `data/processed/as_of_inference_log.parquet`: Exactly **8,335 rows** (legacy alias for out-of-sample replay stream).
   * `data/processed/skipped_audit_log.parquet`: Exactly **796 rows** (100% unique timestamps, skipped partial candles).
   * `data/processed/full_candle_audit_log.parquet`: Exactly **21,632 rows** (chronological stream of all 2023 bars, 20,836 valid + 796 skipped).
   * `data/processed/retrospective_inference_log.parquet`: Exactly **50 rows** (retrospective audit records).

### 6.2 Record Field Specification
| Field Name | Type | Description |
| :--- | :--- | :--- |
| `record_id` | string (hex) | Unique SHA256 hash identifying the record. |
| `timestamp_utc` | string (ISO-8601) | Bar completion timestamp in UTC. |
| `model_version` | string | Unique model identifier (`Gaussian_HMM_K3_M1_v1` or `GMM_HMM_K3_M3_v1`). |
| `feature_version` | string | Feature pipeline version (`Config_2_MinimalOrthogonal_v1`). |
| `status` | string | Status: `valid_update` or `skipped_partial`. |
| `is_complete` | boolean | True for complete M15 candles; False for partial candles. |
| `is_valid_prediction` | boolean | True for valid filtering updates; False for skipped audit records. |
| `filtered_probs` | array of floats / null | Filtered regime posterior $\boldsymbol{\alpha}_t$ (null for skipped records). |
| `pred_state` | integer / null | Argmax regime state $\hat{S}_t$ (null for skipped records). |
| `pred_log_likelihood` | float / null | Causal predictive log-likelihood (null for skipped records). |
| `gap_type` | string | Interval classification: `normal`, `active_missing`, `market_closure`, `partial_bar_skipped`. |
| `missing_bars` | integer | Number of unobserved intermediate bars during the interval. |
| `inference_type` | string | Stream category: `in_sample_filtered`, `as_of_replay`, `as_of`, `skipped_audit`, or `retrospective`. |
| `run_id` | string | Identifier of the deployment or simulation session. |
| `feature_r_oc` | float | Realized body return $r_{OC, t}$. |
| `feature_log_vol` | float | Realized log-volatility $\ln(\ln(H_t/L_t))$. |
| `logged_at_utc` | string (ISO-8601) | Clock timestamp when record was committed to disk. |

---

## 7. Artifact Checklist & Deliverables
- [x] `src/inference/calendar.py`: FX market calendar and gap classification module.
- [x] `src/inference/versioning.py`: ModelArtifact schema, serialization, and parameter SHA256 checksums.
- [x] `src/inference/engine.py`: RealTimeInferenceEngine with sequential step filtering, partial-bar skipping, and state persistence.
- [x] `src/inference/storage.py`: ImmutableProbabilityStore with append-only JSONL, duplicate rejection, and Parquet export.
- [x] `src/inference/retraining.py`: RetrainingValidator, RetrainingManager, and warm-up sensitivity evaluator.
- [x] `tests/test_inference.py`: Unit and regression test suite covering all required test criteria (100% pass).
- [x] `models/Gaussian_HMM_K3_M1_v1.json`: Serialized artifact with parameter hash.
- [x] `models/GMM_HMM_K3_M2_v1.json`: Serialized artifact with parameter hash.
- [x] `models/GMM_HMM_K3_M3_v1.json`: Serialized artifact with parameter hash.
- [x] `data/reports/figures/13_warmup_length_sensitivity.png`: Warm-up convergence diagnostic figure.
- [x] `data/reports/figures/14_realtime_filtered_probabilities_and_cutover.png`: Real-time probability timeline and model cutover.
- [x] `data/reports/figures/15_missing_bar_propagation_vs_closure.png`: Gap propagation vs market closure diagnostic figure.
- [x] `data/processed/as_of_inference_log.jsonl`: Immutable append-only log (21,682 total records).
- [x] `data/processed/in_sample_inference_log.parquet`: Exactly 12,501 in-sample filtered training records.
- [x] `data/processed/as_of_replay_inference_log.parquet`: Exactly 8,335 out-of-sample historical replay records.
- [x] `data/processed/as_of_inference_log.parquet`: Exactly 8,335 records (legacy alias for out-of-sample replay stream).
- [x] `data/processed/skipped_audit_log.parquet`: Exactly 796 skipped partial-candle audit records.
- [x] `data/processed/full_candle_audit_log.parquet`: Exactly 21,632 chronological records matching raw input.
- [x] `data/processed/retrospective_inference_log.parquet`: Exactly 50 retrospective evaluation records.
- [x] `data/reports/phase5_warmup_sensitivity.json`: Multi-window empirical warm-up sweep results.
- [x] `data/reports/phase5_deployment_history.json`: Model cutover migration audit log.
- [x] `data/reports/phase5_inference_summary.json`: Comprehensive execution metrics.

Phase 5 is complete, mathematically verified, audited, and ready for Phase 6 (Regime Interpretation & Label Alignment).