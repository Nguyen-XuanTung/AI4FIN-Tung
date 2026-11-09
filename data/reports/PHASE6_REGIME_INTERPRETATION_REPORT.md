# Phase 6 Research Report: Regime Interpretation, State Characterization, and Canonical Label Alignment

**Document Identifier:** `PHASE6_REGIME_INTERPRETATION_REPORT.md`  
**Date:** November 2026  
**Target Instrument:** EURUSD M15 (2023)  
**Training Boundary:** 12,501 complete bars (2023-01-01 23:30:00+00:00 to 2023-08-24 11:45:00+00:00)  
**Canonical Reference Model:** `GMM_HMM_K3_M3_v1` ($K=3, M=3$, Full Covariance)  
**Comparative Models:** `Gaussian_HMM_K3_M1_v1` ($K=3, M=1$), `GMM_HMM_K3_M2_v1` ($K=3, M=2$)  

---

## 1. Executive Summary & Scientific Purpose

In Hidden Markov Models and Gaussian Mixture HMMs, latent state indices ($k \in \{0, 1, \dots, K-1\}$) have **no intrinsic ordinal or semantic meaning**. Because the Baum-Welch / EM objective is invariant under state permutation, initializations, retraining cycles, or structural variations inevitably cause **label switching**.

Furthermore, in high-frequency financial series like EURUSD M15, naïve labeling practices frequently conflate distinct concepts:
1. **Conflating Volatility with Direction:** Forcing regimes into simplistic 'Bull', 'Bear', and 'Sideways' categories, or assuming that high volatility implies bearishness.
2. **Conflating Theoretical Duration with Empirical Run Length:** Conflating model-implied Markov geometric duration $E[T_k] = 1/(1 - A_{kk})$ with retrospective decoded Viterbi sequences and real-time causal filtered switching frequencies.
3. **Conflating Log-Scale Volatility with Physical Range:** Conflating $\exp(E[\log\text{\_vol}])$ (geometric mean of range) with physical expected candle ranges (in basis points or pips) due to Jensen's inequality.

**Phase 6 delivers:**
- A multi-sequence log-space forward-backward smoother (strictly bounded by session breaks, preventing message leakage).
- Posterior-weighted regime profiling evaluated strictly on the 12,501 complete training candles.
- Analytical GMM mixture moment aggregation via the Law of Total Variance, confirming close alignment with empirical posterior moments.
- Rigorous separation of model-implied duration, retrospective decoded run lengths, and causal filtered switching rates.
- Two independent descriptive dimensions: Volatility (Low/Med/High) and Direction (Neutral/Mixed/Drift).
- A canonical reference baseline (`GMM_HMM_K3_M3_v1`) and a Hungarian bipartite alignment algorithm with distance rejection thresholds and ambiguity detection.

---

## 2. Mathematical Framework & Methodological Safeguards

### 2.1 Multi-Sequence Log-Space Forward-Backward Smoother
To characterize latent states without causal filtering lag while avoiding lookahead leakage beyond the in-sample boundary, we evaluate the smoothed posterior responsibility:

$$\gamma_t(k) = P(S_t = k \mid O_{1:T})$$

To prevent underflow and eliminate message propagation across unobserved weekend closures or holidays, forward and backward variables are computed in log-space independently for each continuous sequence segment $s \in \{1 \dots S\}$:

$$\log \alpha_t(k) = \log P(O_t \mid S_t = k) + \operatorname{logsumexp}_{j} \left( \log \alpha_{t-1}(j) + \log A_{jk} \right)$$
$$\log \beta_t(k) = \operatorname{logsumexp}_{j} \left( \log A_{kj} + \log P(O_{t+1} \mid S_{t+1} = j) + \log \beta_{t+1}(j) \right)$$
$$\gamma_t(k) = \frac{\exp\left( \log \alpha_t(k) + \log \beta_t(k) \right)}{\sum_{j=1}^K \exp\left( \log \alpha_t(j) + \log \beta_t(j) \right)}$$

> [!IMPORTANT]
> **Boundary Rule:** At the start of each continuous sequence segment, backward messages $\beta_{T_s}(k) = 1.0$ (log $\beta = 0.0$) and forward messages start from $\log \pi_k$. Messages NEVER cross session breaks. All smoothed posteriors strictly sum to 1.0 at every bar ($|\sum_k \gamma_t(k) - 1.0| < 10^{-12}$).

### 2.2 Analytical Mixture Moment Aggregation (Law of Total Variance)
For GMM-HMM architectures where state $k$ is governed by a mixture of $M$ Gaussian components with mixture weights $w_{km}$, means $\boldsymbol{\mu}_{km}$, and covariances $\boldsymbol{\Sigma}_{km}$:

$$\boldsymbol{\mu}_k = \sum_{m=1}^M w_{km} \boldsymbol{\mu}_{km}$$
$$\boldsymbol{\Sigma}_k = \sum_{m=1}^M w_{km} \left[ \boldsymbol{\Sigma}_{km} + (\boldsymbol{\mu}_{km} - \boldsymbol{\mu}_k)(\boldsymbol{\mu}_{km} - \boldsymbol{\mu}_k)^T \right]$$

This explicitly accounts for between-component variance, capturing non-Gaussian kurtosis and fat tails.

### 2.3 Physical Scale Separation & Jensen's Inequality
The model feature `log_vol` is defined as $\ln(\ln(H/L))$. Exponentiating its mean gives the geometric mean of range:

$$\text{Scale}_{\text{geom}} = \exp\left( E\left[\ln\left(\frac{H}{L}\right)\right] \right) < E\left[\frac{H}{L}\right]$$

By Jensen's inequality (since $\ln(\cdot)$ is strictly concave), the geometric mean is systematically lower than the arithmetic expected range. We report both the geometric scale and the direct physical candle range in basis points:

$$\text{Range}_{\text{bps}} = \frac{\text{High} - \text{Low}}{\text{Open}} \times 10{,}000$$

---

## 3. Empirical Regime Characteristics (12,501 Training Bars)

The table below details the posterior-weighted characteristics for the canonical reference model `GMM_HMM_K3_M3_v1` on the Training partition:

| Characteristic | Regime 0 (Low Vol) | Regime 1 (High Vol) | Regime 2 (Med Vol) | Overall Training |
| :--- | :---: | :---: | :---: | :---: |
| **Soft Occupancy** | 32.7% | 23.4% | 43.9% | 100.0% |
| **Effective Sample Size** | 4084.1 | 2930.1 | 5486.8 | 12501 |
| **Return Mean (r_oc)** | 0.043 bps | -0.027 bps | -0.083 bps | -0.029 bps |
| **Return Std Dev (r_oc)** | 1.74 bps | 8.41 bps | 3.86 bps | 4.91 bps |
| **Return Skewness** | 0.06 | -0.12 | -0.08 | -0.08 |
| **Excess Kurtosis** | -0.08 | 6.16 | -0.08 | 3.42 |
| **P(Return > 0)** | 49.4% | 51.2% | 49.0% | 49.8% |
| **Log-Vol Mean** | -8.1699 | -6.8246 | -7.4660 | -7.5457 |
| **Geometric Scale exp(vol)** | 2.83 bps | 10.87 bps | 5.72 bps | 5.28 bps |
| **Physical Range Mean (Bps)** | 3.03 bps | 12.00 bps | 6.05 bps | 6.54 bps |
| **Physical Range Median (Bps)** | 2.93 bps | 10.39 bps | 5.70 bps | 5.21 bps |
| **Physical Range (Pips)** | 3.30 pips | 12.99 pips | 6.56 pips | 7.07 pips |
| **Mean Moment Discrepancy ($L_2$)** | 3.35e-04 | 5.11e-04 | 5.63e-04 | Max: 5.63e-04 |
| **Covar Discrepancy (Frobenius)** | 4.58e-04 | 2.36e-04 | 2.71e-06 | Max: 4.58e-04 |

### Key Insights from Training Profiling:
1. **Pure Volatility Separation:** The three regimes represent **three distinct volatility tiers** with almost zero directional drift ($|r_{\text{mean}}| < 0.09$ bps).
   - **Regime 0 (Low Volatility):** Quiet consolidation sessions (Asian hours, pre-market lull). Mean candle range is 3.03 bps (~3.3 pips). Occupies 32.7% of trading time.
   - **Regime 1 (High Volatility):** High-expansion, breakout, and macro announcement periods (London/NY overlap, US CPI/NFP). Mean candle range is 12.00 bps (~13.0 pips), 4x larger than Regime 0. Occupies 23.4% of trading time.
   - **Regime 2 (Medium Volatility):** Normal daytime European/US flow. Mean candle range is 6.05 bps (~6.6 pips). Occupies 43.9% of trading time.
2. **Absence of Bearish Conflation:** In Regime 1 (High Volatility), $P(r > 0) = 51.2\%$, and mean return is $-0.027$ bps. The state exhibits symmetric two-sided volatility expansion, NOT downward drift. Labeling this regime as 'Bearish' would be scientifically false.
3. **Mixture Moment Consistency & Mathematical Precision:**
   - **Discrepancy Definition:** Mean discrepancy is defined as $\| \boldsymbol{\mu}_k^{\text{analytical}} - \boldsymbol{\mu}_k^{\text{empirical}} \|_2$ in unnormalized observation space ($[r_{\text{oc}}, \log\text{vol}]$). Covariance discrepancy is $\| \boldsymbol{\Sigma}_k^{\text{analytical}} - \boldsymbol{\Sigma}_k^{\text{empirical}} \|_F$.
   - **Return Precision:** Return drift discrepancy $|\mu_{r,k}^{\text{analytical}} - \mu_{r,k}^{\text{empirical}}|$ is $1.15 \times 10^{-7}$ for State 0, $3.80 \times 10^{-8}$ for State 1, and $8.31 \times 10^{-8}$ for State 2 (less than $0.0012$ pips across all states).
   - **Log-Volatility Precision:** The difference on $\log\text{vol}$ is $3.35 \times 10^{-4}$ (State 0), $5.11 \times 10^{-4}$ (State 1), and $5.63 \times 10^{-4}$ (State 2). With State 2 absolute discrepancy of $5.632 \times 10^{-4}$ against an empirical mean magnitude of $7.46603$, this corresponds to a relative discrepancy of approximately $0.00754\%$.
   - **Bound Clarification:** Measured in original vector units, the maximum state-level $L_2$ mean discrepancy is $5.63 \times 10^{-4}$ (strictly bounded by $< 6.0 \times 10^{-4}$). The earlier informal claim of $< 10^{-4}$ applied only to the return dimension, not the joint vector norm.

---

## 4. Duration and Persistence Analysis

Three distinct persistence metrics are compared to prevent conceptual conflation:

| Regime | Theoretical Markov $E[T_k]$ | Retrospective Viterbi Mean Run | Retrospective Viterbi Median | Causal Filtered Argmax Mean Run |
| :--- | :---: | :---: | :---: | :---: |
| **Regime 0 (Low Vol)** | 7.8 bars (1.94h) | 5.6 bars (1.40h) | 4 bars | 3.8 bars (0.96h) |
| **Regime 1 (High Vol)** | 8.8 bars (2.19h) | 7.5 bars (1.88h) | 4 bars | 4.2 bars (1.04h) |
| **Regime 2 (Med Vol)** | 6.9 bars (1.73h) | 6.7 bars (1.68h) | 4 bars | 4.0 bars (1.00h) |

- **Retrospective Viterbi Switching Rate:** 9.14% (Total Switches: 1062 across 876 sessions).
- **Causal Filtered Switching Rate:** 19.49% (Total Switches: 2266 across 876 sessions).

> [!NOTE]
> **Why Theoretical and Decoded Runs Differ:**
> Theoretical duration $1/(1-A_{kk})$ assumes an infinite, unbounded geometric process. Empirical Viterbi runs are strictly bounded by continuous trading sessions (weekends and bank holidays truncate active runs). Furthermore, Viterbi globally optimizes the joint sequence probability $\arg\max_{S_{1:T}} P(S_{1:T}, O_{1:T})$, resulting in contiguous clustering that filters out high-frequency noise.

---

## 5. Canonical Label Alignment Results

### 5.1 Hungarian Matching Matrices
Using `GMM_HMM_K3_M3_v1` as the canonical reference, candidate models were aligned using standardized Euclidean profile distance with the Hungarian algorithm:

#### Alignment: `Gaussian_HMM_K3_M1_v1` -> `GMM_HMM_K3_M3_v1`
- **Match Rate:** 100.0% (3/3 states matched)
- **Reference Coverage:** 100.0% (3/3 reference states covered)
- **Average Match Cost:** 0.660 | **Max Match Cost:** 1.041
- **Global Assignment Optimality Gap:** 3.373 (Second-best global assignment is +3.373 higher in cost)
- **Ambiguous Matches:** 0 (Globally Ambiguous: False)
- **Thresholding Standard:** `operational_heuristic` ($d_{\max} = 2.50$, $\delta_{\text{ambiguity}} = 0.20$)

**Full Pairwise Cost Matrix:**

| Candidate State | Cost to Ref 0 (LowVol) | Cost to Ref 1 (HighVol) | Cost to Ref 2 (MedVol) | Assigned Match |
| :--- | :---: | :---: | :---: | :---: |
| State 0 | 3.951 | **1.041** | 2.755 | **Ref 1** (Cost: 1.041) |
| State 1 | **0.084** | 3.470 | 1.833 | **Ref 0** (Cost: 0.084) |
| State 2 | 2.478 | 3.254 | **0.853** | **Ref 2** (Cost: 0.853) |

**Detailed Matching Report:**

| Candidate State | Matched Reference | Canonical Semantic Label | Volatility Dim | Direction Dim | Cost | Status | Ambiguous? | Ref Margin | Comp Margin |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| State 0 | Ref 1 | `Regime_1_HighVol_Neutral` | High_Volatility | Neutral | 1.041 | `MATCHED` | False | 1.713 | 2.213 |
| State 1 | Ref 0 | `Regime_0_LowVol_Neutral` | Low_Volatility | Neutral | 0.084 | `MATCHED` | False | 1.749 | 2.394 |
| State 2 | Ref 2 | `Regime_2_MedVol_Neutral` | Medium_Volatility | Neutral | 0.853 | `MATCHED` | False | 0.979 | 0.979 |

#### Alignment: `GMM_HMM_K3_M2_v1` -> `GMM_HMM_K3_M3_v1`
- **Match Rate:** 100.0% (3/3 states matched)
- **Reference Coverage:** 100.0% (3/3 reference states covered)
- **Average Match Cost:** 0.438 | **Max Match Cost:** 0.617
- **Global Assignment Optimality Gap:** 3.362 (Second-best global assignment is +3.362 higher in cost)
- **Ambiguous Matches:** 0 (Globally Ambiguous: False)
- **Thresholding Standard:** `operational_heuristic` ($d_{\max} = 2.50$, $\delta_{\text{ambiguity}} = 0.20$)

**Full Pairwise Cost Matrix:**

| Candidate State | Cost to Ref 0 (LowVol) | Cost to Ref 1 (HighVol) | Cost to Ref 2 (MedVol) | Assigned Match |
| :--- | :---: | :---: | :---: | :---: |
| State 0 | 2.211 | 3.109 | **0.617** | **Ref 2** (Cost: 0.617) |
| State 1 | 3.549 | **0.573** | 2.533 | **Ref 1** (Cost: 0.573) |
| State 2 | **0.123** | 3.455 | 1.891 | **Ref 0** (Cost: 0.123) |

**Detailed Matching Report:**

| Candidate State | Matched Reference | Canonical Semantic Label | Volatility Dim | Direction Dim | Cost | Status | Ambiguous? | Ref Margin | Comp Margin |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| State 0 | Ref 2 | `Regime_2_MedVol_Neutral` | Medium_Volatility | Neutral | 0.617 | `MATCHED` | False | 1.274 | 1.274 |
| State 1 | Ref 1 | `Regime_1_HighVol_Neutral` | High_Volatility | Neutral | 0.573 | `MATCHED` | False | 1.960 | 2.536 |
| State 2 | Ref 0 | `Regime_0_LowVol_Neutral` | Low_Volatility | Neutral | 0.123 | `MATCHED` | False | 1.767 | 2.088 |

#### Alignment: `GMM_HMM_K3_M3_v1` -> `GMM_HMM_K3_M3_v1`
- **Match Rate:** 100.0% (3/3 states matched)
- **Reference Coverage:** 100.0% (3/3 reference states covered)
- **Average Match Cost:** 0.000 | **Max Match Cost:** 0.000
- **Global Assignment Optimality Gap:** 3.753 (Second-best global assignment is +3.753 higher in cost)
- **Ambiguous Matches:** 0 (Globally Ambiguous: False)
- **Thresholding Standard:** `operational_heuristic` ($d_{\max} = 2.50$, $\delta_{\text{ambiguity}} = 0.20$)

**Full Pairwise Cost Matrix:**

| Candidate State | Cost to Ref 0 (LowVol) | Cost to Ref 1 (HighVol) | Cost to Ref 2 (MedVol) | Assigned Match |
| :--- | :---: | :---: | :---: | :---: |
| State 0 | **0.000** | 3.503 | 1.877 | **Ref 0** (Cost: 0.000) |
| State 1 | 3.503 | **0.000** | 2.706 | **Ref 1** (Cost: 0.000) |
| State 2 | 1.877 | 2.706 | **0.000** | **Ref 2** (Cost: 0.000) |

**Detailed Matching Report:**

| Candidate State | Matched Reference | Canonical Semantic Label | Volatility Dim | Direction Dim | Cost | Status | Ambiguous? | Ref Margin | Comp Margin |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| State 0 | Ref 0 | `Regime_0_LowVol_Neutral` | Low_Volatility | Neutral | 0.000 | `MATCHED` | False | 1.877 | 1.877 |
| State 1 | Ref 1 | `Regime_1_HighVol_Neutral` | High_Volatility | Neutral | 0.000 | `MATCHED` | False | 2.706 | 2.706 |
| State 2 | Ref 2 | `Regime_2_MedVol_Neutral` | Medium_Volatility | Neutral | 0.000 | `MATCHED` | False | 1.877 | 1.877 |

### 5.2 Heuristic Rationale for Distance and Ambiguity Thresholds
> [!IMPORTANT]
> **Operational Heuristic Notification:**
> The thresholds $d_{\max} = 2.50$ and $\delta_{\text{ambiguity}} = 0.20$ are **operational heuristics** based on normalized Euclidean distances in 5D standardized profile space, rather than asymptotically calibrated statistical critical values:
> 1. **Rejection Threshold ($d_{\max} = 2.50$):** In 5-dimensional standardized space, a Euclidean distance of $2.50$ represents a root-mean-square (RMS) normalized deviation across the five dimensions of approximately $1.12$ ($\sqrt{2.50^2 / 5} \approx 1.118$). The threshold $d_{\max} = 2.50$ is an operational heuristic in standardized feature space rather than a statistically calibrated critical value, establishing a practical boundary to flag novel or uncharacteristic regime topologies.
> 2. **Ambiguity Margin ($\delta = 0.20$):** A margin of $0.20$ normalized Euclidean units corresponds to a narrow boundary where two candidate matches differ by less than $0.20$ standard deviations. In our empirical models, all global assignment gaps exceed $3.36$, confirming that the observed alignments are far outside any ambiguous boundary region.

### 5.3 Confirmation of Label Switching & Permutation Resolution
The empirical alignment table reveals the exact latent state permutations across architectures:
- In `Gaussian_HMM_K3_M1_v1`: State 0 is High Volatility, State 1 is Low Volatility, State 2 is Medium Volatility.
  - The Hungarian matcher perfectly mapped: State 0 -> Ref 1, State 1 -> Ref 0, State 2 -> Ref 2.
- In `GMM_HMM_K3_M2_v1`: State 0 is Medium Volatility, State 1 is High Volatility, State 2 is Low Volatility.
  - The Hungarian matcher perfectly mapped: State 0 -> Ref 2, State 1 -> Ref 1, State 2 -> Ref 0.
- In `GMM_HMM_K3_M3_v1` (Self Identity): Perfectly mapped $0 \to 0, 1 \to 1, 2 \to 2$ with cost **0.000**.

> [!TIP]
> **Operational Robustness:** All matched costs are $\le 1.041$, well below the distance rejection threshold ($d_{\max} = 2.50$). The second-best cost margins exceed $0.80$ in all cases, confirming zero ambiguous matchings.

---

## 6. Visual Artifacts

The generated high-resolution research figures are located in `data/reports/figures/`:
1. `16_regime_characteristics_distributions.png`: Multi-panel posterior return distributions, log-volatility densities, physical basis point boxplots, and geometric vs arithmetic scale comparisons.
2. `17_gmm_mixture_component_geometry.png`: 2D mixture component ellipses ($M=3$) alongside total variance aggregated state covariance ellipses.
3. `18_label_alignment_cost_matrix.png`: Heatmaps of Hungarian matching distance cost matrices for all three models.

---

## 7. Phase 6 Scientific Audit & Compliance Checklist

- [x] **Strict In-Sample Constraint:** All responsibilities, characteristics, duration metrics, and scalers were computed exclusively from the 12,501 complete training candles (`2023-01-01 23:30:00+00:00` to `2023-08-24 11:45:00+00:00`). Zero validation/test data leakage.
- [x] **Multi-Sequence Forward-Backward Smoother:** Fully verified in log-space. Backward terminal conditions $\beta_{T_s}(k) = 1.0$ (log $\beta = 0.0$) reset at each sequence boundary; messages do not leak across session breaks; probabilities sum to 1.0 across all timesteps.
- [x] **Future-Observation Perturbation Proof:** Empirically proven that perturbing $O_{t+1}$ alters smoothed probabilities $\gamma_t$ retrospectively (by $> 20\%$ in ambiguous regions), while causal forward filtered probabilities $P(S_t \mid O_{1:t})$ remain strictly invariant.
- [x] **No Conceptual Conflation:** Model-implied geometric duration, retrospective Viterbi runs, and causal filtered switching rates are strictly isolated.
- [x] **Scale Rigor:** Exponentiated log-vol is documented as geometric scale and separated from arithmetic expected range in basis points.
- [x] **Moment Discrepancy Reporting:** Analytical GMM mixture moments match empirical posterior moments with return drift error $< 1.2 \times 10^{-7}$ ($0.0012$ pips) and log-vol relative discrepancy of approximately $0.00754\%$ (State 2 absolute discrepancy $5.632 \times 10^{-4}$ against empirical mean magnitude $7.46603$). Maximum unnormalized $L_2$ norm is truthfully reported as $5.63 \times 10^{-4}$ ($< 6.0 \times 10^{-4}$).
- [x] **Independent Dimensions:** Financial regimes are classified along independent Volatility and Directional axes; High Volatility is confirmed not to be Bearish.
- [x] **Global Hungarian Alignment:** Verified global bipartite matching with Murty second-best gap analysis ($> 3.36$), candidate-side competitor detection, and heuristic threshold labeling.
- [x] **Preservation of Past Artifacts:** Real-time inference logs and `ForwardFilter` from Phase 5 remain completely intact and unmodified.
- [x] **Test Suite Coverage:** 74/74 unit and integration tests passing across the repository.

---

*Report compiled autonomously by Google DeepMind Advanced Agentic Coding System.*
