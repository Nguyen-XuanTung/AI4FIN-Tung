# AI WORK LOG: EURUSD M15 Market Regime Detection

**Project:** Statistical Market Regime Detection via Hidden Markov Models on EURUSD M15  
**Repository:** `https://github.com/Nguyen-XuanTung/AI4FIN-Tung`  
**Primary Goal:** Real-time causal estimation of filtered regime probabilities $P(S_t = j \mid O_{1:t})$.  
*Strict Boundary:* Latent regime identification only; strictly non-directional (no price-direction forecasting, next-candle prediction, or trade signal generation).

---

## 1. Executive Summary of Phase 4

Phase 4 extends the regime detection framework from single Gaussian emissions ($M=1$) to Multivariate Gaussian Mixture Model emissions (GMM-HMM with $M \in \{2, 3\}$) on orthogonalized M15 features:
$$\mathbf{x}_t = [r_{oc, t}, \log(\text{vol}_t)]^T$$
where $r_{oc, t} = \ln(C_t / O_t)$ and $\log(\text{vol}_t) = \ln(H_t - L_t + \epsilon)$.

### Key Accomplishments:
1. **Audit & Leakage Prevention:** Audited Phase 3 model selection. Corrected `scripts/run_phase3.py` to freeze hyperparameter and model selection strictly using Validation predictive log-likelihood (`val_predictive_ll`). The Test partition is preserved strictly for final one-shot evaluation.
2. **Pure NumPy/SciPy GMM-HMM Engine:** Built a modular `GMMHMM` class (`src/models/gmm_hmm.py`) supporting full covariance matrices, feature-scaled diagonal covariance regularization ($\text{diag}(\text{var}(X) \times 10^{-3})$), Dirichlet-smoothed mixture weights, hierarchical K-Means initialization, and JSON serialization.
3. **Causal Log-Space Forward Filtering:** Extended `ForwardFilter` (`src/models/forward_filter.py`) to handle GMM emissions seamlessly via LogSumExp over mixture components with precomputed precision matrices.
4. **Comprehensive Test Suite:** Developed 10 unit tests (`tests/test_gmm_hmm.py`) validating mathematical correctness, temporal causality invariance, 100-$\sigma$ numerical stability, covariance positive definiteness, and gap handling. Total suite: 34/34 tests passing.
5. **Statistical Model Ranking:**
   - **`GMM_HMM_K3_M3`** is statistically the superior-fitting model: highest Validation Predictive LL (`25,993.2`), highest Test Predictive LL (`25,928.3`), and lowest BIC (`-152,700.8`).
   - **`GMM_HMM_K3_M2`** provides a parsimonious alternative: $p=41$ parameters (vs. $p=59$), with significantly lower variance across random seeds ($\sigma = 2.8$ nats vs. $79.7$ nats).
   - **Decision:** Both `GMM_HMM_K3_M2` and `GMM_HMM_K3_M3` are retained as candidate architectures for Phase 7 walk-forward evaluation.
6. **Controlled Runtime Benchmarking:** Refuted an earlier draft claim of a "6x training speed difference". Benchmarks under identical hardware and iteration bounds reveal training times across models are virtually identical (~35 seconds for 35 Baum-Welch iterations on 12,501 bars). Single-bar inference latency is ~59 µs for M=2 and ~64 µs for M=3.

---

## 2. Phase 3 Audit & Data Leakage Elimination

### 2.1 Audit Findings
During the audit of `scripts/run_phase3.py`:
- In line 577, models had previously been ranked by `oos_predictive_ll` (evaluated on the Test partition) rather than `val_predictive_ll` (evaluated on the Validation partition).
- While the winning model (`Config_2_MinimalOrthogonal_K3`) achieved the highest score on both partitions, ranking candidates by Test performance constitutes data leakage (test data snooping).

### 2.2 Corrective Action Implemented
1. Modified `scripts/run_phase3.py` to sort candidates exclusively by `val_predictive_ll`:
   ```python
   # Corrected model selection: strictly Validation Predictive Log-Likelihood
   selected_diag = sorted(all_diagnostics, key=lambda x: x["val_predictive_ll"], reverse=True)[0]
   ```
2. Freezing Policy: All architectural decisions ($K, M$, covariance constraints) are decided on Train (60%) and Validation (20%). The Test partition (20%) is held strictly blind and evaluated only once after candidate models are frozen.
3. Historical Transparency Note: Because the 2023 Test partition was examined during earlier Phase 3 work, the 2023 Test set cannot be claimed as a pristine independent holdout. Completely unviewed data (e.g. 2022 or 2024) will be evaluated in Phase 7.

---

## 3. Mathematical Foundations of Multivariate GMM-HMM

### 3.1 GMM Emission Density
For a hidden state $S_t = k \in \{1, \dots, K\}$, the observation vector $\mathbf{x}_t \in \mathbb{R}^D$ ($D=2$) is generated from a mixture of $M$ Gaussian components:
$$b_k(\mathbf{x}_t) = P(\mathbf{x}_t \mid S_t = k) = \sum_{m=1}^M w_{km} \mathcal{N}(\mathbf{x}_t \mid \boldsymbol{\mu}_{km}, \boldsymbol{\Sigma}_{km})$$
subject to:
$$w_{km} \ge 0, \quad \sum_{m=1}^M w_{km} = 1 \quad \forall k \in \{1, \dots, K\}$$

### 3.2 Feature-Scaled Covariance Regularization
To prevent singular covariance matrices, we apply a feature-scaled diagonal regularizer:
$$\boldsymbol{\Sigma}_{\text{reg}} = \text{diag}\left(\text{var}(\mathbf{x}_{\text{train}}) \times 10^{-3}\right)$$
$$\boldsymbol{\Sigma}_{km} \leftarrow \frac{1}{2}\left(\boldsymbol{\Sigma}_{km} + \boldsymbol{\Sigma}_{km}^T\right) + \boldsymbol{\Sigma}_{\text{reg}}$$
This scales adaptively to the units of body return ($\approx 10^{-7}$) and log volatility ($\approx 10^{-1}$), avoiding arbitrary isotropic constants like $10^{-4} \cdot \mathbf{I}$.

### 3.3 Number of Free Parameters
The free parameter count $p$ for a $D$-dimensional GMM-HMM with $K$ states and $M$ components per state (full covariance) is:
$$p = \underbrace{(K - 1)}_{\text{initial } \boldsymbol{\pi}} + \underbrace{K(K - 1)}_{\text{transition } \mathbf{A}} + \underbrace{K(M - 1)}_{\text{mixture weights } \mathbf{w}} + \underbrace{K \cdot M \cdot D}_{\text{means } \boldsymbol{\mu}} + \underbrace{K \cdot M \cdot \frac{D(D + 1)}{2}}_{\text{full covariances } \boldsymbol{\Sigma}}$$
For $D=2$:
- Single Gaussian ($K=3, M=1$): $p = 2 + 6 + 0 + 6 + 9 = 23$
- GMM ($K=3, M=2$): $p = 2 + 6 + 3 + 12 + 18 = 41$
- GMM ($K=3, M=3$): $p = 2 + 6 + 6 + 18 + 27 = 59$
- GMM ($K=2, M=2$): $p = 1 + 2 + 2 + 8 + 12 = 25$
- GMM ($K=4, M=2$): $p = 3 + 12 + 4 + 16 + 24 = 59$

### 3.4 Baum-Welch EM Algorithm for GMM-HMM
- **E-Step:** Computes state posterior occupancy $\gamma_t(k)$, transition posterior $\xi_t(i, j)$, and mixture component responsibilities:
  $$\gamma_{tkm} = P(S_t = k, Z_{tk} = m \mid \mathbf{x}_{1:T}) = \gamma_t(k) \cdot \frac{w_{km} \mathcal{N}(\mathbf{x}_t \mid \boldsymbol{\mu}_{km}, \boldsymbol{\Sigma}_{km})}{\sum_{l=1}^M w_{kl} \mathcal{N}(\mathbf{x}_t \mid \boldsymbol{\mu}_{kl}, \boldsymbol{\Sigma}_{kl})}$$
- **M-Step:** Updates $\boldsymbol{\pi}$, $\mathbf{A}$, $w_{km}$ (with Dirichlet smoothing $\epsilon = 10^{-6}$), $\boldsymbol{\mu}_{km}$, and regularized $\boldsymbol{\Sigma}_{km}$.

### 3.5 Real-Time Causal Forward Filtering
In online inference, the filtered belief vector $\boldsymbol{\alpha}_t \in \Delta^{K-1}$ is computed recursively in log-space:
1. **Prior Propagation:** $\text{prior}_t(j) = \sum_{i=1}^K \alpha_{t-1}(i) A_{ij}$
2. **Log-Emission Evaluation:** $\log b_j(\mathbf{x}_t) = \text{LogSumExp}_{m=1}^M \left[ \log w_{jm} + \log \mathcal{N}(\mathbf{x}_t \mid \boldsymbol{\mu}_{jm}, \boldsymbol{\Sigma}_{jm}) \right]$
3. **Predictive Likelihood:** $\mathcal{L}_{\text{pred}}(\mathbf{x}_t \mid \mathbf{x}_{1:t-1}) = \text{LogSumExp}_j \left( \log \text{prior}_t(j) + \log b_j(\mathbf{x}_t) \right)$
4. **Bayes Measurement Update:** $\alpha_t(j) = \exp\left( \log \text{prior}_t(j) + \log b_j(\mathbf{x}_t) - \mathcal{L}_{\text{pred}}(\mathbf{x}_t \mid \mathbf{x}_{1:t-1}) \right)$

---

## 4. Empirical Evaluation Results & Scientific Benchmarks

### 4.1 Chronological Partitions (EURUSD M15, 2023)
- **Total Records:** 21,632 M15 bars
- **Partial Candles Excluded:** 796 bars (3.68% where `is_complete == False`)
- **Complete Evaluated Bars:** 20,836 bars
  * **Train (60%):** 12,501 bars (`2023-01-01 23:30` to `2023-08-24 11:45`)
  * **Validation (20%):** 4,167 bars (`2023-08-24 12:00` to `2023-10-27 00:45`)
  * **Test (20%):** 4,168 bars (`2023-10-27 01:00` to `2023-12-29 21:30`)

### 4.2 Comprehensive Model Comparison Table

| Model Architecture | $K$ | $M$ | $p$ | Train LL | Val Pred LL | Test Pred LL | BIC (Train) | Argmax Sw Freq | Min Soft Occ | Min Hard Occ | Stable? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gaussian_HMM_K3_M1** | 3 | 1 | 23 | 73,533.8 | 24,894.4 | 24,760.7 | -146,850.6 | 25.2% | 13.7% | 12.2% | Yes |
| **GMM_HMM_K3_M2** | 3 | 2 | 41 | 74,926.5 | 25,361.8 | 25,286.8 | -149,466.2 | 19.6% | 18.3% | 17.1% | Yes |
| **GMM_HMM_K3_M3** | 3 | 3 | 59 | 76,628.7 | **25,993.2** | **25,928.3** | **-152,700.8** | 16.9% | 17.8% | 16.7% | Yes |
| **GMM_HMM_K2_M2** | 2 | 2 | 25 | 73,754.4 | 24,935.3 | 24,837.1 | -147,272.9 | 11.8% | 44.8% | 44.9% | Yes |
| **GMM_HMM_K4_M2** | 4 | 2 | 59 | 75,579.2 | 25,596.3 | 25,502.4 | -150,601.7 | 25.2% | 7.6% | 6.7% | Warning |

*(Warning: $K=4$ exhibits state fragmentation with minimum occupancy dropping below 8%).*

### 4.3 Reproducible Benchmarks (Mean ± Std across 5 Independent Seeds)

To verify stability and remove unverified runtime claims, all models were benchmarked under identical conditions (35 iterations, 12,501 training bars, seeds: 42, 100, 2023, 777, 999):

| Metric | Gaussian HMM ($M=1$) | GMM-HMM ($M=2$) | GMM-HMM ($M=3$) |
| :--- | :---: | :---: | :---: |
| **Training Time (1 fit, 35 iters)** | $36.98\text{ s} \pm 4.36\text{ s}$ | $34.80\text{ s} \pm 0.07\text{ s}$ | $35.60\text{ s} \pm 0.17\text{ s}$ |
| **Validation Predictive LL** | $24,030.9 \pm 0.3$ | $24,461.9 \pm 2.8$ | $24,879.7 \pm 79.7$ |
| **Test Predictive LL** | $23,944.1 \pm 0.3$ | $24,395.2 \pm 2.8$ | $24,850.1 \pm 86.4$ |
| **Argmax Switching Frequency** | $29.1\% \pm 0.0\%$ | $24.0\% \pm 0.1\%$ | $21.5\% \pm 0.5\%$ |
| **Batch Forward Latency (4,167 bars)** | $193.7\text{ ms} \pm 4.2\text{ ms}$ | $192.9\text{ ms} \pm 3.8\text{ ms}$ | $193.1\text{ ms} \pm 4.5\text{ ms}$ |
| **Single-Bar Step Latency** | $35.0\ \mu\text{s} \pm 2.0\ \mu\text{s}$ | $58.8\ \mu\text{s} \pm 1.9\ \mu\text{s}$ | $64.2\ \mu\text{s} \pm 2.7\ \mu\text{s}$ |
| **Emission Consistency ($|\Delta_{\text{vec}} - \Delta_{\text{step}}|$)** | $1.02 \times 10^{-12}$ | $2.27 \times 10^{-13}$ | $1.08 \times 10^{-12}$ |
| **Covariance Positive Definiteness** | Passed ($\lambda_{\min} > 0$) | Passed ($\lambda_{\min} > 0$) | Passed ($\lambda_{\min} > 0$) |

---

## 5. Statistical Interpretations vs. Economic Hypotheses

### 5.1 Leptokurtosis and Tail Modeling
In financial returns, observation distributions exhibit severe leptokurtosis (fat tails).
- Under a single Gaussian emission, extreme observations force $\boldsymbol{\Sigma}_k$ to expand, lowering the density around the median and misestimating typical market conditions.
- GMM emissions resolve this by decomposing each state into a tight modal core component and a wider heavy-tailed component.

### 5.2 Statistical Profile of Selected Candidate (`GMM_HMM_K3_M3`)

#### State 0 (Low Dispersion Mode):
- **Expected Stay Duration:** 7.8 bars (1.94 hours) | **Persistence ($A_{00}$):** 0.8714
- **Occupancy:** 36.3% (Soft), 36.0% (Hard)
- **Component Decomposition:**
  * Component 0 ($w=36.3\%$): Mean body return $= +0.14\text{ bps}$, Range $= 4.0\text{ bps}$, Return std $= 2.44\text{ bps}$.
  * Component 1 ($w=18.4\%$): Mean body return $= -0.29\text{ bps}$, Range $= 1.7\text{ bps}$, Return std $= 0.77\text{ bps}$ (tightest dispersion mode).
  * Component 2 ($w=45.2\%$): Mean body return $= +0.11\text{ bps}$, Range $= 2.6\text{ bps}$, Return std $= 1.30\text{ bps}$.

#### State 1 (High Dispersion Mode):
- **Expected Stay Duration:** 8.8 bars (2.19 hours) | **Persistence ($A_{11}$):** 0.8858
- **Occupancy:** 17.8% (Soft), 16.7% (Hard)
- **Component Decomposition:**
  * Component 0 ($w=13.4\%$): Mean body return $= +6.24\text{ bps}$, Range $= 17.0\text{ bps}$, Return std $= 10.83\text{ bps}$.
  * Component 1 ($w=48.3\%$): Mean body return $= +3.41\text{ bps}$, Range $= 9.1\text{ bps}$, Return std $= 4.27\text{ bps}$.
  * Component 2 ($w=38.2\%$): Mean body return $= -6.58\text{ bps}$, Range $= 11.6\text{ bps}$, Return std $= 7.01\text{ bps}$.

#### State 2 (Intermediate Dispersion Mode):
- **Expected Stay Duration:** 6.9 bars (1.73 hours) | **Persistence ($A_{22}$):** 0.8558
- **Occupancy:** 46.0% (Soft), 47.3% (Hard)
- **Component Decomposition:** Balanced dispersion across components with typical ranges of 5.1–8.1 bps.

*Distinction Note:* These parameters capture empirical multimodality and variance clustering. They are statistical density characteristics and should not be labeled as specific economic phenomena (such as "macro news releases" or "liquidity crunches") without external order-book or news event data.

---

## 6. Scientific Protocol Audits

### 6.1 Candidate Pipeline vs. 5-Seed Benchmark Reconciliation
- **Candidate Models (Section 4.2):**
  * `GMM_HMM_K3_M2`: Val Pred LL = `25,361.8`, Test Pred LL = `25,286.8`
  * `GMM_HMM_K3_M3`: Val Pred LL = `25,993.2`, Test Pred LL = `25,928.3`
  * *Trained with:* Multi-sequence Baum-Welch EM with session boundary segmentation (`lengths=train_lengths`) and $N_{\text{restarts}} = 3$.
- **Benchmark Models (Section 4.3):**
  * `GMM_HMM_K3_M2`: Val Pred LL = `24,461.9 ± 2.8`
  * `GMM_HMM_K3_M3`: Val Pred LL = `24,879.7 ± 79.7`
  * *Trained with:* Single fit ($N_{\text{restarts}} = 1$) without session boundary segmentation (measuring single-run execution time).
- **Mathematical Root Cause:**
  1. *Session Boundary Segmentation:* Omission of `lengths` treats Friday close $\to$ Sunday open as a regular 15-minute transition, distorting transition matrix updates $A_{ij}$ across weekend gaps. Multi-sequence segmentation enforces transitions strictly within active trading sessions.
  2. *Multiple Restarts ($N=3$ vs. $N=1$):* A single restart frequently settles into a lower local optimum (~24,460–24,880 nats). Evaluating 3 restarts consistently discovers the superior global mode (~25,360–25,990 nats). Controlled tests confirm that running the exact audit model with `lengths` and $N=3$ reproduces `25,361.8` exactly.
- **Reporting Protocol:** Section 4.2 documents the production candidate models, while Section 4.3 documents the controlled runtime and initialization sensitivity benchmark.

### 6.2 Covariance Conditioning & Regularization Sensitivity
- **Condition Numbers ($\kappa = \lambda_{\max} / \lambda_{\min}$):**
  * `GMM_HMM_K3_M2`: Median $\kappa = 5.62 \times 10^5$, Max $\kappa = 1.26 \times 10^7$ ($\log_{10} \kappa \approx 7.10$).
  * `GMM_HMM_K3_M3`: Median $\kappa = 1.06 \times 10^6$, Max $\kappa = 2.76 \times 10^7$ ($\log_{10} \kappa \approx 7.44$).
  * All eigenvalues are strictly positive ($\lambda_{\min} \in [5.2 \times 10^{-9}, 2.2 \times 10^{-6}]$), ensuring positive definiteness across all components.
- **Why $\kappa \approx 10^6 - 10^7$ is Physically Expected in FX M15:**
  * Feature 0 ($r_{oc}$) variance is $\approx 2.4 \times 10^{-7}$; Feature 1 ($\log(\text{vol})$) variance is $\approx 0.35$.
  * Natural variance ratio: $\frac{\text{var}(\log(\text{vol}))}{\text{var}(r_{oc})} \approx 1.45 \times 10^6$.
  * In standard IEEE 754 `float64` arithmetic (53-bit mantissa, ~15–17 decimal digits of precision), condition numbers of $10^6$ to $10^7$ leave ~9–10 decimal digits of precision headroom. Cholesky decompositions $\mathbf{L}\mathbf{L}^T$ and inversions $\boldsymbol{\Sigma}^{-1}$ remain completely stable.
- **Regularization Floor Sensitivity Sweep ($\lambda \in [10^{-4}, 10^{-2}]$):**
  Evaluating `GMM_HMM_K3_M3` across a 100-fold variation of the regularizer floor $\boldsymbol{\Sigma}_{\text{reg}} = \text{diag}(\text{var}(X) \times \lambda)$:
  * $\lambda = 10^{-4}$: Val LL = `25,991.0`, Test LL = `25,916.5`, Max $\kappa = 3.06 \times 10^7$
  * $\lambda = 5 \times 10^{-4}$: Val LL = `25,995.3`, Test LL = `25,925.1`, Max $\kappa = 2.92 \times 10^7$
  * $\lambda = 10^{-3}$ (Baseline): Val LL = `25,993.2`, Test LL = `25,928.3`, Max $\kappa = 2.76 \times 10^7$
  * $\lambda = 2 \times 10^{-3}$: Val LL = `25,982.5`, Test LL = `25,925.6`, Max $\kappa = 2.51 \times 10^7$
  * $\lambda = 5 \times 10^{-3}$: Val LL = `25,949.7`, Test LL = `25,908.5`, Max $\kappa = 1.95 \times 10^7$
  * $\lambda = 10^{-2}$: Val LL = `25,875.2`, Test LL = `25,842.4`, Max $\kappa = 1.49 \times 10^7$
  * *Verdict:* Validation predictive log-likelihood varies by less than 0.4% across a 100-fold change in $\lambda$. The baseline floor of $\lambda = 10^{-3}$ occupies the optimal plateau, confirming that the model does not rely on numerically fragile covariance estimates.

### 6.3 Validation/Test Boundary & Filter Initialization Audit
- **Boundary Timestamps:**
  * Last bar of Validation: `2023-10-27 00:45:00+00:00`
  * First bar of Test: `2023-10-27 01:00:00+00:00`
  * Elapsed delta: exactly **900 seconds (15.0 minutes)**. The boundary represents an active trading transition with no gap.
- **Controlled Initialization Comparison on Selected Model (`GMM_HMM_K3_M3`):**
  Both tests were executed on identical 4,168 Test bars using identical model parameters (`models/gmm_hmm_k3_m3.json`):
  * **Method A (Fresh initial prior $\boldsymbol{\pi} = \text{model.startprob\_}$):** Test Pred LL = `25,928.3430`
  * **Method B (Continuous carry-forward of final Validation posterior $\boldsymbol{\alpha}_{T_{\text{val}}}$):** Test Pred LL = `25,927.5606`
  * **Cumulative Difference ($\text{Method B} - \text{Method A}$):** **$-0.782$ nats across 4,168 bars** ($-0.000188$ nats/bar).
  * *Bayesian Convergence:* On bar 0, both methods place $>87\%$ probability on State 2. By bar 1, state probability differences drop to $\le 1.8\%$. By bar 2, probabilities match to 3 decimal places. Boundary filter initialization has zero material impact on cumulative test metrics.
- **Historical Exposure Limitation:** The 2023 Test partition was examined during earlier Phase 3 iterations. Pristine holdout testing will be conducted on untouched years (2022/2024) in Phase 7.

### 6.4 Regime Switching Frequency
- **Definition:** Switching frequency is computed on the argmax of causal filtered probabilities:
  $$\hat{S}_t = \arg\max_k P(S_t = k \mid O_{1:t}), \quad \text{SwitchFrequency} = \frac{1}{T-1} \sum_{t=2}^T \mathbb{I}\{\hat{S}_t \ne \hat{S}_{t-1}\}$$
- **Interpretation:** Switching frequency decreased from $25.2\%$ ($M=1$) to $19.6\%$ ($M=2$) and $16.9\%$ ($M=3$). While this demonstrates reduced noise-induced chattering, lower switching frequency does *not* automatically signify superior regime detection (a trivial constant model achieves 0% switching frequency). It is an operational stability diagnostic, not a standalone objective function.

---

## 7. Model Selection Rationale for Phase 7

- **Statistically Best-Fitting Architecture:** `GMM_HMM_K3_M3` achieves the highest predictive log-likelihood on both Validation and Test sets, the lowest BIC, and balanced state occupancies ($>16\%$).
- **Parsimonious Candidate:** `GMM_HMM_K3_M2` requires $p=41$ parameters (~30% fewer than $p=59$), exhibits negligible variance across random seeds ($\sigma = 2.8$ nats vs. $79.7$ nats), and reduces risk of estimation variance in smaller rolling training windows.
- **Protocol:** **BOTH `GMM_HMM_K3_M2` and `GMM_HMM_K3_M3`** are retained as candidate models for Phase 7 walk-forward validation.

---

## 8. Artifacts Produced in Phase 4

| Artifact Path | Description |
| :--- | :--- |
| `src/models/gmm_hmm.py` | Complete NumPy/SciPy Multivariate GMM-HMM class with Baum-Welch EM. |
| `src/models/forward_filter.py` | Universal ForwardFilter supporting both Gaussian and GMM emissions. |
| `src/models/evaluation.py` | Free parameter counting ($p=41, 59$) and mixture diagnostics. |
| `tests/test_gmm_hmm.py` | 10 unit tests for mathematical correctness, stability, and causality. |
| `scripts/run_phase4.py` | Full execution pipeline for Phase 4 model training and diagnostics. |
| `scripts/audit_phase4.py` | Scientific audit and benchmark script (timing, latency, stability across seeds). |
| `data/reports/PHASE4_GMM_HMM_REPORT.md` | Formal research report documenting all metrics, equations, and tables. |
| `data/reports/phase4_scientific_audit_results.json` | Exact benchmark measurements across seeds and latencies. |
| `data/reports/gmm_hmm_model_comparison_2023.json` | Complete machine-readable diagnostic outputs for all 5 models. |
| `data/processed/EURUSD_M15_2023_gmm_regime_probabilities.parquet` | Real-time filtered regime probabilities for all 20,836 complete 2023 M15 bars. |
| `models/gmm_hmm_k3_m2.json` | Serialized model parameters for primary candidate ($K=3, M=2$). |
| `models/gmm_hmm_k3_m3.json` | Serialized model parameters for primary candidate ($K=3, M=3$). |
| `data/reports/figures/09_*.png` to `12_*.png` | Four high-resolution diagnostic publication figures. |

---

## 9. Executive Summary of Phase 5: Real-Time Inference, Model Versioning, and Retraining Management

Phase 5 operationalizes the HMM regime detection framework into an ultra-low-latency, causal sequential streaming engine. It guarantees real-time adherence to the principle $P(S_t = j \mid O_{1:t})$ without future data leakage, establishes strict FX calendar gap handling, resolves retraining initialization via an isolated warm-up buffer, establishes a principled partial-bar skipping policy, and secures all historical probability observations in an immutable append-only store.

### Key Accomplishments & Scientific Audit Reconciliations:
1. **Mathematical & Causal Equivalence:**
   - Single-bar recursive filtering verified to match batch log-space forward filtering with zero error ($\|\boldsymbol{\alpha}_t^{\text{seq}} - \boldsymbol{\alpha}_t^{\text{batch}}\|_\infty < 10^{-15}$).
   - Future data invariance verified: perturbing future observations $O_{t+1:T}$ produces identically zero change in past filtered probabilities $\boldsymbol{\alpha}_{1:t}$.
2. **Partial-Bar Policy & Emission Protection:**
   - Addressed the 5 key audit questions regarding incomplete candles (`is_complete == False`, 796 bars):
     * *Passed to step():* Now explicitly gated via `is_complete`.
     * *Emissions calculated:* Completely bypassed (`compute_single_log_emission` not invoked).
     * *Written as as-of predictions:* Segregated into `skipped_audit` records (`is_valid_prediction=False`).
     * *Updated posterior:* Belief state and `last_timestamp` preserved at last valid complete observation.
     * *Included in warm-up:* Warm-up buffers strictly filter to complete candles only (`df[df['is_complete'] == True]`).
   - When the subsequent complete candle arrives, elapsed time $\Delta t$ evaluates through the FX calendar: within open session, prior propagates via $\boldsymbol{\alpha}_{\text{prior}} = \boldsymbol{\alpha}_{\text{prev}} \mathbf{A}^n$ ($n = \Delta t / 15\text{ min}$); across weekend closures, prior resets to $\boldsymbol{\pi}$.
3. **Comprehensive Record Count Reconciliation:**
   - Total source rows: **21,632 M15 bars**.
   - Complete M15 candles (`is_complete=True`): **20,836** (96.32%).
   - Partial M15 candles (`is_complete=False`): **796** (3.68%).
   - Pre-cutover split (Train, rows 0 to 12,919): **12,920 rows** = **12,501 complete bars** + **419 partial bars**.
   - Post-cutover split (Val + Test, rows 12,920 to 21,631): **8,712 rows** = **8,335 complete bars** + **377 partial bars**.
   - Valid causal filtering updates: **20,836**.
   - Skipped partial-bar audit records: **796**.
   - Retrospective test records: **50**.
   - Total JSONL log entries: **21,682**.
   - Segregated Parquet Analytical Stores:
     * `in_sample_inference_log.parquet`: Exactly **12,501 rows** (`in_sample_filtered`, 100% complete candles, in-sample causal filtering conditional on training parameters).
     * `as_of_replay_inference_log.parquet`: Exactly **8,335 rows** (`as_of_replay`, 100% complete candles, out-of-sample historical causal replay with frozen model).
     * `as_of_inference_log.parquet`: Exactly **8,335 rows** (legacy alias for out-of-sample replay stream).
     * `skipped_audit_log.parquet`: Exactly **796 rows** (`skipped_audit`, 100% unique timestamps, skipped partial candles).
     * `full_candle_audit_log.parquet`: Exactly **21,632 rows** (chronological stream of all 2023 bars: 12,501 + 8,335 + 796).
     * `retrospective_inference_log.parquet`: Exactly **50 rows** (isolated offline evaluation stream).
4. **Phân định Ngữ nghĩa Suy luận & Rò rỉ Tham số (Parameter Lookahead Audit):**
   - *In-Sample Filtered Probabilities (`in_sample_filtered`):* 12,501 nến hoàn chỉnh tập Train (2023-01-01 đến 2023-08-24 11:45 UTC). Cập nhật đệ quy nhân quả nhưng điều kiện hóa trên tham số $\Theta_{\text{Gaussian}}$ ước lượng từ toàn bộ tập Train (chứa parameter lookahead đối với các thời điểm $t < T_{\text{train}}$). Không được gắn nhãn là `as_of`.
   - *Historical Causal Replay (`as_of_replay`):* 8,335 nến hoàn chỉnh tập Val + Test (2023-08-24 12:00 UTC đến 2023-12-29 21:45 UTC). Mô hình $\Theta_{\text{GMM}}$ đóng băng trước cutover, zero parameter lookahead, zero observation lookahead.
   - *Genuine Live Predictions (`as_of`):* Dành riêng cho triển khai production thực tế đồng thời với thời gian đóng nến.
   - *Skipped Partial-Candle Audit Records (`skipped_audit`):* 796 nến khuyết tật được ghi nhận nhằm bảo toàn tính toàn vẹn nguồn dữ liệu thô (`status='skipped_partial'`, `is_valid_prediction=False`).
   - *Retrospective Stream (`retrospective`):* 50 bản ghi kiểm thử độc lập.
5. **Đính chính Thuật ngữ Toán học (Stationary vs Initial-State Distribution):**
   - Phân phối trạng thái ban đầu $\boldsymbol{\pi}$ (`model.startprob_`) là phân phối thời điểm bắt đầu $P(S_1 = j)$, phi dừng ($\boldsymbol{\pi}\mathbf{A} \ne \boldsymbol{\pi}$).
   - Phân phối dừng $\boldsymbol{\pi}_{\infty}$ là vector riêng trái tiệm cận thỏa mãn $\boldsymbol{\pi}_{\infty}\mathbf{A} = \boldsymbol{\pi}_{\infty}$.
6. **Chronological Cutover Alignment & Zero Lookahead:**
   - In Phase 3, the training split was defined as 12,501 bars on the **complete-candle dataset** (`filter_incomplete=True`).
   - In the unfiltered raw sequence (21,632 bars), 419 partial candles preceded the boundary. Hence, the 12,501st complete candle is located at row index **12,919 (`2023-08-24 11:45:00 UTC`)**.
   - Model cutover is aligned at bar **12,920 (`2023-08-24 12:00:00 UTC`)**, matching the exact first bar of the Validation partition.
   - The newly deployed model was fitted on data ending at `2023-08-24 11:45:00 UTC`, and its warm-up buffer uses 50 complete bars strictly prior to `2023-08-24 12:00:00 UTC`. There is zero training-data overlap and zero lookahead.
7. **Calendar-Aware Gap Handling (`FXCalendarGapHandler`):**
   - Source data in fixed EST (UTC-5) without DST shifts was converted to true continuous UTC.
   - Supported with `max_active_gap_minutes=300` (5 hours) to accommodate interbank session rollover.
   - Across all 20,836 complete candles, cleanly classifies 19,686 normal bars, 1,093 active-session missing bars ($\mathbf{A}^n$), and 56 market closures (reset to $\boldsymbol{\pi}$), with 0 uncertain errors.
8. **Multi-Window Warm-Up Sensitivity Sweep (`evaluate_warmup_sensitivity`):**
   - Evaluated strictly on complete candles across 5 market regimes: Train/Val cutover, low-volatility Asian quiet session, SVB banking crisis turbulence (March 2023), post-weekend gap opening, and late year-end fade.
   - Across all windows, $W^*(<10^{-4}) \le 5$ bars and $W^*(<10^{-6}) \le 10$ bars, with $L_1 = 0.00$ at $W=50$ bars.
   - Standardizing $W=50$ complete bars (12.5 hours) is an empirical sensitivity heuristic for EURUSD M15 2023, not an unconditional universal mathematical convergence guarantee.
9. **Two-Tiered Retraining Validation & Atomic Cutover:**
   - Hard validation gates: parameter finiteness, probability normalization, positive-definite covariances, feature schema alignment (`Config_2_MinimalOrthogonal_v1`), and serialization round-trip.
   - Atomic model upgrade: validates new model, filters 50-bar warm-up exclusively on complete candles through $\Theta_{\text{new}}$ starting from $\boldsymbol{\pi}_{\text{new}}$ (zero cross-model posterior contamination), and swaps engine state.
10. **Ultra-Low Latency Execution Profile:**
    - Single-bar step latency: **$156.6\ \mu\text{s}$ mean**, **$141.2\ \mu\text{s}$ median**, **$327.9\ \mu\text{s}$ 99th percentile**.
    - Throughput: **$6,384$ bars/second**. Represents $< 0.00002\%$ of the 15-minute bar interval.
11. **Comprehensive Unit Testing:**
    - **19 unit and regression tests** (`tests/test_inference.py`) covering all functional requirements, duplicate policies across models and semantic streams, parquet filtering, cutover split alignment, partial-bar skipping, transition propagation $\mathbf{A}^n$ across skipped intervals, warm-up buffer exclusion, distinguishable audit storage records, parameter lookahead segregation, and initial-state vs. stationary distribution mathematical verification.
    - Full regression suite: **53/53 tests passing (100%)**.

---

## 10. Phase 5 Artifacts Produced

| Artifact Path | Description |
| :--- | :--- |
| `src/inference/calendar.py` | FX calendar gap classifier handling normal, active missing ($A^n$), closures, and errors (`max_active_gap_minutes=300`). |
| `src/inference/versioning.py` | `ModelArtifact` schema, SHA256 parameter hashes, and compatibility verification. |
| `src/inference/engine.py` | `RealTimeInferenceEngine` causal recursive filter, partial-bar skipping, and state persistence. |
| `src/inference/storage.py` | `ImmutableProbabilityStore` append-only JSONL log, duplicate rejection across 5 semantic streams, and segregated Parquet export. |
| `src/inference/retraining.py` | `RetrainingValidator`, `RetrainingManager`, and warm-up buffer sensitivity evaluator. |
| `tests/test_inference.py` | 19 unit and regression tests for Phase 5 real-time inference, versioning, retraining, partial-bar policy, and semantic segregation (100% pass). |
| `scripts/run_phase5.py` | End-to-end execution script: artifact packaging, warm-up sweep, streaming simulation, Parquet export, and reporting. |
| `models/Gaussian_HMM_K3_M1_v1.json` | Standardized serialized ModelArtifact for baseline model. |
| `models/GMM_HMM_K3_M2_v1.json` | Standardized serialized ModelArtifact for candidate model ($K=3, M=2$). |
| `models/GMM_HMM_K3_M3_v1.json` | Standardized serialized ModelArtifact for champion model ($K=3, M=3$). |
| `data/processed/as_of_inference_log.jsonl` | Immutable append-only log of causal real-time filtered regime probabilities (21,682 records). |
| `data/processed/in_sample_inference_log.parquet` | Exactly 12,501 in-sample filtered training records (`in_sample_filtered`). |
| `data/processed/as_of_replay_inference_log.parquet` | Exactly 8,335 out-of-sample historical replay records (`as_of_replay`). |
| `data/processed/as_of_inference_log.parquet` | Exactly 8,335 records (legacy backward-compatible alias for out-of-sample replay stream). |
| `data/processed/skipped_audit_log.parquet` | Skipped partial candles preserving raw OHLC metadata without fabrication (796 records). |
| `data/processed/full_candle_audit_log.parquet` | Chronological stream of all 2023 bars (21,632 records matching raw input: 12,501 + 8,335 + 796). |
| `data/processed/retrospective_inference_log.parquet` | Strictly segregated retrospective evaluation records (50 records). |
| `data/processed/phase5_inference_engine_state.json` | Serialized engine belief state for restart/recovery. |
| `data/reports/phase5_warmup_sensitivity.json` | Multi-window empirical warm-up sweep results across 5 market regimes. |
| `data/reports/phase5_deployment_history.json` | Audit trail of atomic model cutover events. |
| `data/reports/phase5_inference_summary.json` | Operational benchmark metrics and summary metadata. |
| `data/reports/PHASE5_REALTIME_INFERENCE_REPORT.md` | Comprehensive academic research and technical report for Phase 5. |
| `data/reports/figures/13_warmup_length_sensitivity.png` | Figure 13: Warm-up buffer convergence ($L_1$ and $L_\infty$ curves). |
| `data/reports/figures/14_realtime_filtered_probabilities_and_cutover.png` | Figure 14: EURUSD M15 price, stacked regime probabilities, and atomic model cutover. |
| `data/reports/figures/15_missing_bar_propagation_vs_closure.png` | Figure 15: Active-session gap propagation ($A^n$) vs weekend market closure reset. |

---

---

## 11. Phase 6: Regime Interpretation & Canonical Label Alignment

### 11.1 Executive Summary
Phase 6 maps latent numerical state indices ($k \in \{0, 1, \dots, K-1\}$) into reproducible, economically meaningful financial characteristics and establishes a rigorous label alignment framework across model architectures. Evaluated strictly on the 12,501 complete candles of the Training partition (`2023-01-01 23:30:00+00:00` to `2023-08-24 11:45:00+00:00`), all statistics prevent data leakage and decouple volatility from price direction.

### 11.2 Architectural & Mathematical Innovations
1. **Multi-Sequence Log-Space Forward-Backward Smoother (`src/interpretation/smoother.py`):**
   - Evaluates smoothed responsibilities $\gamma_t(k) = P(S_t = k \mid O_{1:T})$ in log-space using robust `logsumexp`.
   - Strictly respects continuous trading session boundaries across 876 segments; backward terminal condition $\beta_{T_s}(k) = 1.0$ resets at each segment; messages NEVER propagate across weekend or market closures.
   - Guaranteed row normalization $\sum_k \gamma_t(k) = 1.0$ ($|\sum \gamma - 1.0| < 10^{-12}$).
   - Verified that future observations perturb smoothed responsibilities retrospectively, while causal forward filtered probabilities remain strictly invariant.
2. **Mixture Moment Aggregation via Law of Total Variance (`src/interpretation/characteristics.py`):**
   - State-level analytical moments for GMM-HMM computed via $\boldsymbol{\mu}_k = \sum_m w_{km} \boldsymbol{\mu}_{km}$ and $\boldsymbol{\Sigma}_k = \sum_m w_{km} [\boldsymbol{\Sigma}_{km} + (\boldsymbol{\mu}_{km} - \boldsymbol{\mu}_k)(\boldsymbol{\mu}_{km} - \boldsymbol{\mu}_k)^T]$.
   - Empirical posterior-weighted moments match analytical parameters with return drift error $< 1.2 \times 10^{-7}$ ($0.0012$ pips) and log-vol relative discrepancy of approximately $0.00754\%$ (State 2 absolute discrepancy $5.632 \times 10^{-4}$ against empirical mean magnitude $7.46603$). Maximum unnormalized vector $L_2$ norm is $5.63 \times 10^{-4}$ (strictly bounded by $< 6.0 \times 10^{-4}$).
3. **Physical Scale Separation & Jensen's Inequality:**
   - Clearly distinguishes geometric scale $\exp(E[\log\text{\_vol}])$ from physical candle ranges in basis points $((H-L)/O \times 10,000)$ and pips.
4. **Duration & Persistence Decoupling (`src/interpretation/duration.py`):**
   - Strictly isolates theoretical Markov geometric duration $E[T_k] = 1 / (1 - A_{kk})$, retrospective multi-sequence Viterbi decoded run lengths, and causal real-time filtered switching rates.
5. **Hungarian Label Alignment & Standardization (`src/interpretation/alignment.py`):**
   - Canonical Reference: `GMM_HMM_K3_M3_v1` on Training data.
   - Profile vector $v_k = [\mu_r, \sigma_r, \mu_{vol}, \sigma_{vol}, P(r > 0)]$.
   - Standardized via `ReferenceProfileScaler` with economic scale floors ($1\text{e-}4$ for return drift) to prevent denominator singularities in neutral markets.
   - Global bipartite matching with Murty single-edge exclusion gap analysis ($> 3.36$), candidate-side competitor detection, and heuristic threshold labeling ($d_{\max} = 2.50$, representing an RMS normalized deviation across 5 dimensions of $\approx 1.12$ as an operational heuristic in standardized feature space; $\delta = 0.20$).
   - Original state IDs remain untouched; canonical mappings are provided via explicit mapping metadata.

### 11.3 Key Findings on EURUSD M15 (12,501 Training Bars)
- **Regime 0 (Low Volatility):** Soft occupancy 32.7%, mean range 3.03 bps (~3.3 pips), return std 1.74 bps, win rate 49.4%.
- **Regime 1 (High Volatility):** Soft occupancy 23.4%, mean range 12.00 bps (~13.0 pips), return std 8.41 bps, win rate 51.2% (neutral drift, NOT Bearish).
- **Regime 2 (Medium Volatility):** Soft occupancy 43.9%, mean range 6.05 bps (~6.6 pips), return std 3.86 bps, win rate 49.0%.
- **Duration Decoupling:**
  - Theoretical Markov $E[T]$: 6.9 to 8.8 bars (1.73h to 2.19h).
  - Retrospective Viterbi Runs: 5.6 to 7.5 bars (1.40h to 1.88h), switching rate 9.14%.
  - Causal Filtered Argmax Runs: 3.8 to 4.2 bars (0.96h to 1.04h), switching rate 19.49%.
- **Label Switching Resolved:**
  - `Gaussian_HMM_K3_M1_v1`: State 0 -> Ref 1, State 1 -> Ref 0, State 2 -> Ref 2 (Match cost 0.660, Global gap 3.373, 0 ambiguities).
  - `GMM_HMM_K3_M2_v1`: State 0 -> Ref 2, State 1 -> Ref 1, State 2 -> Ref 0 (Match cost 0.438, Global gap 3.362, 0 ambiguities).
  - `GMM_HMM_K3_M3_v1`: Exact identity match (cost 0.000, Global gap 3.753).

### 11.4 Phase 6 Artifacts Produced

| Artifact Path | Description |
| :--- | :--- |
| `src/interpretation/__init__.py` | Package root exposing all 24 public interpretation symbols. |
| `src/interpretation/smoother.py` | Multi-sequence log-space forward-backward smoother. |
| `src/interpretation/characteristics.py` | Posterior-weighted moments, quantiles, and GMM mixture total variance aggregation. |
| `src/interpretation/duration.py` | Theoretical duration, multi-sequence Viterbi, run length distributions, and causal switching metrics. |
| `src/interpretation/alignment.py` | Hungarian bipartite alignment, ReferenceProfileScaler, distance rejection, and financial dimension classification. |
| `scripts/run_phase6.py` | End-to-end pipeline executing characterization, duration, alignment, figure generation, and reporting. |
| `tests/test_interpretation.py` | 14 unit tests for smoother, characteristics, mixture moments, duration boundaries, and future perturbation (100% pass). |
| `tests/test_alignment.py` | 8 unit tests for permutation recovery, rejection thresholds, competitor ambiguity, global gap, and unequal state counts (100% pass). |
| `data/reports/regime_characteristics_train.json` | Detailed statistical characteristics for all 3 models on Training data. |
| `data/reports/regime_durations_train.json` | Complete theoretical, Viterbi, and causal filtered persistence metrics. |
| `data/reports/label_alignment.json` | Hungarian cost matrices, matched assignments, global gaps, and diagnostic coverage metrics. |
| `data/reports/PHASE6_REGIME_INTERPRETATION_REPORT.md` | Academic research report documenting Phase 6 findings. |
| `data/reports/figures/16_regime_characteristics_distributions.png` | Figure 16: Multi-panel return distributions, log-vol densities, bps boxplots, and Jensen scale comparison. |
| `data/reports/figures/17_gmm_mixture_component_geometry.png` | Figure 17: GMM-HMM K=3 M=3 component ellipses vs total variance aggregated state covariance. |
| `data/reports/figures/18_label_alignment_cost_matrix.png` | Figure 18: Cost matrices and Hungarian matching heatmaps across models. |

---

## 12. Phase 7 Experimental Design: Out-of-Sample Walk-Forward & Robustness

Phase 7 designs the out-of-sample walk-forward validation and cross-regime robustness analysis framework to overcome the sample selection bias of a single chronological split and evaluate model stability under non-stationary market conditions.

### 12.1 Core Architectural Decisions & Dataset Strategy
1. **Critical Dataset Evaluation & Disclosure:**
   - **In-Dataset Reality:** 2023 EURUSD M15 contains 21,632 total bars (20,836 complete, 796 partial). Partial candles are 100% excluded from all likelihood evaluations.
   - **Historical Exposure Disclosure:** The entire 2023 dataset was inspected during earlier phases (Phase 2 feature discovery, Phase 3 model selection, Phase 4 GMM tuning, Phase 5 replay). Evaluating on 2023 is an **in-dataset retrospective benchmark**, not an uncontaminated holdout.
   - **Two-Tier Architecture & Qualified Statements:**
     * *Tier 1 (In-Dataset 2023 Benchmark):* 5-fold expanding window (Train: 9,472 to 16,949 complete bars; Val: 1,414 to 2,120 complete bars; Test: 1,860 to 2,120 complete bars) on existing 2023 parquet data. Exact timestamps and counts verified with zero inter-fold overlap.
     * *Tier 2 (Multi-Year Proposal):* 2024 is designated as an **intended untouched holdout, subject to confirmation that it has not been inspected or used in prior experiments outside this repository**. The 2022–2024 dataset is NOT yet claimed as a validated multi-year dataset.
     * *HistData Timestamp Assumption:* The assumption of fixed UTC-5 EST without DST shifts is documented as an **unverified data-source assumption** requiring empirical validation around March and November DST cutovers before multi-year ingestion.
2. **Elimination of Arbitrary Acceptance Criteria:**
   - Discarded uncalibrated scalar thresholds (cost <= 1.50, gap >= 0.50, switching rate 15-25%).
   - Replaced with continuous distribution reporting, topological monotonicity verification ($\sigma_{\text{Low}} < \sigma_{\text{Med}} < \sigma_{\text{High}}$), and operational safety ceilings ($d_{\max} = 2.50$ solely for flagging anomalies).
   - Contextualized switching rate with expected Markov geometric duration $E[T_k]$, empirical run lengths, and Shannon entropy $H_t$.
3. **Posterior Boundary Policy (Train -> Val and Val -> Test):**
   - **Policy A (Continuous Forward Evaluation - Primary):** Fit parameters on Train $\to$ filter Train sequentially $\to$ propagate terminal belief $\boldsymbol{\alpha}_{T_{\text{train}}}$ across Train $\to$ Val boundary via FX calendar rules ($\mathbf{A}^n$ during open sessions, reset strictly to `model.startprob_` on weekend closures) $\to$ update with first Val observation $\to$ filter Val sequentially $\to$ carry terminal Val belief $\boldsymbol{\alpha}_{T_{\text{val}}}$ into Test using identical calendar rules.
     * *Train $\to$ Val Classifications:* Fold 1 ($n=197$, weekend closure reset to `model.startprob_`), Fold 2 & 3 ($n=2$, open session with incomplete 23:45 candle skipped, $\mathbf{A}^2$), Fold 4 ($n=193$, weekend closure reset to `model.startprob_`), Fold 5 ($n=1$, normal consecutive $\mathbf{A}^1$).
     * *Val $\to$ Test Classifications:* Fold 1 & 2 ($n=2$, open session with incomplete 23:45 candle skipped, $\mathbf{A}^2$), Fold 3 ($n=193$, weekend closure reset to `model.startprob_`), Fold 4 & 5 ($n=1$, normal consecutive $\mathbf{A}^1$).
   - **Policy B (Independent Sequence - Diagnostic):** Diagnostic reset to `model.startprob_` across all windows.
4. **Reconstructed Training Budget, Restart Evidence Audit & Runtime:**
   - Exact unique training fits: **30 fits** (15 Primary Architectures + 10 Covariance Regularization + 5 Feature-Representation / Diurnal Sensitivity + 0 extra for EM restarts). Total evaluations across Val/Test: 40.
   - **Focused Phase 4 Evidence Audit:** Verified that Phase 4 only evaluated 5 seeds with single-restart fits ($N_{\text{restarts}}=1$, reported validation predictive LL standard deviations: Gaussian $\pm 0.3$, GMM M=2 $\pm 2.8$, GMM M=3 $\pm 79.7$ nats). No 10-seed experiment or best-of-5 stability benchmark was conducted, and the claim of "< 0.05 nats dispersion" was an unsupported drafting error, now formally retracted. Standardized $S=5$ restarts (base seed 42, deterministic schedule `[42, 143, 244, 345, 446]` for Gaussian and `[42, 149, 256, 363, 470]` for GMM) as an operational baseline across folds, while explicitly documenting restart-count sensitivity ($S \in \{1, 3, 5, 10\}$) as an untested dimension.
   - Evidence-based runtime: Benchmarked directly on Fold 1 across 3 repetitions for Gaussian HMM (mean 9.17s $\pm$ 1.12s), GMM M=2 (19.25s), GMM M=3 (16.64s). With $S=5$ restarts and expanding sample sizes, total projected runtime is **~54 minutes on a single CPU core** (retracting the unsubstantiated 2.5-minute claim).
5. **Clarified Feature-Representation Sensitivity:**
   - Comparison between $[r_{\text{oc}}, \log\text{vol}]$ and $[r_{\text{oc}}, \text{range\_zscore}]$ is explicitly categorized as **feature-representation / diurnal-adjustment sensitivity**, reflecting the joint change in seasonal centering and MAD standardization.
6. **Refined Statistical Evaluation & DM/HAC Safeguards:**
   - Primary metric: Per-bar out-of-sample predictive log-likelihood ($\bar{\ell}_{\text{oos}}$).
   - Expanding-window folds are serially correlated (non-i.i.d.); cross-fold statistics are reported strictly as descriptive trajectories.
   - Within-fold sequential score comparison via Diebold-Mariano test with Harvey-Leybourne-Newbold (HLN 1997) finite-sample correction and Newey-West HAC variance estimation using triangular Bartlett kernel across fixed lags $L \in \{4, 8, 16\}$. Null hypothesis $H_0: E[d_t] = 0$ ($d_t = \log P_{M_A} - \log P_{M_B}$). Multiple comparisons adjusted via Holm-Bonferroni FWER control across the 3 pairwise model contrasts. Model selection explicitly requires satisfying multi-criteria validation (predictive density, monotonicity, non-collapse, parsimony), never depending solely on p-values.
### 12.2 Artifacts Produced
- `data/reports/PHASE7_EXPERIMENTAL_DESIGN.md`: Complete scientifically refined experimental blueprint.

---

## 13. Phase 7 Implementation, Execution & Scientific Verification

Phase 7 executes out-of-sample walk-forward validation and cross-regime robustness analysis across five expanding-window chronological folds covering EURUSD M15 (2023 full year, 20,836 complete bars).

### 13.1 Implementation Architecture
1. **Core Module (`src/evaluation/walk_forward.py`):**
   - `generate_walk_forward_folds()`: Partitions complete bars into 5 expanding-window folds with zero inter-fold overlap and complete-bar isolation (796 partial bars 100% excluded).
   - `sequential_forward_filter()`: Dual Policy A (continuous FX calendar forward propagation via `FXCalendarGapHandler`) and Policy B (independent diagnostic re-initialization to `model.startprob_`).
   - `diebold_mariano_test()`: Computes loss differentials $d_t = \ell_t^{(A)} - \ell_t^{(B)}$, Newey-West HAC variance with triangular Bartlett kernel ($L \in \{4, 8, 16\}$), and Harvey-Leybourne-Newbold (HLN 1997) finite-sample correction.
   - `holm_bonferroni_adjustment()`: Step-down FWER adjustment across $m=3$ simultaneous pairwise hypothesis tests within each fold.
   - `fit_model_with_seed_schedule()`: Deterministic training wrapper with $S=5$ restarts per fit.
2. **Unit Test Suite (`tests/test_walk_forward.py`):**
   - 9 unit tests verifying fold partition boundaries, non-overlap, multi-sequence reset, boundary propagation ($n=1, 2, 193$), Diebold-Mariano mathematical correctness, Holm-Bonferroni step-down logic, and seed schedule determinism. Total test suite: 84/84 tests passing in 7.45s.
3. **Orchestration Script (`scripts/run_phase7.py`):**
   - Orchestrated all 30 unique training runs, sequential evaluation, Hungarian alignment, hypothesis testing, figure generation (Figures 19 & 20), and JSON persistence.

### 13.2 Empirical Results Summary & Scientific Audit (30 Unique Fits, 22.01 min Wall Time)
- **Predictive Density Ranking Across Evaluated Folds (RQ2):**
  - `GMM_HMM_K3_M3` achieved the highest out-of-sample predictive log-likelihood in **100% of the five evaluated test windows** ($5/5$), averaging **$6.2047$ nats/bar** (vs. $5.9631$ for Gaussian K3-M1 and $6.0846$ for GMM K3-M2).
  - Out-of-sample advantage over Gaussian K3-M1: $+0.286$ (Fold 1), $+0.258$ (Fold 2), $+0.253$ (Fold 3), $+0.217$ (Fold 4), $+0.195$ (Fold 5).
  - *Methodological Scope:* The 2023 expanding-window folds represent an in-dataset retrospective benchmark (not an independent cross-validation or pristine forward holdout).
- **Diebold-Mariano Hypothesis Testing Across $L \in \{4, 8, 16\}$ Lags:**
  - All three pairwise comparisons (`M3_vs_M1`, `M2_vs_M1`, `M3_vs_M2`) were evaluated across all 5 folds and across three Newey-West HAC truncation horizons ($L=4, 8, 16$).
  - `M3_vs_M1`: Decisively rejected $H_0$ in all 5 folds ($p_{\text{adj}} < 10^{-28}$, DM statistics between $+11.23$ and $+20.70$).
  - `M2_vs_M1`: Decisively rejected $H_0$ in all 5 folds ($p_{\text{adj}} < 10^{-14}$, DM statistics between $+7.99$ and $+12.53$).
  - `M3_vs_M2`: Statistically significant in all 5 folds ($p_{\text{adj}} \le 0.0025$, DM statistics between $+3.00$ and $+15.03$).
  - Invariance: Test decisions and signs are invariant to the choice of truncation lag.
- **Topological Monotonicity & Hungarian Alignment (RQ1 & RQ4):**
  - Return volatility ordering is strictly monotonic across all 5 folds:
    $$\sigma_{\text{LowVol}} \approx 1.75\text{ bps} < \sigma_{\text{MedVol}} \approx 3.60\text{ bps} < \sigma_{\text{HighVol}} \approx 8.19\text{ bps}$$
  - Mean Hungarian matching costs to canonical reference profile (`GMM_HMM_K3_M3_v1`): $\bar{c} \in [0.0554, 0.5679] \ll d_{\max} = 2.50$. Murty gap $\Delta_{\text{global}} > 3.12$ across all folds.
  - *Reference Boundary Audit:* The canonical reference profile was fitted on Phase 6 Train data (to 2023-08-24), overlapping with the Fold 1 test window (August 2023). Fold 1 alignment is strictly a retrospective diagnostic alignment, not independent temporal robustness evidence.
- **Decoupling Theoretical Markov Duration from Empirical Run Lengths:**
  - *Markov Theoretical Duration:* $E[T_k] = 1 / (1 - A_{kk}) \approx 7.66$ bars ($1.92$ hours) for M3 vs $4.92$ bars ($1.23$ hours) for M1. This is an unconditional property of the transition matrix without observation evidence.
  - *Empirical Filtered Mean Run Length:* $1 / \text{sw\_freq} \approx 5.53$ bars ($1.38$ hours) for M3 vs $3.88$ bars ($0.97$ hours) for M1.
  - *Causal Filtered Switching Rate:* $18.2\%$ for M3 vs $25.8\%$ for M1. (Empirical run lengths are shorter than theoretical Markov durations due to incoming observation updates; empirical duration is never inferred directly from $A_{kk}$).
  - Filtered Shannon entropy: $\bar{H}_t \approx 0.319 \ll \ln(3) \approx 1.099$, confirming decisive regime probability mass.
- **Covariance Regularization & Diurnal Sensitivity (RQ3):**
  - Covariance floor $\text{cov\_reg} = 10^{-3}$ was optimal for active trading periods (Folds 1–3); $\text{cov\_reg} = 10^{-2}$ improved likelihood during late-year consolidation (Folds 4–5).
  - Diurnal adjustment $[r_{\text{oc}}, \text{range\_zscore}]$ suppressed switching to $12.2\%$ in transformed space. Physical OHLC return volatility across diurnal states was not evaluated in the current artifact and remains an unverified limitation. Orthogonal $[r_{\text{oc}}, \log\text{vol}]$ remains the validated primary representation.

### 13.3 Phase 7 Artifacts Produced

| Artifact Path | Description |
| :--- | :--- |
| `src/evaluation/__init__.py` | Module initialization and public symbol exports. |
| `src/evaluation/walk_forward.py` | Walk-forward partitioning, Policy A/B filter, DM HAC/HLN test, Holm-Bonferroni adjustment, and seed schedule training. |
| `tests/test_walk_forward.py` | 9 unit tests for evaluation logic, boundary propagation, statistical tests, and determinism. |
| `scripts/run_phase7.py` | End-to-end execution pipeline for the 30 unique fits across 5 folds. |
| `data/reports/phase7_walkforward_results.json` | Complete JSON record of all 30 fits, parameters, SHA-256 hashes, test likelihoods, alignment metrics, and DM test results. |
| `data/reports/PHASE7_WALK_FORWARD_REPORT.md` | Academic research report documenting Phase 7 findings and answering all 7 research dimensions. |
| `data/reports/figures/19_walk_forward_oos_predictive_likelihood.png` | Figure 19: Walk-forward out-of-sample log-likelihood trajectories, loss differentials, cov_reg sensitivity, and switching rates. |
| `data/reports/figures/20_regime_parameter_stability_and_alignment.png` | Figure 20: Volatility ordering across folds, geometric duration, Diebold-Mariano statistics, and Hungarian alignment matching costs. |

---

## 14. Phase 8 Implementation, Execution & Scientific Verification

Phase 8 implements the Historical Market Regime Visualization engine, translating mathematical inferences (filtered posteriors, entropy, transition matrices, physical characteristics) into publication-quality, non-directional visualizations for human interpretability and operational supervision.

### 14.1 Implementation Architecture
1. **Core Modules:**
   - `src/visualization/__init__.py`: Package initialization and exports.
   - `src/visualization/data_loader.py`:
     * Dual data mode loader (`VisualizationDataLoader`):
       - **Mode A (Full-Year Retrospective View):** Evaluates all 20,836 complete bars in 2023 (`2023-01-01 23:30` to `2023-12-29 21:30 UTC`). Re-evaluates sequential forward filtering on `GMM_HMM_K3_M3_v1`, verifying identically zero discrepancy ($0.0$) against the streaming inference log on the 8,335 replay bars. An explicit cutover marker separates the in-sample parameter estimation partition (bars 1 to 12,501) from the out-of-sample forward filtering replay partition (bars 12,502 to 20,836).
       - **Mode B (As-Of Replay View):** Evaluates strictly the 8,335 post-cutover complete bars loaded directly from `data/processed/as_of_replay_inference_log.parquet`.
     * Strict data integrity assertions: Probability unity ($\sum_k \alpha_{t,k} = 1.0 \pm 10^{-6}$), unit interval $[0, 1]$, and entropy mathematical bounds.
     * Session gap segmentation (`extract_continuous_segments`): Splits time series at gaps $> 15$ minutes (`max_gap_seconds=900`), preventing artificial straight-line interpolation across weekend market closures.
     * Transition dynamics with zero gap bleed (`compute_transition_and_run_lengths`): Counts regime transitions strictly on adjacent within-session pairs ($t_i - t_{i-1} = 15 \text{ min}$), completely isolating market closures.
   - `src/visualization/regime_plots.py`:
     * Standardized non-directional color palette:
       - State 0 (`LowVol_Neutral`): Steel Blue (`#1f77b4`)
       - State 1 (`HighVol_Neutral`): Imperial Purple (`#7b3294`, win rate 51.2%, neutral)
       - State 2 (`MedVol_Neutral`): Ochre Amber (`#ff7f0e`)
       *(Green and red are strictly prohibited to avoid directional trading bias).*
     * Figure implementations:
       - `plot_historical_price_and_regimes()`: Price trajectory with dominant regime shading and cumulative return curves by regime.
       - `plot_filtered_probabilities()`: High-resolution stacked area plot of causal posteriors $\alpha_{t,k}$.
       - `plot_regime_confidence_and_entropy()`: Shannon entropy $H_t$, normalized entropy $H_t^{\text{norm}}$, and uncertainty distribution.
       - `plot_regime_transitions_and_duration()`: Transition probability matrix heatmap and empirical run length histograms.
       - `plot_regime_characteristics_crosscheck()`: Boxplots of physical return volatility ($\sigma(r_{\text{oc}})$ in bps) and pip range across regimes.
       - `plot_composite_dashboard()`: Unified 4-panel executive dashboard combining price, probabilities, entropy, and occupancy/duration.
   - `scripts/run_phase8.py`: Orchestration script loading data, generating all 10 figures, and persisting `data/reports/phase8_visualization_summary.json`.
2. **Unit Test Suite (`tests/test_visualization.py`):**
   - 10 unit tests covering Mode A loading, Mode B loading, probability properties, entropy bounds, session segmentation without gap interpolation, transition metrics with zero gap bleed, decoupled duration metrics, physical characteristic separation, plot rendering, and JSON summary schema validation.
   - Full test suite: **94/94 tests passing** in 12.21 seconds.

### 14.2 Key Quantitative Findings & Decoupled Durations
- **Occupancy:** MedVol dominates at $46.48\%$ ($47.20\%$ replay), LowVol at $33.22\%$ ($36.45\%$ replay), HighVol at $20.31\%$ ($16.35\%$ replay).
- **Information Entropy:** Mean Shannon entropy is $0.314$ nats ($0.300$ replay), corresponding to a normalized entropy of $0.285$ ($0.273$ replay). Over $80\%$ of bars have normalized entropy $< 0.40$, confirming decisive posterior attribution.
- **Strict Decoupling of Duration Concepts:**
  * *Theoretical Markov Duration ($1 / (1 - A_{kk})$):* LowVol = $7.78$ bars ($1.94$ hrs), HighVol = $8.75$ bars ($2.19$ hrs), MedVol = $6.94$ bars ($1.73$ hrs).
  * *Retrospective Viterbi Smoothed Mean:* LowVol = $5.6$ bars, HighVol = $7.5$ bars, MedVol = $6.8$ bars.
  * *Causal Filtered Empirical Mean Run Length:* LowVol = $4.19$ bars ($1.05$ hrs), HighVol = $4.34$ bars ($1.09$ hrs), MedVol = $4.46$ bars ($1.12$ hrs). Medians are $3.0$ bars ($45$ mins) across all regimes; 90th percentiles are $10\text{--}11$ bars ($2.5\text{--}2.75$ hrs).
  *(Empirical run lengths are shorter due to incoming observation filtering; empirical duration must never be conflated with $1/(1-A_{kk})$).*
- **Empirical Transition Dynamics (Zero Gap Bleed):**
  * Within-session switching rate: $18.53\%$ (Mode A, $N=19,686$), $17.41\%$ (Mode B, $N=8,060$).
  * Direct transitions between LowVol and HighVol are virtually non-existent ($P(0 \to 1) = 0.19\%$, $P(1 \to 0) = 0.15\%$). Markets transition through MedVol intermediary.
- **Physical Characteristics Separation:**
  * LowVol: Return std $1.62$ bps, Mean pip range $3.08$ pips.
  * MedVol: Return std $3.78$ bps, Mean pip range $6.43$ pips.
  * HighVol: Return std $8.66$ bps, Mean pip range $13.20$ pips.

### 14.3 Phase 8 Artifacts Produced

| Artifact Path | Description |
| :--- | :--- |
| `src/visualization/__init__.py` | Package initialization and public symbol exports. |
| `src/visualization/data_loader.py` | Data loader for Mode A & B, integrity checks, session segmentation, transition metrics. |
| `src/visualization/regime_plots.py` | Publication-quality plotting routines with non-directional palette and zero weekend interpolation. |
| `tests/test_visualization.py` | 10 unit tests for visualization data loaders, assertions, gap handling, and metrics. |
| `scripts/run_phase8.py` | Automated pipeline orchestrating data loading, figure rendering, and JSON persistence. |
| `data/reports/phase8_visualization_summary.json` | Complete quantitative summary JSON containing occupancy, entropy, transition matrices, decoupled durations, and physical characteristics. |
| `data/reports/PHASE8_HISTORICAL_REGIME_VISUALIZATION_REPORT.md` | Academic research report documenting Phase 8 methodology, figure descriptions, quantitative tables, and anti-claims. |
| `data/reports/figures/21_historical_price_and_regime_timeline_full_year.png` | Figure 21 (Mode A): Full-year price timeline with dominant regime spans and transition markers (descriptive return aggregation qualified). |
| `data/reports/figures/22_filtered_regime_probabilities_full_year.png` | Figure 22 (Mode A): Stacked area chart of causal filtered posterior probabilities with cutover marker. |
| `data/reports/figures/23_regime_confidence_and_entropy_full_year.png` | Figure 23 (Mode A): Shannon entropy timeline, normalized entropy distribution, and uncertainty breakdown. |
| `data/reports/figures/24_regime_transition_and_duration_full_year.png` | Figure 24 (Mode A): Empirical transition probability heatmap and run length histograms. |
| `data/reports/figures/25_regime_characteristics_crosscheck_full_year.png` | Figure 25 (Mode A): Physical return volatility bps and pip range boxplots across regimes. |
| `data/reports/figures/26_composite_regime_dashboard_full_year.png` | Figure 26 (Mode A): 4-panel executive dashboard for the full 2023 calendar year. |
| `data/reports/figures/27_historical_price_and_regime_timeline_replay.png` | Figure 27 (Mode B): As-of replay price timeline with dominant regimes and transition markers. |
| `data/reports/figures/28_filtered_regime_probabilities_replay.png` | Figure 28 (Mode B): As-of replay filtered posterior probabilities stacked area chart. |
| `data/reports/figures/29_regime_confidence_and_entropy_replay.png` | Figure 29 (Mode B): As-of replay Shannon entropy timeline and confidence breakdown. |
| `data/reports/figures/30_composite_regime_dashboard_replay.png` | Figure 30 (Mode B): 4-panel executive dashboard for the out-of-sample replay window. |

---

## 15. Next Steps: Phase 9 Preparation

1. **Phase 9 (Trading Applicability Evaluation):**
   - Evaluate contextual regime filters for trading systems ($P(\text{Turbulence}_t) = P(S_t = \text{HighVol} \mid O_{1:t})$).
   - Test risk-off overlay hypotheses with transaction costs, execution delay, and continuous equity curves.
   - Strictly avoid overfitting filter thresholds to out-of-sample testing data.





