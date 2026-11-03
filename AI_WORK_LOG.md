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

## 9. Next Steps for Phase 5
1. Incremental streaming inference engine consuming live/simulated tick-aggregated M15 candles.
2. Formalization of trading-calendar policy (weekend vs. holiday vs. missing bars in open sessions).
3. Immutable metadata versioning for "as-of" filtered probability logs.
4. Retraining warm-up protocol ensuring newly fitted models do not reuse stale prior vectors.
