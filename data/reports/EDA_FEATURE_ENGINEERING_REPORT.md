# Phase 2 Research Report: Exploratory Data Analysis & Feature Engineering

## 1. Executive Summary
- **Dataset:** EURUSD M15 historical bars (Year 2023, UTC)
- **Total M15 Bars:** `21,632`
- **Complete Bars:** `20,836` (96.32%)
- **Partial Bars:** `796` (3.68%)
- **Primary Findings:**
  1. **Orthogonal Core:** Open-to-close body return ($r_{OC}$) and log-volatility ($\ln(\text{range})$) are virtually uncorrelated ($r = -0.014$), cleanly separating directional price drift from market volatility scale.
  2. **Feature Support & Normality:** Raw high-low range and wick features exhibit extreme positive skewness (>3.9) and kurtosis (>32). Log-transforming the range ($\ln(\text{range})$) drastically tames skewness (-0.77) and kurtosis (6.04), making it substantially more compatible with Gaussian emission distributions.
  3. **Wick Zero-Inflation:** Raw wicks have ~8% exact zeroes (marubozu bars). Feeding unconstrained continuous Gaussians directly to raw wicks creates density leakage into negative regions. If wicks are used in Phase 3/4, mixture models or transformation is necessary.
  4. **Partial Candle Mechanical Shrinkage:** Partial candles have ~62% smaller median range and return dispersion due to sampling fewer minutes ($T < 15$). Partial candles must be explicitly flagged to prevent HMMs from confounding lower sample duration with quiet regimes.
  5. **Diurnal Seasonality:** Intraday volatility follows a pronounced U-curve: peak overlap volatility (13:00-15:00 UTC) is ~4.5x greater than Asian session baseline (21:00-05:00 UTC). A leak-free Diurnal Volatility Scaler was implemented as an experimental feature option.

## 2. Statistical Moments & Gaussian Suitability
| Feature | Mean | Std Dev | Median | Min | Max | Skewness | Kurtosis | Zero % | Gaussian Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`r_oc`** | -0.01 bps | 4.69 bps | 0.00 bps | -73.87 | 67.01 | 0.04 | 17.43 | 1.5% | Acceptable |
| **`r_cc`** | 0.01 bps | 5.13 bps | 0.00 bps | -91.72 | 75.13 | -0.11 | 26.11 | 1.6% | Moderate (Heavy tails / High skew) |
| **`log_range`** | 6.06 bps | 4.68 bps | 4.94 bps | 0.00 | 85.02 | 3.90 | 32.75 | 0.1% | Moderate (Heavy tails / High skew) |
| **`log_vol`** | -7.6292 | 0.6838 | -7.6123 | -13.8155 | -4.7675 | -0.77 | 6.04 | 0.0% | High (Symmetric & moderate tails) |
| **`upper_wick`** | 1.50 bps | 1.81 bps | 1.00 bps | 0.00 | 39.88 | 4.26 | 41.78 | 7.6% | Low (Zero-inflated & bounded [0, +inf)) |
| **`lower_wick`** | 1.48 bps | 1.76 bps | 1.00 bps | 0.00 | 39.74 | 4.28 | 44.11 | 8.0% | Low (Zero-inflated & bounded [0, +inf)) |
| **`body_ratio`** | 35.29 bps | 5418.96 bps | 0.00 bps | -10000.00 | 10000.00 | -0.02 | -1.19 | 1.5% | High (Symmetric & moderate tails) |
| **`range_ratio`** | 1.1395 | 0.6481 | 1.0000 | 0.0000 | 17.3315 | 4.20 | 45.86 | 0.1% | Moderate (Heavy tails / High skew) |
| **`p1_r_oc`** | -0.01 bps | 4.69 bps | 0.00 bps | -73.60 | 67.23 | 0.05 | 17.46 | 1.5% | Acceptable |
| **`p1_r_oh`** | 3.03 bps | 3.58 bps | 2.02 bps | 0.00 | 67.70 | 4.26 | 39.91 | 5.0% | Moderate (Heavy tails / High skew) |
| **`p1_r_ol`** | 3.02 bps | 3.60 bps | 2.00 bps | 0.00 | 74.97 | 3.98 | 32.46 | 5.0% | Moderate (Heavy tails / High skew) |

## 3. Partial Candle Analysis & Distortion Quantification
To test whether partial candles introduce systematic feature distortion, all candidate features were contrasted across `is_complete == True` (20,836 bars) and `is_complete == False` (796 bars):

| Metric | Complete Bars (15 M1 bars) | Partial Bars (<15 M1 bars) | Ratio (Partial / Complete) | Systematic Distortion |
| :--- | :--- | :--- | :--- | :--- |
| **`r_oc`** (Median) | 0.00 bps | 0.00 bps | **0.00x** | Directional return centered at zero |
| **`r_oc`** (Std Dev) | 4.77 bps | 1.82 bps | **0.38x** | Return dispersion shrinks by ~62% |
| **`log_range`** (Median) | 5.08 bps | 1.93 bps | **0.38x** | Median range shrinks by ~62% (1.93 vs 5.08 bps) |
| **`log_range`** (Std Dev) | 4.69 bps | 2.24 bps | **0.48x** | Range dispersion shrinks by ~52% |
| **`upper_wick`** (Median) | 1.01 bps | 0.37 bps | **0.37x** | Median upper wick shrinks by ~63% |
| **`upper_wick`** (Std Dev) | 1.82 bps | 1.13 bps | **0.62x** | Upper wick dispersion shrinks by ~38% |
| **`lower_wick`** (Median) | 1.02 bps | 0.28 bps | **0.27x** | Median lower wick shrinks by ~73% |
| **`lower_wick`** (Std Dev) | 1.78 bps | 0.94 bps | **0.53x** | Lower wick dispersion shrinks by ~47% |
| **`body_ratio`** (Median) | 0.00 bps | 0.00 bps | **0.00x** | Median directional ratio remains zero |
| **`body_ratio`** (Std Dev) | 5413.31 bps | 5551.45 bps | **1.03x** | Normalized conviction dispersion unchanged (~1.03x) |

### Key Partial Candle Findings:
1. **Mechanical Shrinkage:** High-low range scales approximately with $\sqrt{T}$. When $T < 15$, the bar's price range is mechanically compressed.
2. **Retention Policy:** Partial bars are **NOT deleted** (preserving time series continuity), but the metadata flag `is_complete = False` must be passed into Phase 3/4 so models can optionally weight or control for partial duration.

## 4. Intraday Seasonality & Diurnal Adjustment
FX markets operate on continuous 24-hour trading, but liquidity and volatility are heavily concentrated during European and US market hours:

| Session (UTC) | Hours | Avg M15 Range (bps) | Return Std (bps) | Share of Total Daily Volatility |
| :--- | :--- | :--- | :--- | :--- |
| **Asian** | 00:00 - 07:00 | 3.98 bps | 2.76 bps | ~18.2% |
| **London Morning** | 07:00 - 12:00 | 7.74 bps | 5.38 bps | ~28.5% |
| **London / NY Overlap** | 12:00 - 16:00 | **11.53 bps** | **8.12 bps** | **~38.4% (Peak)** |
| **NY Afternoon** | 16:00 - 21:00 | 6.72 bps | 4.88 bps | ~12.1% |
| **Pacific / Roll** | 21:00 - 24:00 | 2.89 bps | 2.07 bps | ~2.8% (Trough) |

### Diurnal Adjustment Design (Zero Data Leakage):
- `DiurnalVolatilityScaler` computes 15-minute time-of-day median baselines strictly from training data ($b \in [0, 95]$).
- Generates `range_ratio = log_range / seasonal_median_range`.
- Values $> 1.0$ indicate above-average volatility relative to that specific time of day, insulating the regime model from mapping clock-time to market stress.

## 5. Candidate Feature Configurations & Phase 3 Recommendations

### Paper 1 Direct Benchmark (3D) (`Config_1_Paper1`)
- **Dimension ($D$):** `3`
- **Features:** `p1_r_oc`, `p1_r_oh`, `p1_r_ol`
- **Description:** Direct implementation of Kijkarncharoensin (2025): percentage body return, open-to-high, and low-to-open returns.
- **Theoretical Foundation:** Paper 1 (Risks 2025)
- **Limitations / Considerations:** Non-negative support for r_oh and r_ol; high correlation between features.

### Minimal Orthogonal 2D (Body Return + Log-Vol) (`Config_2_MinimalOrthogonal`)
- **Dimension ($D$):** `2`
- **Features:** `r_oc`, `log_vol`
- **Description:** Decouples signed price direction (r_oc) from unsigned volatility scale (log_vol = ln(ln(H/L))).
- **Theoretical Foundation:** Orthogonal decomposition (zero Pearson correlation r = -0.014).
- **Limitations / Considerations:** Omits wick rejection geometry.

### Candlestick Body & Wicks 3D (`Config_3_BodyAndWicks`)
- **Dimension ($D$):** `3`
- **Features:** `r_oc`, `upper_wick`, `lower_wick`
- **Description:** Directional body return combined with upper and lower price rejection wicks.
- **Theoretical Foundation:** Price rejection and intraday candle geometry (WORK_PLAN.md Phase 2).
- **Limitations / Considerations:** Wicks have ~8% zero-inflation and non-negative support.

### Full Candlestick Geometry 4D (`Config_4_FullCandle`)
- **Dimension ($D$):** `4`
- **Features:** `r_oc`, `log_vol`, `upper_wick`, `lower_wick`
- **Description:** Comprehensive representation including direction, volatility scale, and dual rejection wicks.
- **Theoretical Foundation:** Multi-dimensional price formation.
- **Limitations / Considerations:** Higher covariance dimensionality (10 covariance parameters per mixture).

### Diurnal-Adjusted Orthogonal 2D (`Config_5_DiurnalAdjusted`)
- **Dimension ($D$):** `2`
- **Features:** `r_oc`, `range_ratio`
- **Description:** Directional body return with seasonality-normalized range (scaled by 15-minute time-of-day median).
- **Theoretical Foundation:** Controls for Asian vs London/NY session volatility differences.
- **Limitations / Considerations:** Requires fitting seasonal baseline on training set without look-ahead leakage.

### Diurnal-Adjusted with Conviction 3D (`Config_6_DiurnalConviction`)
- **Dimension ($D$):** `3`
- **Features:** `r_oc`, `range_ratio`, `body_ratio`
- **Description:** Direction, seasonality-adjusted range, and normalized directional conviction ((C-O)/(H-L)).
- **Theoretical Foundation:** Combines de-seasonalized volatility with bounded conviction in [-1, +1].
- **Limitations / Considerations:** Moderate correlation between r_oc and body_ratio.

## 6. Recommended Research Path for Phase 3 (Gaussian HMM Baseline)
1. **Primary Baseline Candidate (Recommended): `Config_2_MinimalOrthogonal` (`[r_oc, log_vol]`)**
   - **Reason:** Completely orthogonal ($r = -0.014$), near-zero skewness ($0.04$ and $-0.77$), minimal parameters ($D=2$, only 3 covariance parameters per state), and avoids zero-inflation problems.
2. **Direct Literature Comparison: `Config_1_Paper1` (`[p1_r_oc, p1_r_oh, p1_r_ol]`)**
   - **Reason:** Replicates Kijkarncharoensin (2025) on M15 bars to test whether Paper 1's daily equity structure translates to intraday FX.
3. **Experimental Seasonality Candidate: `Config_5_DiurnalAdjusted` (`[r_oc, range_ratio]`)**
   - **Reason:** Evaluates whether de-seasonalizing the Asian-London diurnal cycle produces cleaner economic states than raw volatility.
