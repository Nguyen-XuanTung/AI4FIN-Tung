# Phase 1 Data Quality & Integrity Report: EURUSD M15

## 1. Dataset Overview
- **Symbol:** `EURUSD`
- **Timeframe:** `M15` (Left-closed, Left-labeled bar start)
- **Total Bars:** `21,632`
- **Sample Start (UTC):** `2023-01-01 22:00:00+00:00`
- **Sample End (UTC):** `2023-12-29 21:45:00+00:00`
- **Strict Monotonic Ordering:** `PASSED`
- **Duplicate Timestamps:** `0`

## 2. OHLC Integrity & Consistency
- **Invalid OHLC Records (H < L, H < max(O,C), L > min(O,C), price <= 0):** `0`
- **NaN / Null Values:** `0`

## 3. M15 Candle Completeness & Active M1 Bars
- **Complete Bars (`active_m1_bars == 15`, `is_complete == True`):** `20,836` (96.32%)
- **Partial Bars (`active_m1_bars < 15`, `is_complete == False`):** `796` (3.68%)

Partial bars are explicitly flagged with `is_complete = False` and strictly preserved without price fabrication or artificial interpolation.

### Distribution of Active M1 Bars per M15 Candle
| Active M1 Bars in 15m Window | Candle Count | Percentage | Classification |
| :--- | :--- | :--- | :--- |
| **1** / 15 bars | 3 | 0.01% | Partial |
| **2** / 15 bars | 5 | 0.02% | Partial |
| **3** / 15 bars | 3 | 0.01% | Partial |
| **4** / 15 bars | 3 | 0.01% | Partial |
| **5** / 15 bars | 9 | 0.04% | Partial |
| **6** / 15 bars | 13 | 0.06% | Partial |
| **7** / 15 bars | 12 | 0.06% | Partial |
| **8** / 15 bars | 13 | 0.06% | Partial |
| **9** / 15 bars | 18 | 0.08% | Partial |
| **10** / 15 bars | 24 | 0.11% | Partial |
| **11** / 15 bars | 48 | 0.22% | Partial |
| **12** / 15 bars | 63 | 0.29% | Partial |
| **13** / 15 bars | 141 | 0.65% | Partial |
| **14** / 15 bars | 441 | 2.04% | Partial |
| **15** / 15 bars | 20,836 | 96.32% | Complete |

### Price Distribution Summary
| Field | Min | Max | Mean | Std Dev |
| :--- | :--- | :--- | :--- | :--- |
| **OPEN** | 1.04524 | 1.12718 | 1.08077 | 0.01627 |
| **HIGH** | 1.04559 | 1.12756 | 1.08109 | 0.01627 |
| **LOW** | 1.04483 | 1.12643 | 1.08044 | 0.01628 |
| **CLOSE** | 1.04525 | 1.12718 | 1.08076 | 0.01628 |

## 4. Time Gap Analysis & Categorization
As mandated by Phase 1 of `WORK_PLAN.md`, all non-15-minute time gaps are categorized without fabricating candles:

| Gap Category | Occurrence Count | Missing 15m Bars | Policy / Interpretation |
| :--- | :--- | :--- | :--- |
| **Class 1: Single Missing Bar in Session** | 0 | 0 | Isolated 1-bar gap during open session; state propagation via $\boldsymbol{\alpha}_{t+1} = \boldsymbol{\alpha}_t A$ in Phase 5. |
| **Class 2: Multiple Missing Bars in Session** | 679 | 3071 | Consecutive missing bars; multi-step propagation $\boldsymbol{\alpha}_{t+n} = \boldsymbol{\alpha}_t A^n$. |
| **Class 3: Weekend Closures** | 51 | 9992 | Friday ~21:00 UTC to Sunday ~21:00 UTC. Handled by trading-calendar boundary policy. |
| **Class 3: Holiday Closures** | 2 | 57 | Global FX closures (Christmas, New Year). Handled by holiday calendar. |
| **Class 4: Invalid OHLC Records** | 0 | 0 | Malformed records flagged and quarantined. |
| **Total Non-Standard Intervals** | **732** | **13,120** | **Zero interpolation / zero artificial candle fabrication.** |

### In-Depth Investigation of Class 2 (Multi-Bar) Gaps
- **Empirical Duration Distribution:**
  * **75 minutes (4 missing 15m bars):** 591 occurrences (~87.0% of Class 2 gaps). These correspond to exact 1-hour gaps in raw feeds (from XX:59 to (XX+2):00).
  * **135 minutes (8 missing 15m bars):** 85 occurrences (~12.5% of Class 2 gaps). These correspond to exact 2-hour gaps.
  * **195 minutes (12 missing 15m bars):** 2 occurrences.
  * **60 minutes (3 missing 15m bars):** 1 occurrence.
- **Inspection of Surrounding Raw M1 Records:** Examination of tick and 1-minute records immediately before and after these missing intervals reveals observable price jumps (e.g. 3 to 10+ pips). This confirms that interbank FX trading continued during these missing intervals, but the data vendor feed omitted records.
- **Methodological Conclusion:** It is not assumed that missing records represent periods of low market activity or market closure. They represent authentic feed omissions during active trading sessions. The existing **Class 2 classification is retained** as required, and these intervals will be handled in Phase 5 via Markov chain transition-matrix propagation ($\boldsymbol{\alpha}_{t+n} = \boldsymbol{\alpha}_t A^n$) without synthetic price interpolation.

## 5. Notable Outliers / Extreme Volatility Spikes
Total bars exceeding 2.0% high-low range: `0`

No abnormal volatility spikes (>2.0% bar range) detected.

## 6. Compliance with WORK_PLAN.md Directives
- [x] **Zero Candle Fabrication:** Missing intervals are strictly preserved as gaps, not artificially filled.
- [x] **Timezone Standardization:** Standardized to UTC from fixed EST (UTC-5, without DST bias).
- [x] **Volume Transparency:** Documented that tick volume reflects OTC broker quote frequency, not centralized exchange volume.
- [x] **Data Separation:** Raw archives remain untouched in `data/raw/`; processed parquet/csv datasets stored in `data/processed/`; audit reports stored in `data/reports/`.
