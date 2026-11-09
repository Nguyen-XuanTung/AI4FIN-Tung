# Phase 9 — Trading applicability

## Frozen design
The highest train-only soft-weighted return standard deviation defines turbulence
(state 1). Filtered probabilities from the Phase 5 replay are used.
The prior candle's body sign provides a simple momentum signal at candle close.
Enter at the immediately following candle open and exit at that candle close.
Signals are canceled across gaps or incomplete candles. Each split starts flat.
This is a one-bar hypothesis, with separate round-trip costs on every entry.
Position: 10,000 EUR; initial cash: USD 10,000; no compounding or overnight holdings.

Threshold candidates are 0.25, 0.50, 0.75. Validation net profit selects
**0.25**, then this threshold is frozen for test and all cost scenarios.
Ties select the lower threshold. No test optimization is performed.

## Costs and execution limitations
Base spread: 0.1 pip; commission: USD 3.50 per standard lot per side;
slippage assumption: 0.1 pip per side. Total round trip for 0.1 lot: USD 1.00.
Spread/commission reference: [IC Markets EU Raw Spread account](https://www.icmarkets.eu/en/trading-accounts/raw-spread-account),
accessed 2026-11-09. These are current indicative terms, not reconstructed 2023 quotes.
Slippage is an explicit scenario assumption, not an observed broker statistic.
HistData OHLC is treated as an indicative reference price; bid/ask and exact
execution latency are unavailable. Same-boundary next-open fills are idealized.
Costs are stress-tested at 1x/2x/3x without retuning. No swaps because exits occur
within the entry bar. Execution and spread variability require independent validation.

## Test results
| Metric | Baseline | Turbulence filter |
| --- | ---: | ---: |
| Trades | 3988 | 3140 |
| Net P&L (USD) | -3916.60 | -3270.90 |
| Return (%) | -39.166 | -32.709 |
| Continuous close-equity max drawdown (%) | 39.267 | 32.812 |
| Bar ES95 (% initial cash) | 0.1161 | 0.0908 |

VaR/ES use all test bars, including zero-exposure bars. Drawdown includes initial
cash and every test close, but does not measure intrabar drawdown. See the JSON
for validation scores, period boundaries, input hashes and cost sensitivity.

## Applicability assessment
This evaluates contextual risk filtering, not directional prediction by HMM states.
Reduced exposure can reduce both loss and opportunity; comparison with cash and
exposure-matched controls is necessary before attributing benefits to regimes.
2023 has already informed model research: this is a retrospective benchmark,
not a pristine holdout or evidence of deployable profitability. A single base
strategy and fixed size cannot establish general trading value. Multi-year
holdout, broker execution data and additional predeclared strategies remain future work.
