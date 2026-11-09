"""Reproduce the frozen 2023 trading-applicability experiment."""
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.trading import TradingCosts, backtest, select_threshold
from src.visualization.data_loader import load_as_of_replay_data, load_canonical_model
from src.models.splits import split_chronological
import pandas as pd


def main():
    model = ROOT / "models/GMM_HMM_K3_M3_v1.json"
    artifact = load_canonical_model(str(model))
    profile_path = ROOT / "data/reports/regime_characteristics_train.json"
    profile = json.loads(profile_path.read_text())[artifact.model_version]
    state = int(max(profile["states"], key=lambda s: profile["states"][s]["return_r_oc"]["std"]))
    ohlc = ROOT / "data/processed/EURUSD_M15_2023.parquet"
    replay = ROOT / "data/processed/as_of_replay_inference_log.parquet"
    df = load_as_of_replay_data(str(replay), str(ohlc))
    if not df.model_version.eq(artifact.model_version).all():
        raise ValueError("Replay model version mismatch")
    splits = split_chronological(pd.read_parquet(ohlc))
    periods = {}
    for name in ("val", "test"):
        expected = pd.to_datetime(splits[name].timestamp, utc=True)
        periods[name] = df[df.timestamp.isin(expected)].copy()
        if len(periods[name]) != len(expected):
            raise ValueError(f"Incomplete {name} replay coverage")
    costs = TradingCosts()
    threshold, validation = select_threshold(periods["val"], state, costs)
    baseline, base_ledger = backtest(periods["test"], state, costs=costs)
    filtered, filtered_ledger = backtest(periods["test"], state, threshold, costs)
    stress = {}
    for multiplier in (1, 2, 3):
        c = TradingCosts(spread_pips=costs.spread_pips * multiplier,
                         commission_per_lot_side=costs.commission_per_lot_side * multiplier,
                         slippage_pips_side=costs.slippage_pips_side * multiplier)
        stress[str(multiplier)] = {"baseline": backtest(periods["test"], state, costs=c)[0],
                                  "filtered": backtest(periods["test"], state, threshold, c)[0]}
    result = dict(model_version=artifact.model_version, turbulence_state=state,
                  threshold=threshold, costs=asdict(costs), validation=validation,
                  test=dict(baseline=baseline, filtered=filtered), cost_sensitivity=stress,
                  periods={k: dict(start=str(v.timestamp.min()), end=str(v.timestamp.max()), bars=len(v)) for k,v in periods.items()},
                  input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (model, profile_path, ohlc, replay)})
    reports = ROOT / "data/reports"
    (reports / "phase9_trading_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    ledger = base_ledger.rename(columns={c: "baseline_" + c for c in base_ledger if c != "timestamp"})
    ledger = ledger.merge(filtered_ledger.rename(columns={c: "filtered_" + c for c in filtered_ledger if c != "timestamp"}), on="timestamp", validate="one_to_one")
    ledger.to_parquet(ROOT / "data/processed/phase9_test_equity.parquet", index=False)
    report = f"""# Phase 9 — Trading applicability

## Frozen design
The highest train-only soft-weighted return standard deviation defines turbulence
(state {state}). Filtered probabilities from the Phase 5 replay are used.
The prior candle's body sign provides a simple momentum signal at candle close.
Enter at the immediately following candle open and exit at that candle close.
Signals are canceled across gaps or incomplete candles. Each split starts flat.
This is a one-bar hypothesis, with separate round-trip costs on every entry.
Position: 10,000 EUR; initial cash: USD 10,000; no compounding or overnight holdings.

Threshold candidates are 0.25, 0.50, 0.75. Validation net profit selects
**{threshold}**, then this threshold is frozen for test and all cost scenarios.
Ties select the lower threshold. No test optimization is performed.

## Costs and execution limitations
Base spread: 0.1 pip; commission: USD 3.50 per standard lot per side;
slippage assumption: 0.1 pip per side. Total round trip for 0.1 lot: USD {costs.round_trip_usd:.2f}.
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
| Trades | {baseline['trades']} | {filtered['trades']} |
| Net P&L (USD) | {baseline['net_profit_usd']:.2f} | {filtered['net_profit_usd']:.2f} |
| Return (%) | {baseline['return_pct']:.3f} | {filtered['return_pct']:.3f} |
| Continuous close-equity max drawdown (%) | {baseline['max_drawdown_pct']:.3f} | {filtered['max_drawdown_pct']:.3f} |
| Bar ES95 (% initial cash) | {baseline['es95_bar_pct']:.4f} | {filtered['es95_bar_pct']:.4f} |

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
"""
    (reports / "PHASE9_TRADING_APPLICABILITY_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
