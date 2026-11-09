# EURUSD M15 Market Regime Research

Research prototype for Gaussian HMM and GMM-HMM regimes, causal inference,
walk-forward evaluation and contextual trading filters on EURUSD 2023.
Phases 1–10 are implemented; multi-year validation and live broker integration
remain future work.

## Install and verify

Validated with Python 3.14 and the versions in `requirements.txt`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
New-Item -ItemType Directory -Force tmp | Out-Null
python -m pytest -q --basetemp=tmp/pytest
```

On other systems, activate `.venv/bin/activate` and use `mkdir -p tmp`.
The explicit test directory avoids Windows temporary-directory permission issues.

## Reproduce results

Tracked Parquet data, model JSON, reports and figures allow offline review.
The default reproduces Phase 9 without downloads or model refitting:

```powershell
python scripts/run_pipeline.py
```

Regenerate historical figures and trading evaluation:

```powershell
python scripts/run_pipeline.py --start-phase 8 --end-phase 9
```

Rebuild all stages (downloads data, refits models and overwrites generated
artifacts; may take tens of minutes or longer):

```powershell
python scripts/run_pipeline.py --start-phase 1 --end-phase 9
```

The runner uses established EURUSD 2023 defaults and stops on the first failure.
Individual `scripts/run_phaseN.py` entry points remain available. Phase 10 is
integration/documentation, not a separate fitting stage. Use a fresh checkout
for full rebuilds: Phase 5 maintains an append-only local inference journal.
Changing only the year in Phase 1 does not generalize later stages, which contain
explicit 2023 experiment boundaries.

## Workflow and outputs

| Phase | Purpose | Report |
| --- | --- | --- |
| 1 | Acquisition, UTC conversion, gap-aware resampling | [Quality](data/reports/DATA_QUALITY_REPORT_EURUSD_M15_2023.md) |
| 2 | Features and train-only seasonality | [Features](data/reports/EDA_FEATURE_ENGINEERING_REPORT.md) |
| 3–4 | Gaussian and mixture HMM comparison | [Gaussian](data/reports/PHASE3_GAUSSIAN_HMM_BASELINE_REPORT.md), [GMM](data/reports/PHASE4_GMM_HMM_REPORT.md) |
| 5 | Causal replay and versioned storage | [Inference](data/reports/PHASE5_REALTIME_INFERENCE_REPORT.md) |
| 6 | Interpretation and label alignment | [Interpretation](data/reports/PHASE6_REGIME_INTERPRETATION_REPORT.md) |
| 7 | Walk-forward evaluation | [Evaluation](data/reports/PHASE7_WALK_FORWARD_REPORT.md) |
| 8 | Probabilities, entropy and dashboards | [Visualization](data/reports/PHASE8_HISTORICAL_REGIME_VISUALIZATION_REPORT.md) |
| 9 | Costed baseline vs turbulence filter | [Trading](data/reports/PHASE9_TRADING_APPLICABILITY_REPORT.md) |
| 10 | Integration and research synthesis | [Final report](data/reports/FINAL_RESEARCH_REPORT.md) |

`src/` contains reusable modules; `scripts/` experiment runners; `tests/` numerical
and temporal checks. `models/` retains distinct fitted candidates and versioned
artifacts. `papers/`, `WORK_PLAN.md` and `AI_WORK_LOG.md` preserve research provenance.

## Limitations

Regimes characterize volatility, not buy/sell directions. Real-time-style output
uses filtered probabilities; smoothing and full-sequence Viterbi are retrospective.
2023 already informed model research and is not a pristine holdout. Phase 9's
one-bar momentum hypothesis loses money after costs with and without filtering.
Reduced losses do not establish profitable trading value.

HistData's fixed UTC−5 interpretation still needs independent DST-boundary checks.
Costs use current indicative broker terms plus assumed slippage, not historical
bid/ask fills. Probabilities and entropy describe model-conditional uncertainty.
See the final report for remaining research questions.
