# EURUSD M15 Market Regime Detection Using Hidden Markov Models

## Phase 1: Data Acquisition & Quality Control

### Overview
This package implements Phase 1 of `WORK_PLAN.md`:
- **Acquires raw historical data:** Automated, reproducible download of EURUSD 1-minute historical data from HistData, stored unmodified in `data/raw/`.
- **Timezone Standardization:** Converts raw timestamps from fixed Eastern Standard Time (EST, UTC-5 without DST changes) to standard UTC.
- **Strict Resampling (M1 to M15):** Aggregates 1-minute bars into 15-minute bars (`open: first`, `high: max`, `low: min`, `close: last`).
- **Zero Candle Fabrication:** Inactive periods (weekends, holidays, halts) are never interpolated or forward-filled. Gaps are preserved as authentic market interruptions.
- **Data Quality Audit:** Runs comprehensive integrity checks (monotonic timestamps, duplicate checks, OHLC validity $H \ge \max(O,C)$, $L \le \min(O,C)$, $H \ge L$, price $> 0$) and classifies all time gaps into 4 distinct categories per `WORK_PLAN.md`:
  - **Class 1:** Single missing 15m bar during active trading session.
  - **Class 2:** Multiple consecutive missing 15m bars during active trading session.
  - **Class 3:** Regular market closures (Weekends & global FX holidays).
  - **Class 4:** Invalid/corrupted OHLC records.
- **Reporting:** Emits machine-readable JSON (`data/reports/quality_report_*.json`) and Markdown audit reports (`data/reports/DATA_QUALITY_REPORT_*.md`).

---

### Project Structure
```
AI4FIN-Tung/
├── data/
│   ├── raw/                  # Unmodified raw data archives (DAT_ASCII_EURUSD_M1_*.zip)
│   ├── processed/            # Cleaned M15 datasets (.parquet and .csv)
│   └── reports/              # Data quality audit reports (.json and .md)
├── papers/                   # 4 primary research reference papers
├── scripts/
│   └── run_phase1.py         # End-to-end CLI runner for Phase 1
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── acquire.py        # Reproducible download and raw loading
│       ├── resample.py       # Strict aggregation without fabrication
│       └── quality.py        # Validation and gap classification
├── tests/
│   ├── __init__.py
│   └── test_quality.py       # Pytest unit tests for quality checks & resampling
├── WORK_PLAN.md              # Master research plan
└── README.md                 # Project documentation
```

---

### How to Run

#### 1. Run Unit Tests
```powershell
python -m pytest -v tests/
```

#### 2. Run Phase 1 Data Pipeline (Default: EURUSD 2023)
```powershell
python scripts/run_phase1.py --pair eurusd --year 2023
```

#### 3. Custom Arguments
```powershell
python scripts/run_phase1.py --pair eurusd --year 2022 --format parquet
```
Options:
- `--pair`: Currency pair (default: `eurusd`).
- `--year`: Calendar year to process (e.g. `2023`).
- `--raw-dir`: Directory for raw downloads (default: `data/raw`).
- `--processed-dir`: Directory for clean M15 data (default: `data/processed`).
- `--reports-dir`: Directory for audit reports (default: `data/reports`).
- `--format`: Output format: `parquet`, `csv`, or `both` (default: `both`).

