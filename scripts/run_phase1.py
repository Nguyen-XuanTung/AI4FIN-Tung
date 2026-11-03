"""End-to-End Pipeline for Phase 1: Data Acquisition & Quality Control.

Usage:
    python scripts/run_phase1.py --year 2023 --pair eurusd
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path so 'src' is importable directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.acquire import acquire_histdata_year, load_raw_histdata_csv
from src.data.resample import resample_m1_to_m15
from src.data.quality import check_data_quality, generate_quality_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Phase1Runner")


def run_phase_1(
    pair: str = "eurusd",
    year: int = 2023,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    reports_dir: str = "data/reports",
    export_format: str = "both",
) -> None:
    """Execute complete Phase 1 pipeline."""
    pair_clean = pair.lower()
    symbol = pair.upper()
    prefix = f"{symbol}_M15_{year}"

    logger.info("=============================================================")
    logger.info("STARTING PHASE 1: DATA ACQUISITION & QUALITY CONTROL")
    logger.info("Symbol: %s | Year: %d | Timeframe: M15", symbol, year)
    logger.info("=============================================================")

    # Step 1: Acquire raw data (idempotent, raw archive remains untouched)
    raw_zip_path = acquire_histdata_year(
        pair=pair_clean,
        year=year,
        raw_dir=raw_dir,
        force_download=False,
    )

    # Step 2: Load raw M1 data & convert fixed EST (UTC-5) to UTC
    df_m1 = load_raw_histdata_csv(raw_zip_path)

    # Step 3: Resample M1 to M15 (strict zero-fabrication)
    df_m15 = resample_m1_to_m15(df_m1)

    # Step 4: Perform comprehensive Data Quality Control checks
    report = check_data_quality(df_m15, symbol=symbol, timeframe="M15")

    # Step 5: Save processed clean M15 dataset
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    if export_format in ["parquet", "both"]:
        parquet_file = proc_path / f"{prefix}.parquet"
        df_m15.to_parquet(parquet_file, index=False)
        logger.info("Saved processed dataset (Parquet): %s (%d rows)", parquet_file, len(df_m15))

    if export_format in ["csv", "both"]:
        csv_file = proc_path / f"{prefix}.csv"
        df_m15.to_csv(csv_file, index=False)
        logger.info("Saved processed dataset (CSV): %s (%d rows)", csv_file, len(df_m15))

    # Step 6: Generate JSON & Markdown Quality Reports
    json_path, md_path = generate_quality_report(
        report=report,
        output_dir=reports_dir,
        prefix=prefix,
    )

    logger.info("=============================================================")
    logger.info("PHASE 1 EXECUTION SUMMARY:")
    logger.info("  - Total M15 Bars Generated: %d", report.total_bars)
    logger.info("  - Start Timestamp (UTC):     %s", report.start_time)
    logger.info("  - End Timestamp (UTC):       %s", report.end_time)
    logger.info("  - Monotonic Timestamp Order: %s", "PASSED" if report.is_monotonic_increasing else "FAILED")
    logger.info("  - Duplicate Timestamps:      %d", report.duplicate_timestamps)
    logger.info("  - Invalid OHLC Bars:         %d", report.invalid_ohlc_count)
    logger.info("  - Total Gaps Detected:       %d", report.total_gaps)
    logger.info("    * Class 1 (Single Bar):    %d", report.class_1_gaps_count)
    logger.info("    * Class 2 (Multi-Bar):     %d", report.class_2_gaps_count)
    logger.info("    * Class 3 (Weekend):       %d", report.class_3_weekend_gaps_count)
    logger.info("    * Class 3 (Holiday):       %d", report.class_3_holiday_gaps_count)
    logger.info("    * Class 4 (Invalid OHLC):  %d", report.class_4_invalid_count)
    logger.info("  - Quality Report (Markdown): %s", md_path)
    logger.info("  - Quality Report (JSON):     %s", json_path)
    logger.info("=============================================================")
    logger.info("PHASE 1 COMPLETE.")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: EURUSD M15 Data Acquisition and Quality Control")
    parser.add_argument("--pair", type=str, default="eurusd", help="Currency pair (e.g. eurusd)")
    parser.add_argument("--year", type=int, default=2023, help="Calendar year (e.g. 2023)")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--reports-dir", type=str, default="data/reports", help="Path to reports directory")
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["parquet", "csv", "both"],
        help="Export format for processed data",
    )

    args = parser.parse_args()
    run_phase_1(
        pair=args.pair,
        year=args.year,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        reports_dir=args.reports_dir,
        export_format=args.format,
    )


if __name__ == "__main__":
    main()

