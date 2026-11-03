"""Data acquisition for EURUSD M1/M15 historical data.

Provides reproducible downloading from HistData and loading of raw historical data.
Raw downloaded files are kept strictly unmodified in data/raw/.
"""

import io
import logging
from datetime import timedelta, timezone
from pathlib import Path
from typing import Optional, Union
import zipfile
import pandas as pd
import histdata

logger = logging.getLogger(__name__)

# HistData timestamps are in Eastern Standard Time (EST) all year round (no DST shifts).
# EST = UTC - 5 hours.
HISTDATA_EST_TZ = timezone(timedelta(hours=-5))


def acquire_histdata_year(
    pair: str = "eurusd",
    year: int = 2023,
    raw_dir: Union[str, Path] = "data/raw",
    force_download: bool = False,
) -> Path:
    """Download raw 1-minute historical data for a given year from HistData.

    The raw archive is stored unmodified in `raw_dir`.

    Args:
        pair: Currency pair identifier (e.g. 'eurusd').
        year: Full calendar year (e.g. 2023).
        raw_dir: Target directory for raw downloaded files.
        force_download: If True, re-download even if archive already exists.

    Returns:
        Path to the downloaded raw zip file.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    expected_filename = f"DAT_ASCII_{pair.upper()}_M1_{year}.zip"
    target_file = raw_path / expected_filename

    if target_file.exists() and not force_download:
        logger.info("Raw data already exists at %s, skipping download.", target_file)
        return target_file

    logger.info("Downloading %s M1 data for year %d from HistData...", pair.upper(), year)
    downloaded_path = histdata.download_hist_data(
        year=str(year),
        month=None,
        pair=pair.lower(),
        time_frame="M1",
        platform="ASCII",
        output_directory=str(raw_path),
        verbose=False,
    )

    zip_file = Path(downloaded_path)
    if zip_file.name != expected_filename:
        # Standardize archive filename if needed
        dest = raw_path / expected_filename
        if dest.exists():
            dest.unlink()
        zip_file.rename(dest)
        target_file = dest
    else:
        target_file = zip_file

    logger.info("Successfully acquired raw archive: %s (size: %d bytes)", target_file, target_file.stat().st_size)
    return target_file


def load_raw_histdata_csv(
    file_path: Union[str, Path],
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load raw HistData M1 ASCII file (from .zip or .csv) into a pandas DataFrame.

    Converts EST (UTC-5, without DST) timestamps to standard UTC timestamps.
    Preserves exact raw OHLC prices without fabrication or interpolation.

    Args:
        file_path: Path to the raw .zip archive or extracted .csv file.
        max_rows: Optional limit on number of rows to load (useful for testing).

    Returns:
        DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        where timestamp is UTC timezone-aware datetime, sorted chronologically.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    logger.info("Reading raw HistData from %s...", path)

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as z:
            csv_candidates = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csv_candidates:
                raise ValueError(f"No CSV file found inside archive: {path}")
            csv_name = csv_candidates[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(
                    f,
                    sep=";",
                    header=None,
                    names=["dt_str", "open", "high", "low", "close", "volume"],
                    nrows=max_rows,
                    dtype={
                        "dt_str": str,
                        "open": float,
                        "high": float,
                        "low": float,
                        "close": float,
                        "volume": float,
                    },
                )
    else:
        df = pd.read_csv(
            path,
            sep=";",
            header=None,
            names=["dt_str", "open", "high", "low", "close", "volume"],
            nrows=max_rows,
            dtype={
                "dt_str": str,
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
            },
        )

    # Parse timestamps: format is "YYYYMMDD HHMMSS"
    logger.info("Parsing timestamps and converting EST (UTC-5) to UTC...")
    naive_dt = pd.to_datetime(df["dt_str"], format="%Y%m%d %H%M%S")
    # HistData has NO DST, fixed UTC-5
    df["timestamp"] = naive_dt.dt.tz_localize(HISTDATA_EST_TZ).dt.tz_convert("UTC")
    df.drop(columns=["dt_str"], inplace=True)

    # Sort strictly by timestamp
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Detect and handle duplicate timestamps in raw feed
    dup_count = int(df["timestamp"].duplicated().sum())
    if dup_count > 0:
        logger.warning(
            "Found %d duplicate M1 timestamps in raw data. Deduplicating by keeping the first occurrence.",
            dup_count,
        )
        df.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
        df.reset_index(drop=True, inplace=True)

    logger.info(
        "Loaded %d clean M1 bars spanning from %s to %s (UTC)",
        len(df),
        df["timestamp"].iloc[0].isoformat(),
        df["timestamp"].iloc[-1].isoformat(),
    )
    return df

