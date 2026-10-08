"""Download the NFL Big Data Bowl 2027 data from Kaggle and convert it to Parquet.

Prereqs (one-time):
  1. Join the competition + accept the rules:
     https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/rules
  2. Have a Kaggle API token ready (kaggle.com/settings -> API -> Create New Token).
     The script prompts for it if KAGGLE_API_TOKEN isn't set. Note: kagglehub.login()
     only holds the token in memory, so logging in from a separate command doesn't persist.

Usage:
  uv run python scripts/download_data.py                  # download + parquet
  uv run python scripts/download_data.py --skip-download  # just re-convert CSVs
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import polars as pl

from bigdatabowl.paths import PROCESSED, RAW

COMPETITION = "nfl-big-data-bowl-2027"


def download() -> None:
    import kagglehub

    if kagglehub.auth.get_username() is None:
        kagglehub.login()
    print(f"Downloading {COMPETITION} (~2.3 GB, cached by kagglehub)...")
    src = Path(kagglehub.competition_download(COMPETITION))

    # Hard-link from the kagglehub cache into data/raw (no 2nd copy); copy if on another drive
    for f in src.rglob("*.csv"):
        dst = RAW / f.name
        if dst.exists():
            continue
        try:
            os.link(f, dst)
        except OSError:
            shutil.copy2(f, dst)
        print(f"  {f.name}")


def to_parquet() -> None:
    csvs = sorted(RAW.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSVs in {RAW}. Run without --skip-download first.")
    for csv in csvs:
        out = PROCESSED / f"{csv.stem}.parquet"
        print(f"{csv.name} -> {out.name}")
        # Streaming sink keeps memory flat for the ~1 GB game_tracking files
        pl.scan_csv(csv, null_values=["NA", ""], infer_schema_length=50_000).sink_parquet(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download()
    to_parquet()
    print("Done.")


if __name__ == "__main__":
    main()
