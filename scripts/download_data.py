"""Download the NFL Big Data Bowl 2027 data from Kaggle and convert it to Parquet.

Prereqs (one-time):
  1. Join the competition + accept the rules:
     https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/rules
  2. Create an API token at https://www.kaggle.com/settings -> "Create New Token"
     and save it to ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME / KAGGLE_KEY).

Usage:
  uv run python scripts/download_data.py            # download + unzip + parquet
  uv run python scripts/download_data.py --skip-download   # just re-convert CSVs
"""

from __future__ import annotations

import argparse
import zipfile

import polars as pl

from bigdatabowl.paths import PROCESSED, RAW

COMPETITION = "nfl-big-data-bowl-2027"


def download() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {COMPETITION} -> {RAW} (~2.3 GB)...")
    api.competition_download_files(COMPETITION, path=str(RAW), quiet=False)

    for zf in RAW.glob("*.zip"):
        print(f"Unzipping {zf.name}...")
        with zipfile.ZipFile(zf) as z:
            z.extractall(RAW)
        zf.unlink()


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
