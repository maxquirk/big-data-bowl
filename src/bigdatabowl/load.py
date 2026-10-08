"""Lazy loaders for every competition table.

Everything returns a polars LazyFrame so the ~9M-row game tracking never has to
fit in memory until you `.collect()`. Prefers Parquet in data/processed/, falls
back to the raw CSV.
"""

from __future__ import annotations

import polars as pl

from bigdatabowl.paths import PROCESSED, RAW

SEASONS = (2023, 2024, 2025)


def _scan(name: str) -> pl.LazyFrame:
    pq = PROCESSED / f"{name}.parquet"
    if pq.exists():
        return pl.scan_parquet(pq)
    csv = RAW / f"{name}.csv"
    if csv.exists():
        return pl.scan_csv(csv, null_values=["NA", ""], infer_schema_length=50_000)
    raise FileNotFoundError(f"{name} not found. Run `uv run python scripts/download_data.py`.")


def _parse_time(lf: pl.LazyFrame) -> pl.LazyFrame:
    if lf.collect_schema().get("time") == pl.String:
        lf = lf.with_columns(pl.col("time").str.to_datetime(strict=False))
    return lf


def players() -> pl.LazyFrame:
    return _scan("players")


def combine_results() -> pl.LazyFrame:
    return _scan("combine_results")


def career() -> pl.LazyFrame:
    return _scan("player_career_successes")


def games() -> pl.LazyFrame:
    return _scan("games")


def player_play() -> pl.LazyFrame:
    return _scan("player_play")


def combine_tracking(players_only: bool = True) -> pl.LazyFrame:
    """10 Hz Combine drill tracking. One `event_id` = one drill attempt."""
    lf = _parse_time(_scan("combine_tracking"))
    if players_only:
        lf = lf.filter(pl.col("entity_type") == "PLAYER")
    return lf


def game_tracking(seasons: int | tuple[int, ...] = SEASONS) -> pl.LazyFrame:
    """10 Hz in-game tracking for the rookie cohort, optionally for a subset of seasons."""
    if isinstance(seasons, int):
        seasons = (seasons,)
    frames = [
        _parse_time(_scan(f"game_tracking_{s}")).with_columns(pl.lit(s).alias("season"))
        for s in seasons
    ]
    return pl.concat(frames, how="diagonal_relaxed")


def player_profile() -> pl.DataFrame:
    """One row per prospect: bio + combine testing + career outcomes."""
    return (
        players()
        .join(combine_results().drop("draft_year"), on="nfl_id", how="left")
        .join(career(), on="nfl_id", how="left")
        .collect()
    )
