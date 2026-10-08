"""Movement features from 10 Hz tracking -- shared by Combine drills and game plays.

Conventions:
  * `s` is speed (yd/s), `a` is acceleration magnitude (yd/s^2, unsigned).
  * Signed accel/decel comes from the frame-to-frame change in `s` (`ds_dt`).
  * Heading change uses `dir` wrapped to [-180, 180).
"""

from __future__ import annotations

import polars as pl

DT = 0.1  # seconds between frames
YD_S_TO_MPH = 3600 / 1760

ATTEMPT_KEYS = ["event_id", "nfl_id", "drill_type", "drill_name", "attempt"]
PLAY_KEYS = ["game_id", "play_id", "nfl_id"]


def _wrap_deg(c: pl.Expr) -> pl.Expr:
    """Wrap an angle difference into [-180, 180)."""
    return ((c + 540) % 360) - 180


def add_kinematics(lf: pl.LazyFrame, keys: list[str]) -> pl.LazyFrame:
    """Per-frame derived signals: elapsed time, signed accel, turn rate, cumulative distance."""
    return (
        lf.sort([*keys, "time"])
        .with_columns(
            t=(pl.col("time") - pl.col("time").first()).dt.total_milliseconds().over(keys) / 1000,
            ds_dt=(pl.col("s").diff() / DT).over(keys),
            turn_rate=(_wrap_deg(pl.col("dir").diff()) / DT).over(keys),  # deg/s, signed
            cum_dis=pl.col("dis").cum_sum().over(keys),
        )
        .with_columns(
            # Centripetal-ish "cut load": how hard you're turning while moving fast
            cut_load=(pl.col("turn_rate").abs().radians() * pl.col("s")),
        )
    )


def summarize(lf: pl.LazyFrame, keys: list[str]) -> pl.LazyFrame:
    """One row per attempt/play with speed, burst, braking and change-of-direction metrics."""
    s, t = pl.col("s"), pl.col("t")
    return lf.group_by(keys).agg(
        n_frames=pl.len(),
        duration=t.max(),
        distance=pl.col("dis").sum(),
        max_speed=s.max(),
        max_speed_mph=s.max() * YD_S_TO_MPH,
        time_to_max_speed=t.filter(s == s.max()).min(),  # first frame at top speed
        max_accel=pl.col("ds_dt").max(),
        max_decel=pl.col("ds_dt").min(),  # negative = braking
        p95_accel_mag=pl.col("a").quantile(0.95),
        max_turn_rate=pl.col("turn_rate").abs().max(),
        max_cut_load=pl.col("cut_load").max(),
        total_heading_change=pl.col("turn_rate").abs().sum() * DT,
        # Speed retention: min speed after reaching 50% of peak (how much you bleed in cuts)
        min_speed_after_launch=s.filter(t > t.filter(s > 0.5 * s.max()).min()).min(),
    )


def split_times(lf: pl.LazyFrame, keys: list[str], marks=(10, 20, 40)) -> pl.LazyFrame:
    """Time to reach each cumulative distance mark (e.g. sensor 10/20/40-yd splits)."""
    aggs = [
        pl.col("t").filter(pl.col("cum_dis") >= m).min().alias(f"t_{m}yd") for m in marks
    ]
    out = lf.group_by(keys).agg(aggs)
    if 10 in marks and 40 in marks:
        out = out.with_columns(t_10_40=pl.col("t_40yd") - pl.col("t_10yd"))
    if 20 in marks and 40 in marks:
        out = out.with_columns(flying_20=pl.col("t_40yd") - pl.col("t_20yd"))
    return out


# --- Combine ---------------------------------------------------------------


def combine_attempt_features(combine_tracking: pl.LazyFrame) -> pl.DataFrame:
    """One row per Combine drill attempt."""
    kin = add_kinematics(combine_tracking, ATTEMPT_KEYS)
    feats = summarize(kin, ATTEMPT_KEYS)
    forty = split_times(kin.filter(pl.col("drill_type") == "FORTY_YARD_DASH"), ATTEMPT_KEYS)
    return feats.join(forty, on=ATTEMPT_KEYS, how="left").collect()


def combine_best_by_drill(attempts: pl.DataFrame) -> pl.DataFrame:
    """Best attempt per player x drill (max of 'good' metrics, min of times)."""
    time_cols = [c for c in attempts.columns if c.startswith("t_") or c == "flying_20"]
    return attempts.group_by(["nfl_id", "drill_type", "drill_name"]).agg(
        pl.len().alias("n_attempts"),
        pl.col("max_speed", "max_accel", "max_turn_rate", "max_cut_load").max(),
        pl.col("max_decel", *time_cols).min(),
    )


# --- Games -----------------------------------------------------------------


def snap_times(game_tracking: pl.LazyFrame) -> pl.LazyFrame:
    return (
        game_tracking.filter(pl.col("event") == "ball_snap")
        .group_by(["game_id", "play_id"])
        .agg(snap_time=pl.col("time").min())
    )


def game_play_features(game_tracking: pl.LazyFrame, post_snap_window: float = 3.0) -> pl.LazyFrame:
    """One row per player-play, restricted to the first `post_snap_window` seconds after the snap.

    This is the window where Combine-style traits (get-off, burst, first-step) should show up.
    """
    lf = (
        game_tracking.join(snap_times(game_tracking), on=["game_id", "play_id"], how="inner")
        .with_columns(t_snap=(pl.col("time") - pl.col("snap_time")).dt.total_milliseconds() / 1000)
        .filter(pl.col("t_snap").is_between(0, post_snap_window))
        .drop("snap_time", "t_snap")
    )
    keys = [*PLAY_KEYS, "season"] if "season" in lf.collect_schema() else PLAY_KEYS
    kin = add_kinematics(lf, keys)
    return summarize(kin, keys).join(split_times(kin, keys, marks=(5, 10)), on=keys, how="left")


def game_player_season_features(play_feats: pl.LazyFrame) -> pl.LazyFrame:
    """Robust per-player-season aggregates (high quantiles, not raw maxes -- tracking glitches)."""
    return play_feats.group_by(["nfl_id", "season"]).agg(
        n_plays=pl.len(),
        game_top_speed=pl.col("max_speed").quantile(0.99),
        game_p90_speed=pl.col("max_speed").quantile(0.90),
        game_p90_accel=pl.col("max_accel").quantile(0.90),
        game_p90_decel=pl.col("max_decel").quantile(0.10),
        game_p90_cut_load=pl.col("max_cut_load").quantile(0.90),
        game_median_t_5yd=pl.col("t_5yd").median(),
    )
