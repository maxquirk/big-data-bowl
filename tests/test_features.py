"""Sanity checks on synthetic tracking so features can be trusted before real data lands."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from bigdatabowl import features as F


def _sprint_then_cut(n=60, accel=4.0, top=10.0) -> pl.LazyFrame:
    t = np.arange(n) * F.DT
    s = np.minimum(accel * t, top)
    s[40:] = np.maximum(top - 8.0 * (t[40:] - t[40]), 2.0)  # brake
    direction = np.where(np.arange(n) < 45, 90.0, 180.0)  # 90-degree cut at frame 45
    t0 = datetime(2025, 3, 1, 12)  # noqa: DTZ001 -- tracking timestamps are naive
    return pl.LazyFrame(
        {
            "event_id": "e1",
            "nfl_id": 1,
            "drill_type": "FORTY_YARD_DASH",
            "drill_name": "FORTY",
            "attempt": 1,
            "time": [t0 + timedelta(seconds=float(x)) for x in t],
            "s": s,
            "a": np.abs(np.gradient(s, F.DT)),
            "dis": s * F.DT,
            "dir": direction,
        }
    )


def test_attempt_features():
    out = F.combine_attempt_features(_sprint_then_cut())
    row = out.row(0, named=True)
    assert out.height == 1
    assert abs(row["max_speed"] - 10.0) < 1e-9
    assert abs(row["time_to_max_speed"] - 2.5) < 0.11
    assert abs(row["max_accel"] - 4.0) < 1e-6
    assert row["max_decel"] < -7.9
    assert abs(row["max_turn_rate"] - 900.0) < 1e-6  # 90 deg in 0.1 s
    assert row["t_10yd"] is not None and row["t_10yd"] < row["t_20yd"]


def test_heading_wraparound():
    lf = _sprint_then_cut().with_columns(dir=pl.Series([359.0, 1.0] * 30))
    kin = F.add_kinematics(lf, F.ATTEMPT_KEYS).collect()
    assert kin["turn_rate"].drop_nulls().abs().max() == 20.0  # 2 deg / 0.1 s, not 358
