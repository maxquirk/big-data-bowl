"""R1 -- run first (writes data/processed/_combine_player_drill.parquet).
Scan: which Combine drill x movement metric links to which in-game outcome, vs stopwatch baselines."""
import numpy as np
import polars as pl
import statsmodels.formula.api as smf

from bigdatabowl import features as F
from bigdatabowl import load

pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(200)

# ---------- Combine per-attempt features + burst ----------
ct = load.combine_tracking().filter(~pl.col("drill_type").is_in(["FORTY_YARD_DASH"]))
kin = F.add_kinematics(ct, F.ATTEMPT_KEYS)
summ = F.summarize(kin, F.ATTEMPT_KEYS)
# burst from first movement
kin2 = kin.with_columns(
    t0=pl.col("t").filter(pl.col("s") >= F.START_SPEED).min().over(F.ATTEMPT_KEYS)
).with_columns(tm=pl.col("t") - pl.col("t0"))
burst = kin2.group_by(F.ATTEMPT_KEYS).agg(
    v_05=pl.col("s").filter(pl.col("tm").is_between(0.45, 0.55)).mean(),
    v_10=pl.col("s").filter(pl.col("tm").is_between(0.95, 1.05)).mean(),
    d_10=pl.col("disp").filter(pl.col("tm").is_between(0.95, 1.05)).mean(),
    peak_accel_1s=pl.col("ds_dt").filter(pl.col("tm").is_between(0, 1.0)).max(),
    # re-acceleration: max accel occurring after the hardest brake
    reaccel_after_brake=pl.col("ds_dt").filter(pl.col("t") > pl.col("t").sort_by("ds_dt").first()).max(),
)
att = summ.join(burst, on=F.ATTEMPT_KEYS).collect()
METRICS = ["max_speed", "max_accel", "max_decel", "max_cut_load", "min_speed_after_launch",
           "time_to_max_speed", "v_05", "v_10", "d_10", "peak_accel_1s", "reaccel_after_brake", "duration"]
# player x drill: best attempt for "bigger is better", min for decel/duration/time
agg = []
for m in METRICS:
    agg.append((pl.col(m).min() if m in ("max_decel", "duration", "time_to_max_speed") else pl.col(m).max()).alias(m))
pdrill = att.group_by(["nfl_id", "drill_type", "drill_name"]).agg(agg)
pdrill.write_parquet("data/processed/_combine_player_drill.parquet")

prof = load.player_profile()
pos = prof.select("nfl_id", "nfl_position", "forty", "ten_yd_split", "three_cone", "short_shuttle",
                  "vertical", "broad_jump", "combine_weight", "ngs_athleticism_score", "draft_overall_pick")

# ---------- Game outcomes ----------
pp = load.player_play()

# WR: separation over expected (route + depth + cushion adjusted)
wr = pp.filter(pl.col("separation_at_pass_forward").is_not_null() & pl.col("route_ran").is_not_null()).select(
    "nfl_id", "separation_at_pass_forward", "route_ran", "pass_length", "cushion", "lined_up_position", "target"
).collect().to_pandas()
wr["depth_bin"] = np.clip(wr["pass_length"].fillna(0) // 5, -1, 6)
m = smf.ols("separation_at_pass_forward ~ C(route_ran) + C(depth_bin) + cushion + C(lined_up_position)",
            data=wr.dropna(subset=["cushion"])).fit()
wr2 = wr.dropna(subset=["cushion"]).copy(); wr2["soe"] = m.resid
wr_out = pl.from_pandas(wr2).group_by("nfl_id").agg(pl.len().alias("n"), pl.col("soe").mean().alias("sep_over_expected")).filter(pl.col("n") >= 40)
print("WR SOE model R2", round(m.rsquared, 3), "players", wr_out.height)

# DL/EDGE: get-off and quick-pressure rate
dl_out = pp.filter(pl.col("player_get_off").is_not_null()).group_by("nfl_id").agg(
    pl.len().alias("n"), pl.col("player_get_off").median().alias("get_off"),
    pl.col("quick_pressure").fill_null(False).cast(pl.Float64).mean().alias("quick_pressure_rate"),
    pl.col("sack").fill_null(0).mean().alias("sack_rate"),
).filter(pl.col("n") >= 50).collect()
print("DL players", dl_out.height)

# OL: pressure allowed rate + mean peak pressure prob allowed
ol_out = pp.filter(pl.col("peak_pressure_probability_allowed").is_not_null()).group_by("nfl_id").agg(
    pl.len().alias("n"), pl.col("peak_pressure_probability_allowed").mean().alias("peak_pp_allowed"),
    pl.col("pressure_allowed").fill_null(False).cast(pl.Float64).mean().alias("pressure_allowed_rate"),
).filter(pl.col("n") >= 100).collect()
print("OL players", ol_out.height)


def scan(outcome_df, outcome, drill_types, baselines, label):
    base = outcome_df.join(pos, on="nfl_id")
    rows = []
    for (dt, dn), g in pdrill.filter(pl.col("drill_type").is_in(drill_types)).group_by(["drill_type", "drill_name"]):
        j = base.join(g, on="nfl_id")
        if j.height < 30:
            continue
        for mt in METRICS:
            jj = j.drop_nulls([mt, outcome])
            if jj.height < 30:
                continue
            # within-position correlation
            jj = jj.with_columns([(pl.col(c) - pl.col(c).mean().over("nfl_position")).alias(c + "_w") for c in (mt, outcome)])
            r = jj.select(pl.corr(mt + "_w", outcome + "_w")).item()
            rows.append({"drill": dn, "metric": mt, "n": jj.height, "r_within_pos": r})
    for b in baselines:
        jj = base.drop_nulls([b, outcome]).with_columns([(pl.col(c) - pl.col(c).mean().over("nfl_position")).alias(c + "_w") for c in (b, outcome)])
        rows.append({"drill": "STOPWATCH", "metric": b, "n": jj.height, "r_within_pos": jj.select(pl.corr(b + "_w", outcome + "_w")).item()})
    out = pl.DataFrame(rows).with_columns(
        abs_r=pl.col("r_within_pos").abs(),
        # rough p via Fisher z
        z=(pl.col("r_within_pos").arctanh() * (pl.col("n") - 3).sqrt()).abs(),
    ).sort("abs_r", descending=True)
    n_tests = out.filter(pl.col("drill") != "STOPWATCH").height
    print(f"\n===== {label}: {outcome}  ({n_tests} drill tests; |z|>{3.3:.1f} ~ Bonferroni-ish) =====")
    print(out.head(15))
    print(out.filter(pl.col("drill") == "STOPWATCH"))
    return out


STOP = ["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump", "ngs_athleticism_score"]
scan(wr_out, "sep_over_expected", ["SKILL_DRILLS_WR", "SKILL_DRILLS_TE", "THREE_CONE_DRILL", "SHORT_SHUTTLE"], STOP, "WR/TE")
scan(dl_out, "get_off", ["SKILL_DRILLS_DL", "SKILL_DRILLS_LB", "THREE_CONE_DRILL", "SHORT_SHUTTLE"], STOP, "DL")
scan(dl_out, "quick_pressure_rate", ["SKILL_DRILLS_DL", "SKILL_DRILLS_LB", "THREE_CONE_DRILL", "SHORT_SHUTTLE"], STOP, "DL")
scan(ol_out, "peak_pp_allowed", ["SKILL_DRILLS_OL", "THREE_CONE_DRILL", "SHORT_SHUTTLE"], STOP, "OL")
scan(ol_out, "pressure_allowed_rate", ["SKILL_DRILLS_OL", "THREE_CONE_DRILL", "SHORT_SHUTTLE"], STOP, "OL")
