"""DB: Combine backpedal/transition -> in-game break on the throw (pass_forward -> +1.5 s)."""
import polars as pl
import statsmodels.formula.api as smf

from bigdatabowl import features as F
from bigdatabowl import load

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
SM = {"window_size": 5, "center": True, "min_samples": 3}


def smooth_feats(lf, keys):
    k = F.add_kinematics(lf, keys).with_columns(s_sm=pl.col("s").rolling_mean(**SM).over(keys))
    k = k.with_columns(ds_sm=(pl.col("s_sm").diff() / F.DT).over(keys).rolling_mean(3, center=True, min_samples=2).over(keys))
    return k.group_by(keys).agg(
        decel_sm=pl.col("ds_sm").min(), accel_sm=pl.col("ds_sm").max(),
        cut_p90=pl.col("cut_load").quantile(0.9), vmax=pl.col("s").max(), dur=pl.col("t").max(),
        transition_time=(pl.col("t").filter(pl.col("ds_sm") > 1.0).min() - pl.col("t").filter(pl.col("ds_sm") == pl.col("ds_sm").min()).first()),
    )

# Combine DB drills, player-level mean over attempts (mean is more reliable than best)
db_drills = ["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "W_DRILL", "BOX_DRILL", "GAUNTLET_DRILL", "TERYL_AUSTIN_DRILL_1", "TERYL_AUSTIN_DRILL_2"]
ct = load.combine_tracking().filter((pl.col("drill_type") == "SKILL_DRILLS_DB") & pl.col("drill_name").is_in(db_drills))
cf = smooth_feats(ct, F.ATTEMPT_KEYS).collect()
cmb = cf.group_by("nfl_id", "drill_name").agg(pl.exclude(F.ATTEMPT_KEYS).mean())

# Game: DB on pass plays, window = pass_forward -> +1.5 s
pp = load.player_play()
dbs = pp.filter(pl.col("lined_up_position").is_in(["CB", "SLOT_CB", "HIGH_SAFETY"]) & pl.col("pass_result").is_in(["C", "I", "IN"]))
dbs = dbs.select("game_id", "play_id", "nfl_id", "lined_up_position", "team_coverage_man_zone", "pass_length")
gt = load.game_tracking().join(dbs, on=["game_id", "play_id", "nfl_id"], how="inner")
pf = gt.filter(pl.col("event") == "pass_forward").group_by(["game_id", "play_id"]).agg(pf=pl.col("time").min())
gt = gt.join(pf, on=["game_id", "play_id"]).filter(
    (pl.col("time") >= pl.col("pf")) & (pl.col("time") <= pl.col("pf") + pl.duration(milliseconds=1500)))
keys = ["game_id", "play_id", "nfl_id", "lined_up_position", "team_coverage_man_zone"]
k = F.add_kinematics(gt, keys).with_columns(s_sm=pl.col("s").rolling_mean(**SM).over(keys))
k = k.with_columns(ds_sm=(pl.col("s_sm").diff() / F.DT).over(keys).rolling_mean(3, center=True, min_samples=2).over(keys))
gplay = k.group_by(keys).agg(
    s_at_throw=pl.col("s").first(), s_plus1=pl.col("s").filter(pl.col("t").is_between(0.95, 1.05)).mean(),
    break_accel=pl.col("ds_sm").max(), break_cut=pl.col("cut_load").quantile(0.9),
    n=pl.len(),
).filter(pl.col("n") >= 12).with_columns(speed_gain=pl.col("s_plus1") - pl.col("s_at_throw")).collect()
print("DB plays:", gplay.height)
# adjust for alignment + coverage + speed at throw, then player mean residual
gpd = gplay.drop_nulls().to_pandas()
out = {}
for y in ["break_accel", "speed_gain", "break_cut"]:
    m = smf.ols(f"{y} ~ C(lined_up_position) + C(team_coverage_man_zone) + s_at_throw", data=gpd).fit()
    gpd[y + "_oe"] = m.resid
gpl = pl.from_pandas(gpd).group_by("nfl_id").agg(pl.len().alias("n_plays"), *[pl.col(c + "_oe").mean() for c in ["break_accel", "speed_gain", "break_cut"]]).filter(pl.col("n_plays") >= 40)
# split-half reliability of the game metric
half = pl.from_pandas(gpd).with_columns(h=pl.int_range(pl.len()).over("nfl_id") % 2).group_by("nfl_id", "h").agg(pl.col("break_accel_oe").mean(), pl.len()).pivot(on="h", index="nfl_id", values="break_accel_oe").drop_nulls()
print("game break_accel split-half r:", round(half.select(pl.corr("0", "1")).item(), 3), "n", half.height)
print("DB players with >=40 throws:", gpl.height)

prof = load.player_profile().select("nfl_id", "nfl_position", "forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump")
rows = []
for (dn,), g in cmb.group_by(["drill_name"]):
    j = gpl.join(g, on="nfl_id").join(prof, on="nfl_id")
    for cm in ["decel_sm", "accel_sm", "cut_p90", "vmax", "dur", "transition_time"]:
        for gm in ["break_accel_oe", "speed_gain_oe", "break_cut_oe"]:
            jj = j.drop_nulls([cm, gm])
            if jj.height < 30: continue
            jj = jj.with_columns([(pl.col(c) - pl.col(c).mean().over("nfl_position")).alias(c + "_w") for c in (cm, gm)])
            r = jj.select(pl.corr(cm + "_w", gm + "_w")).item()
            rows.append({"drill": dn[:34], "combine": cm, "game": gm, "n": jj.height, "r": r})
for b in ["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump"]:
    j = gpl.join(prof, on="nfl_id")
    for gm in ["break_accel_oe", "speed_gain_oe", "break_cut_oe"]:
        jj = j.drop_nulls([b, gm]).with_columns([(pl.col(c) - pl.col(c).mean().over("nfl_position")).alias(c + "_w") for c in (b, gm)])
        rows.append({"drill": "STOPWATCH", "combine": b, "game": gm, "n": jj.height, "r": jj.select(pl.corr(b + "_w", gm + "_w")).item()})
res = pl.DataFrame(rows).with_columns(z=(pl.col("r").arctanh() * (pl.col("n") - 3).sqrt()).abs()).sort("z", descending=True)
print(f"{res.filter(pl.col('drill') != 'STOPWATCH').height} drill tests")
print(res.head(20))
print(res.filter(pl.col("drill") == "STOPWATCH").head(8))
