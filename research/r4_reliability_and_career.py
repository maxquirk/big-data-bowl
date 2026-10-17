"""R4 -- (1) attempt-to-attempt reliability of drill metrics, raw vs smoothed.
(2) Does sensor speed predict career snaps/starts beyond draft slot?

Caveat on (1): some drills' two attempts are different tasks (host: PASS_RUSH_DRILL left = rip,
right = swim), so low r there is a task difference, not noise.
"""

import numpy as np
import polars as pl
import statsmodels.formula.api as smf

from bigdatabowl import features as F
from bigdatabowl import load

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(220)

# ---------- (1) reliability ----------
K = F.ATTEMPT_KEYS
kin = (
    F.add_kinematics(load.combine_tracking(), K)
    .with_columns(s_sm=pl.col("s").rolling_mean(5, center=True, min_samples=3).over(K))
    .with_columns(
        ds_sm=(pl.col("s_sm").diff() / F.DT).over(K).rolling_mean(3, center=True, min_samples=2).over(K)
    )
)
att = kin.group_by(K).agg(
    max_speed=pl.col("s").max(),
    max_decel_raw=pl.col("ds_dt").min(),
    max_decel_sm=pl.col("ds_sm").min(),
    max_accel_raw=pl.col("ds_dt").max(),
    max_accel_sm=pl.col("ds_sm").max(),
    cut_raw=pl.col("cut_load").max(),
    cut_p90=pl.col("cut_load").quantile(0.9),
    duration=pl.col("t").max(),
).collect()
M = [c for c in att.columns if c not in K]
two = (
    att.sort("nfl_id", "drill_name", "attempt", "event_id")
    .with_columns(k=pl.int_range(pl.len()).over("nfl_id", "drill_name"))
    .filter(pl.col("k") < 2)
)
w = two.pivot(on="k", index=["nfl_id", "drill_type", "drill_name"], values=M)
rows = []
for (_, dn), g in w.group_by(["drill_type", "drill_name"]):
    g = g.drop_nulls()
    if g.height < 25:
        continue
    r = {"drill": dn[:30], "n": g.height}
    for m in M:
        r[m] = round(g.select(pl.corr(f"{m}_0", f"{m}_1")).item(), 2)
    rows.append(r)
rel = pl.DataFrame(rows).sort("n", descending=True)
print("Test-retest r (first two attempts):")
print(rel)
print(rel.select(pl.exclude("drill", "n").median()))

# ---------- (2) career outcomes vs draft slot ----------
prof = load.player_profile()
a = F.combine_attempt_features(load.combine_tracking().filter(pl.col("drill_type") == "FORTY_YARD_DASH"))
best = F.combine_best_by_drill(a).select("nfl_id", sensor_top=pl.col("max_speed"))
GROUPS = {"T": "OL", "G": "OL", "C": "OL", "DE": "EDGE", "OLB": "EDGE", "DT": "DL", "NT": "DL",
          "SS": "S", "FS": "S", "DB": "S", "MLB": "LB", "ILB": "LB"}
df = (
    prof.join(best, on="nfl_id", how="left")
    .with_columns(
        udfa=pl.col("draft_overall_pick").is_null(),
        log_pick=pl.col("draft_overall_pick").fill_null(262).log(),
        grp=pl.col("nfl_position").replace(GROUPS),
        seasons=2026 - pl.col("draft_year"),
    )
    .with_columns(
        snaps_per_season=(pl.col("career_offensive_snaps") + pl.col("career_defensive_snaps")) / pl.col("seasons"),
        starts_per_season=pl.col("career_games_started") / pl.col("seasons"),
    )
)
d = df.drop_nulls(["sensor_top", "forty"]).filter(pl.len().over("grp") >= 10).to_pandas()
d["sqrt_snaps"] = np.sqrt(d["snaps_per_season"])
base = "C(grp) + C(draft_year) + log_pick + udfa"
for extra in ["", " + forty", " + sensor_top", " + ngs_athleticism_score"]:
    m = smf.ols(f"sqrt_snaps ~ {base}{extra}", data=d).fit()
    print(f"sqrt_snaps ~ ...{extra:28s} adjR2={m.rsquared_adj:.3f}",
          {k: f"{v:+.3f} p={m.pvalues[k]:.3g}" for k, v in m.params.items()
           if k in ("forty", "sensor_top", "log_pick", "ngs_athleticism_score")})
