"""R2 -- needs R1 output.
(1) DL: does hoop drill add beyond forty for get-off?  (2) WR route-matched drill->game translation."""
import polars as pl
import statsmodels.formula.api as smf

from bigdatabowl import features as F
from bigdatabowl import load

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
pdrill = pl.read_parquet("data/processed/_combine_player_drill.parquet")
prof = load.player_profile()
pp = load.player_play()

# ---------------- (1) DL get-off ----------------
dl = pp.filter(pl.col("player_get_off").is_not_null()).group_by("nfl_id").agg(
    pl.len().alias("n"), pl.col("player_get_off").median().alias("get_off")).filter(pl.col("n") >= 50).collect()
hoop = pdrill.filter(pl.col("drill_name") == "RUN_THE_HOOP_DRILL").select("nfl_id", hoop_dur=pl.col("duration"), hoop_ttms=pl.col("time_to_max_speed"), hoop_vmax=pl.col("max_speed"))
d = dl.join(prof.select("nfl_id", "nfl_position", "forty", "ten_yd_split", "broad_jump", "combine_weight"), on="nfl_id").join(hoop, on="nfl_id")
dpd = d.drop_nulls(["forty", "hoop_dur"]).to_pandas()
print("DL n =", len(dpd))
for f in ["get_off ~ C(nfl_position) + combine_weight", "get_off ~ C(nfl_position) + combine_weight + forty",
          "get_off ~ C(nfl_position) + combine_weight + hoop_dur", "get_off ~ C(nfl_position) + combine_weight + forty + hoop_dur",
          "get_off ~ C(nfl_position) + combine_weight + forty + broad_jump + hoop_dur"]:
    m = smf.ols(f, data=dpd.dropna(subset=["broad_jump"]) if "broad" in f else dpd).fit()
    print(f"{f:75s} adjR2={m.rsquared_adj:.3f} n={int(m.nobs)}", {k: f"{v:+.3f} p={m.pvalues[k]:.3g}" for k, v in m.params.items() if k in ("forty", "hoop_dur", "broad_jump", "combine_weight")})

# ---------------- (2) WR route-matched translation ----------------
ROUTE_MAP = {"SLANT_ROUTE_LEFT": "SLANT", "CURL_ROUTE_RIGHT": "HITCH", "GO_ROUTE_RIGHT": "GO",
             "SPEED_OUT_ROUTE_LEFT": "OUT", "DAGGER_ROUTE_LEFT": "IN", "COMEBACK_ROUTE_RIGHT": "HITCH"}
routes = pp.filter(pl.col("route_ran").is_in(list(set(ROUTE_MAP.values())))).select("game_id", "play_id", "nfl_id", "route_ran")
gt = load.game_tracking().join(routes, on=["game_id", "play_id", "nfl_id"], how="inner")
snap = gt.filter(pl.col("event") == "ball_snap").group_by(["game_id", "play_id"]).agg(snap=pl.col("time").min())
fwd = gt.filter(pl.col("event").is_in(["pass_forward", "qb_sack", "run", "handoff"])).group_by(["game_id", "play_id"]).agg(end=pl.col("time").min())
gt = (gt.join(snap, on=["game_id", "play_id"]).join(fwd, on=["game_id", "play_id"], how="left")
      .with_columns(end=pl.coalesce("end", pl.col("snap") + pl.duration(seconds=3)))
      .filter((pl.col("time") >= pl.col("snap")) & (pl.col("time") <= pl.col("end") + pl.duration(milliseconds=500)))
      .drop("snap", "end"))
keys = ["game_id", "play_id", "nfl_id", "route_ran", "season"]
kin = F.add_kinematics(gt, keys)
gplay = F.summarize(kin, keys).collect()
GM = ["max_decel", "min_speed_after_launch", "max_cut_load", "max_speed", "max_accel", "time_to_max_speed"]
groute = gplay.filter(pl.col("n_frames") >= 15).group_by(["nfl_id", "route_ran"]).agg(pl.len().alias("n_plays"), *[pl.col(m).median() for m in GM]).filter(pl.col("n_plays") >= 10)
print("\nplayer-route cells:", groute.height, groute.group_by("route_ran").len())

pos = prof.select("nfl_id", "nfl_position")
rows = []
for drill, groute_name in ROUTE_MAP.items():
    cd = pdrill.filter(pl.col("drill_name") == drill)
    for g_route in sorted(set(ROUTE_MAP.values())):
        gr = groute.filter(pl.col("route_ran") == g_route)
        j = cd.join(gr, on="nfl_id", suffix="_g").join(pos, on="nfl_id")
        if j.height < 20:
            continue
        for m in GM:
            jj = j.drop_nulls([m, m + "_g"]).with_columns(
                [(pl.col(c) - pl.col(c).mean().over("nfl_position")).alias(c + "_w") for c in (m, m + "_g")])
            rows.append({"combine_drill": drill, "game_route": g_route, "matched": g_route == groute_name,
                         "metric": m, "n": jj.height, "r": jj.select(pl.corr(m + "_w", m + "_g_w")).item()})
res = pl.DataFrame(rows)
print("\nSame metric, Combine drill vs game route (within position):")
print(res.filter(pl.col("matched")).sort("metric", "r"))
print("\nMatched vs mismatched route, mean r by metric:")
print(res.group_by("metric", "matched").agg(pl.col("r").mean().round(3), pl.len()).sort("metric", "matched"))
res.write_parquet("data/processed/_route_translation.parquet")
