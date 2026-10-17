"""R5 -- find the strongest Combine-tracking -> game link for every position group, honestly.

For each group:
  * Combine side: smoothed movement metrics per drill, averaged over attempts (more reliable
    than single-rep peaks), for the group's own drills + 40 / 3-cone / shuttle.
  * Game side: the matching movement measured in game tracking for that role (burst off the
    snap, top speed, break on the throw, pass-set displacement...) plus NGS outcomes.
  * Within-position correlations (position means removed) for every combine x game pair.
  * Family-wise permutation test: shuffle players' game rows 1,000x, take the max |r| across
    ALL of the group's tests each time -> adjusted p for each observed r.
  * Incremental check for the winner: does it add over weight + the best stopwatch number?

Run: uv run python -I -X utf8 research/r5_position_scan.py
Writes: data/processed/_game_player_feats.parquet, _combine_player_feats.parquet, outputs/r5_*.csv
"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.formula.api as smf

from bigdatabowl import features as F
from bigdatabowl import load
from bigdatabowl.paths import OUTPUTS, PROCESSED

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_fmt_str_lengths(45)
RNG = np.random.default_rng(2027)
N_PERM = 1000
MIN_N = 25

GROUPS = {
    "WR": ["WR"], "TE": ["TE"], "OL": ["T", "G", "C"], "EDGE": ["DE", "OLB"],
    "IDL": ["DT", "NT"], "CB": ["CB"], "S": ["FS", "SS", "DB"],
}
GROUP_DRILLS = {
    "WR": ["SKILL_DRILLS_WR"], "TE": ["SKILL_DRILLS_TE"], "OL": ["SKILL_DRILLS_OL"],
    "EDGE": ["SKILL_DRILLS_DL", "SKILL_DRILLS_LB"], "IDL": ["SKILL_DRILLS_DL"],
    "CB": ["SKILL_DRILLS_DB"], "S": ["SKILL_DRILLS_DB"],
}
COMMON_DRILLS = ["FORTY_YARD_DASH", "THREE_CONE_DRILL", "SHORT_SHUTTLE"]
STOPWATCH = ["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump"]

# "Same movement" matching: the confirmatory family only pairs Combine and game metrics that
# measure the same physical quality. Defined by metric class, not by looking at winners.
COMBINE_CLASS = {"vmax": "speed", "v05": "burst", "v10": "burst", "d10": "burst", "accel": "burst",
                 "decel": "cod", "cut": "cod", "dur": "drill_time"}
GAME_CLASS = {
    "route_top_speed": "speed", "rush_vmax3": "speed", "run_block_top_speed3": "speed", "db_top_speed": "speed",
    "release_v10": "burst", "release_d10": "burst", "rush_v05": "burst", "rush_d10": "burst",
    "rush_accel_1s": "burst", "set_d10": "burst", "set_v05": "burst", "run_block_d20": "burst",
    "db_d20_post_snap": "burst", "break_speed_gain": "burst", "break_accel": "burst",
    "route_decel": "cod", "route_cut": "cod", "db_cut": "cod",
    "sep_oe": "outcome", "yac_oe": "outcome", "ngs_get_off": "outcome", "quick_pressure_rate": "outcome",
    "pressure_rate": "outcome", "peak_pp_allowed": "outcome", "pressure_allowed_rate": "outcome",
    "tackle_rate": "outcome",
}
MATCHES = {"speed": {"speed"}, "burst": {"burst", "drill_time"}, "cod": {"cod", "drill_time"},
           "outcome": {"burst", "drill_time"}}


def smoothed(lf: pl.LazyFrame, keys: list[str]) -> pl.LazyFrame:
    """Kinematics + 0.5 s smoothed speed/accel (single-frame peaks are ~.3 test-retest)."""
    return (
        F.add_kinematics(lf, keys)
        .with_columns(s_sm=pl.col("s").rolling_mean(5, center=True, min_samples=3).over(keys))
        .with_columns(
            ds_sm=(pl.col("s_sm").diff() / F.DT).over(keys).rolling_mean(3, center=True, min_samples=2).over(keys)
        )
    )


def at(col: str, t: float, tcol: str = "tm") -> pl.Expr:
    return pl.col(col).filter(pl.col(tcol).is_between(t - 0.05, t + 0.05)).mean()


# ---------------------------------------------------------------- Combine side
def combine_features() -> pl.DataFrame:
    K = F.ATTEMPT_KEYS
    kin = smoothed(load.combine_tracking(), K).with_columns(
        tm=pl.col("t") - pl.col("t").filter(pl.col("s") >= F.START_SPEED).min().over(K)
    )
    att = kin.group_by(K).agg(
        vmax=pl.col("s").max(),
        accel=pl.col("ds_sm").max(),
        decel=pl.col("ds_sm").min(),
        cut=pl.col("cut_load").quantile(0.9),
        v05=at("s", 0.5),
        v10=at("s", 1.0),
        d10=at("disp", 1.0),
        dur=pl.col("t").max() - pl.col("t").filter(pl.col("s") >= F.START_SPEED).min(),
    ).collect()
    pdr = att.group_by("nfl_id", "drill_name").agg(pl.exclude(K).mean(), pl.len().alias("reps"))
    drill_type = att.select("drill_name", "drill_type").unique("drill_name")
    pdr = pdr.join(drill_type, on="drill_name")
    pdr.write_parquet(PROCESSED / "_combine_player_feats.parquet")
    return pdr


# ---------------------------------------------------------------- Game side
def game_features() -> pl.DataFrame:
    pp = load.player_play().select(
        "game_id", "play_id", "nfl_id", "lined_up_position", "route_ran", "pass_result",
        "player_get_off", "quick_pressure", "time_to_pressure", "sack", "pressure_allowed",
        "peak_pressure_probability_allowed", "separation_at_pass_forward", "cushion", "pass_length",
        "yards_after_catch", "expected_yards_after_catch", "tackle", "assist", "rush_yards",
    )
    gt = load.game_tracking()
    ev = gt.filter(pl.col("event").is_in(["ball_snap", "pass_forward"])).group_by("game_id", "play_id").agg(
        snap=pl.col("time").filter(pl.col("event") == "ball_snap").min(),
        pf=pl.col("time").filter(pl.col("event") == "pass_forward").min(),
    )
    keys = ["game_id", "play_id", "nfl_id"]
    g = (
        gt.join(ev, on=["game_id", "play_id"])
        .filter(pl.col("snap").is_not_null())
        .with_columns(
            tm=(pl.col("time") - pl.col("snap")).dt.total_milliseconds() / 1000,
            tpf=(pl.col("time") - pl.col("pf")).dt.total_milliseconds() / 1000,
        )
        .filter(pl.col("tm").is_between(-0.2, 10.0))
    )
    kin = smoothed(g, keys).with_columns(
        # displacement from position at the snap
        disp_snap=(
            (pl.col("x") - pl.col("x").filter(pl.col("tm") >= 0).first()).pow(2)
            + (pl.col("y") - pl.col("y").filter(pl.col("tm") >= 0).first()).pow(2)
        ).sqrt().over(keys)
    )
    early = pl.col("tm").is_between(0, 1.0)
    play = kin.group_by(keys).agg(
        g_v05=at("s", 0.5), g_v10=at("s", 1.0),
        g_d10=at("disp_snap", 1.0), g_d20=at("disp_snap", 2.0),
        g_accel_1s=pl.col("ds_sm").filter(early).max(),
        g_vmax3=pl.col("s").filter(pl.col("tm").is_between(0, 3)).max(),
        g_vmax=pl.col("s").filter(pl.col("tm") >= 0).max(),
        g_decel=pl.col("ds_sm").filter(pl.col("tm").is_between(0, 4)).min(),
        g_cut=pl.col("cut_load").filter(pl.col("tm").is_between(0, 4)).quantile(0.9),
        g_throw_gain=at("s", 1.0, "tpf") - at("s", 0.0, "tpf"),
        g_throw_accel=pl.col("ds_sm").filter(pl.col("tpf").is_between(0, 1.5)).max(),
        n=pl.len(),
    ).filter(pl.col("n") >= 15)
    plays = play.join(pp.lazy() if isinstance(pp, pl.DataFrame) else pp, on=keys, how="inner").collect()
    plays.write_parquet(PROCESSED / "_game_play_feats.parquet")
    return plays


def separation_over_expected(plays: pl.DataFrame) -> pl.DataFrame:
    d = plays.filter(pl.col("separation_at_pass_forward").is_not_null() & pl.col("route_ran").is_not_null()
                     & pl.col("cushion").is_not_null()).to_pandas()
    d["depth"] = np.clip(d["pass_length"].fillna(0) // 5, -1, 6)
    m = smf.ols("separation_at_pass_forward ~ C(route_ran) + C(depth) + cushion + C(lined_up_position)", data=d).fit()
    d["soe"] = m.resid
    return pl.from_pandas(d[["nfl_id", "soe"]]).group_by("nfl_id").agg(sep_oe=pl.col("soe").mean(), n_sep=pl.len())


def player_outcomes(plays: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Per-group game metrics, each aggregated over the plays where that role applies."""
    def med(c: str) -> pl.Expr:
        return pl.col(c).median()

    def p90(c: str) -> pl.Expr:
        return pl.col(c).quantile(0.9)

    routes = plays.filter(pl.col("route_ran").is_not_null())
    rec = routes.group_by("nfl_id").agg(
        pl.len().alias("n_plays"), med("g_v10").alias("release_v10"), med("g_d10").alias("release_d10"),
        p90("g_vmax").alias("route_top_speed"), med("g_decel").alias("route_decel"), p90("g_cut").alias("route_cut"),
    ).join(separation_over_expected(plays), on="nfl_id", how="left")
    yac = plays.filter(pl.col("expected_yards_after_catch").is_not_null()).group_by("nfl_id").agg(
        yac_oe=(pl.col("yards_after_catch") - pl.col("expected_yards_after_catch")).mean(), n_yac=pl.len())
    rec = rec.join(yac.filter(pl.col("n_yac") >= 10), on="nfl_id", how="left").filter(pl.col("n_plays") >= 50)

    rush = plays.filter(pl.col("player_get_off").is_not_null()).group_by("nfl_id").agg(
        pl.len().alias("n_plays"), med("player_get_off").alias("ngs_get_off"),
        med("g_v05").alias("rush_v05"), med("g_d10").alias("rush_d10"), med("g_accel_1s").alias("rush_accel_1s"),
        p90("g_vmax3").alias("rush_vmax3"),
        pl.col("quick_pressure").fill_null(False).cast(pl.Float64).mean().alias("quick_pressure_rate"),
        pl.col("time_to_pressure").is_not_null().cast(pl.Float64).mean().alias("pressure_rate"),
    ).filter(pl.col("n_plays") >= 50)

    pass_pro = plays.filter(pl.col("peak_pressure_probability_allowed").is_not_null()).group_by("nfl_id").agg(
        pl.len().alias("n_plays"), med("g_d10").alias("set_d10"), med("g_v05").alias("set_v05"),
        pl.col("peak_pressure_probability_allowed").mean().alias("peak_pp_allowed"),
        pl.col("pressure_allowed").fill_null(False).cast(pl.Float64).mean().alias("pressure_allowed_rate"),
    )
    runs = plays.filter(pl.col("pass_result").is_null() & pl.col("lined_up_position").is_in(["T", "G", "C"])).group_by("nfl_id").agg(
        p90("g_vmax3").alias("run_block_top_speed3"), med("g_d20").alias("run_block_d20"), pl.len().alias("n_runs"))
    ol = pass_pro.join(runs.filter(pl.col("n_runs") >= 50), on="nfl_id", how="left").filter(pl.col("n_plays") >= 100)

    cov = plays.filter(pl.col("lined_up_position").is_in(["CB", "SLOT_CB", "HIGH_SAFETY", "BOX_SAFETY"]))
    db = cov.group_by("nfl_id").agg(
        pl.len().alias("n_plays"), p90("g_vmax").alias("db_top_speed"),
        pl.col("g_throw_gain").mean().alias("break_speed_gain"), pl.col("g_throw_accel").mean().alias("break_accel"),
        med("g_d20").alias("db_d20_post_snap"), p90("g_cut").alias("db_cut"),
        ((pl.col("tackle").fill_null(0) + 0.5 * pl.col("assist").fill_null(0)).mean()).alias("tackle_rate"),
    ).filter(pl.col("n_plays") >= 100)

    return {"WR": rec, "TE": rec, "EDGE": rush, "IDL": rush, "OL": ol, "CB": db, "S": db}


# ---------------------------------------------------------------- Scan
def demean(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return df.with_columns([(pl.col(c) - pl.col(c).mean().over("nfl_position")) for c in cols])


def pairwise_r(C: np.ndarray, g: np.ndarray, min_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete Pearson r of each column of C with vector g."""
    mask = ~np.isnan(C) & ~np.isnan(g)[:, None]
    n = mask.sum(0)
    Cm = np.where(mask, C, 0.0)
    gm = np.where(mask, g[:, None], 0.0)
    mc = Cm.sum(0) / np.maximum(n, 1)
    mg = gm.sum(0) / np.maximum(n, 1)
    dc = np.where(mask, C - mc, 0.0)
    dg = np.where(mask, g[:, None] - mg, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (dc * dg).sum(0) / np.sqrt((dc**2).sum(0) * (dg**2).sum(0))
    r[n < min_n] = np.nan
    return r, n


def scan_group(grp: str, combine: pl.DataFrame, outcomes: pl.DataFrame, prof: pl.DataFrame) -> pl.DataFrame:
    players = prof.filter(pl.col("nfl_position").is_in(GROUPS[grp])).select("nfl_id", "nfl_position", *STOPWATCH, "combine_weight")
    drills = combine.filter(pl.col("drill_type").is_in(GROUP_DRILLS[grp] + COMMON_DRILLS))
    metrics = ["vmax", "accel", "decel", "cut", "v05", "v10", "d10", "dur"]
    wide = drills.pivot(on="drill_name", index="nfl_id", values=metrics)
    base = players.join(outcomes, on="nfl_id", how="inner").join(wide, on="nfl_id", how="left")
    game_cols = [c for c in outcomes.columns if c not in ("nfl_id",) and not c.startswith("n_")]
    comb_cols = [c for c in wide.columns if c != "nfl_id"]
    # stopwatch as extra combine-side columns (baselines, included in the permutation family too)
    all_c = comb_cols + STOPWATCH
    base = demean(base, game_cols + all_c)
    C = base.select(all_c).to_numpy().astype(float)
    G = base.select(game_cols).to_numpy().astype(float)

    obs = np.stack([pairwise_r(C, G[:, j], MIN_N)[0] for j in range(G.shape[1])], 1)  # (n_c, n_g)
    ns = np.stack([pairwise_r(C, G[:, j], MIN_N)[1] for j in range(G.shape[1])], 1)
    sensor_mask = np.array([c not in STOPWATCH for c in all_c])
    matched = np.array([[c not in STOPWATCH and COMBINE_CLASS[c.split("_")[0]] in MATCHES[GAME_CLASS[g]]
                         for g in game_cols] for c in all_c])
    fam_all = sensor_mask[:, None] & np.isfinite(obs)
    fam_m = matched & np.isfinite(obs)
    n_tests, n_matched = int(fam_all.sum()), int(fam_m.sum())
    maxes = np.empty(N_PERM)
    maxes_m = np.empty(N_PERM)
    for i in range(N_PERM):
        Gp = G[RNG.permutation(G.shape[0])]
        rp = np.abs(np.stack([pairwise_r(C, Gp[:, j], MIN_N)[0] for j in range(G.shape[1])], 1))
        maxes[i] = np.nanmax(np.where(fam_all, rp, np.nan))
        maxes_m[i] = np.nanmax(np.where(fam_m, rp, np.nan))
    rows = []
    for ci, c in enumerate(all_c):
        for gj, gname in enumerate(game_cols):
            r = obs[ci, gj]
            if np.isfinite(r):
                metric, _, drill = c.partition("_")
                rows.append({
                    "group": grp, "combine": c if c in STOPWATCH else f"{drill}:{metric}",
                    "is_stopwatch": c in STOPWATCH, "game": gname, "n": int(ns[ci, gj]), "r": r,
                    "p_adj": float((maxes >= abs(r)).mean()), "n_tests": n_tests,
                    "matched": bool(matched[ci, gj]),
                    "p_adj_matched": float((maxes_m >= abs(r)).mean()) if matched[ci, gj] else None,
                    "n_matched": n_matched,
                })
    return pl.DataFrame(rows).sort(pl.col("r").abs(), descending=True), base


def incremental(grp: str, best: dict, prof: pl.DataFrame, combine: pl.DataFrame, outcomes: pl.DataFrame) -> str:
    """Does the winning sensor metric add over weight + the strongest stopwatch number for that outcome?"""
    drill, metric = best["combine"].split(":")
    c = combine.filter(pl.col("drill_name") == drill).select("nfl_id", pl.col(metric).alias("sensor"))
    df = (prof.filter(pl.col("nfl_position").is_in(GROUPS[grp])).join(outcomes, on="nfl_id").join(c, on="nfl_id")
          .select("nfl_id", "nfl_position", "combine_weight", *STOPWATCH, "sensor", best["game"]))
    # strongest stopwatch for this outcome, by within-position |r|
    dm = demean(df, [*STOPWATCH, best["game"]])
    sw = max(STOPWATCH, key=lambda s: abs(dm.drop_nulls([s, best["game"]]).select(pl.corr(s, best["game"])).item() or 0)
             if dm.drop_nulls([s, best["game"]]).height >= MIN_N else 0)
    d = df.drop_nulls(["sensor", best["game"], "combine_weight", sw]).to_pandas()
    if len(d) < MIN_N:
        return f"n={len(d)} too small with {sw}"
    pos_term = "C(nfl_position) + " if d["nfl_position"].nunique() > 1 else ""
    m0 = smf.ols(f"{best['game']} ~ {pos_term}combine_weight + {sw}", data=d).fit()
    m1 = smf.ols(f"{best['game']} ~ {pos_term}combine_weight + {sw} + sensor", data=d).fit()
    return (f"n={len(d)}  vs weight+{sw}: adjR2 {m0.rsquared_adj:.3f} -> {m1.rsquared_adj:.3f}, "
            f"sensor p={m1.pvalues['sensor']:.3g}, {sw} p={m1.pvalues[sw]:.3g}")


def replicate(grp: str, best: dict, prof: pl.DataFrame, combine: pl.DataFrame, plays: pl.DataFrame) -> str:
    """Same correlation recomputed on odd vs even game_ids (independent plays, same players)."""
    drill, metric = best["combine"].split(":")
    c = combine.filter(pl.col("drill_name") == drill).select("nfl_id", pl.col(metric).alias("sensor"))
    out = []
    for parity in (0, 1):
        o = player_outcomes(plays.filter(pl.col("game_id") % 2 == parity))[grp]
        df = (prof.filter(pl.col("nfl_position").is_in(GROUPS[grp])).select("nfl_id", "nfl_position")
              .join(o.select("nfl_id", best["game"]), on="nfl_id").join(c, on="nfl_id").drop_nulls())
        df = demean(df, ["sensor", best["game"]])
        out.append(f"r={df.select(pl.corr('sensor', best['game'])).item():+.2f} (n={df.height})")
    return "odd games " + out[0] + " | even games " + out[1]


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    prof = load.player_profile()
    cf = PROCESSED / "_combine_player_feats.parquet"
    combine = pl.read_parquet(cf) if cf.exists() else combine_features()
    gf = PROCESSED / "_game_play_feats.parquet"
    plays = pl.read_parquet(gf) if gf.exists() else game_features()
    outs = player_outcomes(plays)

    summary = []
    for grp, positions in GROUPS.items():
        res, _ = scan_group(grp, combine, outs[grp], prof)
        res.write_csv(OUTPUTS / f"r5_{grp}.csv")
        sensor = res.filter(pl.col("matched"))
        best = sensor.row(0, named=True)
        sw = res.filter(pl.col("is_stopwatch")).head(1).row(0, named=True)
        inc = incremental(grp, best, prof, combine, outs[grp])
        rep = replicate(grp, best, prof, combine, plays)
        n_players = outs[grp].join(prof.filter(pl.col("nfl_position").is_in(positions)), on="nfl_id").height
        print(f"\n===== {grp}: {best['n_matched']} matched tests ({best['n_tests']} total); players w/ game data = {n_players}")
        print(sensor.head(6).select("combine", "game", "n", pl.col("r").round(3), "p_adj_matched", "p_adj"))
        print("replication:", rep)
        print(f"best stopwatch: {sw['combine']} -> {sw['game']}  r={sw['r']:.3f} n={sw['n']} p_adj={sw['p_adj']:.3f}")
        print("incremental:", inc)
        summary.append({"group": grp, **{k: best[k] for k in ("combine", "game", "n", "r", "p_adj_matched", "n_matched", "p_adj")},
                        "replication": rep,
                        "best_stopwatch": f"{sw['combine']}->{sw['game']} r={sw['r']:.2f}", "incremental": inc})
    s = pl.DataFrame(summary)
    s.write_csv(OUTPUTS / "r5_summary.csv")
    print("\n", s.select("group", "combine", "game", "n", pl.col("r").round(3), "p_adj_matched", "n_matched", "replication"))


if __name__ == "__main__":
    main()
