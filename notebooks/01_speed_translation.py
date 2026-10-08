# %% [markdown]
# # 01 -- Does Combine speed show up on Sundays?
#
# First pass at the core question: how much of a prospect's Combine sensor speed transfers to
# game tracking, and does the sensor tell us anything the stopwatch 40 doesn't?
#
# Open as a notebook: `uv run jupytext --to ipynb notebooks/01_speed_translation.py`
# (or just run cells in VS Code -- the `# %%` markers work natively).

# %%
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf

from bigdatabowl import features as F
from bigdatabowl import load
from bigdatabowl.paths import FIGURES

pl.Config.set_tbl_rows(20)
sns.set_theme(style="whitegrid")
FIGURES.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Prospect table (bio + stopwatch testing + career outcomes)

# %%
profile = load.player_profile()
profile.select(
    "nfl_id", "display_name", "nfl_position", "draft_overall_pick", "forty", "ten_yd_split",
    "ngs_athleticism_score", "career_games_started",
).head()

# %% [markdown]
# ## Combine sensor features -- one row per drill attempt

# %%
attempts = F.combine_attempt_features(load.combine_tracking())
attempts.group_by("drill_type").agg(pl.len(), pl.col("max_speed_mph").median()).sort("len")

# %% [markdown]
# ### Sanity check: sensor 40 vs official 40
#
# Sensor clocks start at first movement, so expect a constant offset vs the electronic time;
# what matters is that the ranking agrees.

# %%
best = F.combine_best_by_drill(attempts.filter(pl.col("drill_type") == "FORTY_YARD_DASH"))
forty = best.select(
    "nfl_id",
    sensor_top_speed=pl.col("max_speed"),
    sensor_max_accel=pl.col("max_accel"),
    sensor_10yd=pl.col("t_10yd"),
    sensor_40yd=pl.col("t_40yd"),
    sensor_flying_20=pl.col("flying_20"),
)
check = forty.join(profile.select("nfl_id", "forty", "ten_yd_split"), on="nfl_id").drop_nulls("forty")
check.select(
    pl.len(),
    pl.corr("forty", "sensor_40yd").alias("r_40"),
    pl.corr("ten_yd_split", "sensor_10yd").alias("r_10"),
    (pl.col("sensor_40yd") - pl.col("forty")).mean().alias("offset_40_s"),
)

# %% [markdown]
# ## Game features -- full plays (12s post-snap), all three seasons
#
# Top speed needs the whole play; the default 3s window is for burst/get-off questions.
# Per player-season we take the p99 of per-play max speed (robust to tracking glitches).

# %%
plays = F.game_play_features(load.game_tracking(), post_snap_window=12.0)
season = F.game_player_season_features(plays).collect()

career = (
    season.filter(pl.col("n_plays") >= 30)
    .group_by("nfl_id")
    .agg(pl.col("game_top_speed").max(), pl.col("n_plays").sum())
)
career.sort("game_top_speed", descending=True).head()

# %% [markdown]
# ## Join and compare

# %%
df = (
    career.join(forty, on="nfl_id", how="inner")
    .join(profile.select("nfl_id", "display_name", "nfl_position", "forty"), on="nfl_id")
    .drop_nulls(["forty", "sensor_top_speed", "game_top_speed"])
    .with_columns(speed_retention=pl.col("game_top_speed") / pl.col("sensor_top_speed"))
)
print(df.height, "players with sensor 40 + official 40 + >=30 game snaps in a season")

# %%
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(df.to_pandas(), x="sensor_top_speed", y="game_top_speed", hue="nfl_position", ax=ax)
lims = [df["sensor_top_speed"].min(), df["sensor_top_speed"].max()]
ax.plot(lims, lims, ls="--", c="gray", lw=1)
ax.set(xlabel="Combine 40 sensor top speed (yd/s)", ylabel="Game top speed, p99 of plays (yd/s)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
fig.savefig(FIGURES / "01_combine_vs_game_top_speed.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Does the sensor beat the stopwatch -- *within position*?
#
# Raw correlations (~0.8) are mostly "linemen are slower than corners." The real test is
# whether each metric explains game speed after controlling for position, and whether the
# sensor adds information once the official 40 is already known.

# %%
pdf = df.to_pandas()
models = {
    "position only": "game_top_speed ~ C(nfl_position)",
    "+ official 40": "game_top_speed ~ C(nfl_position) + forty",
    "+ sensor top speed": "game_top_speed ~ C(nfl_position) + sensor_top_speed",
    "+ both": "game_top_speed ~ C(nfl_position) + forty + sensor_top_speed",
}
rows = []
for name, formula in models.items():
    m = smf.ols(formula, data=pdf).fit()
    row = {"model": name, "n": int(m.nobs), "adj_r2": round(m.rsquared_adj, 4)}
    for k in ("forty", "sensor_top_speed"):
        if k in m.params:
            row[f"{k}_coef"] = round(m.params[k], 3)
            row[f"{k}_p"] = float(f"{m.pvalues[k]:.3g}")
    rows.append(row)
pl.DataFrame(rows)

# %% [markdown]
# ### Within-position residuals
#
# Strip out position means from everything and compare the leftover signal directly.

# %%
resid = df.with_columns(
    [
        (pl.col(c) - pl.col(c).mean().over("nfl_position")).alias(f"{c}_wp")
        for c in ("game_top_speed", "forty", "sensor_top_speed")
    ]
).filter(pl.len().over("nfl_position") >= 5)

r = resid.select(
    pl.len(),
    pl.corr("forty_wp", "game_top_speed_wp").alias("r_within_forty"),
    pl.corr("sensor_top_speed_wp", "game_top_speed_wp").alias("r_within_sensor"),
)
print(r)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, col, label in (
    (axes[0], "forty_wp", "Official 40 vs position avg (s)"),
    (axes[1], "sensor_top_speed_wp", "Sensor top speed vs position avg (yd/s)"),
):
    sns.regplot(resid.to_pandas(), x=col, y="game_top_speed_wp", ax=ax, scatter_kws={"s": 12, "alpha": 0.5})
    ax.set(xlabel=label, ylabel="Game top speed vs position avg (yd/s)")
fig.suptitle("Within-position signal: stopwatch vs sensor")
fig.savefig(FIGURES / "01_within_position_stopwatch_vs_sensor.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ### Speed retention leaders / laggards (position-adjusted)

# %%
ret = df.with_columns(
    retention_vs_pos=pl.col("speed_retention") - pl.col("speed_retention").mean().over("nfl_position")
).filter(pl.len().over("nfl_position") >= 5)
cols = ["display_name", "nfl_position", "forty", "sensor_top_speed", "game_top_speed", "speed_retention", "n_plays"]
print("Most speed carried into games:")
print(ret.sort("retention_vs_pos", descending=True).select(cols).head(10))
print("Least:")
print(ret.sort("retention_vs_pos").select(cols).head(10))
