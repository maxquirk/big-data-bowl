# %% [markdown]
# # 01 -- Does Combine speed show up on Sundays?
#
# First pass at the core question: how much of a prospect's Combine sensor speed / burst
# actually transfers to game tracking, and does the *gap* (Combine minus game) mean anything?
#
# Open as a notebook: `uv run jupytext --to ipynb notebooks/01_speed_translation.py`
# (or just run cells in VS Code -- the `# %%` markers work natively).

# %%
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from bigdatabowl import features as F
from bigdatabowl import load
from bigdatabowl.paths import FIGURES

pl.Config.set_tbl_rows(20)
sns.set_theme(style="whitegrid")

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

# %%
best = F.combine_best_by_drill(attempts)
forty = best.filter(pl.col("drill_type") == "FORTY_YARD_DASH").select(
    "nfl_id",
    combine_top_speed=pl.col("max_speed"),
    combine_max_accel=pl.col("max_accel"),
    sensor_10yd=pl.col("t_10yd"),
    sensor_flying_20=pl.col("flying_20"),
)

# %% [markdown]
# ## Game features -- first 3s after the snap, aggregated per player-season
#
# Start with one season while iterating; the full 9M frames is fine lazily but slower.

# %%
plays = F.game_play_features(load.game_tracking(2023))
season = F.game_player_season_features(plays).collect()
season.sort("game_top_speed", descending=True).head()

# %% [markdown]
# ## Join and compare

# %%
df = (
    season.join(forty, on="nfl_id", how="inner")
    .join(profile.select("nfl_id", "display_name", "nfl_position", "forty"), on="nfl_id")
    .filter(pl.col("n_plays") >= 50)
    .with_columns(speed_retention=pl.col("game_top_speed") / pl.col("combine_top_speed"))
)
print(df.height, "player-seasons")

# %%
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(df.to_pandas(), x="combine_top_speed", y="game_top_speed", hue="nfl_position", ax=ax)
lims = [df["combine_top_speed"].min(), df["combine_top_speed"].max()]
ax.plot(lims, lims, ls="--", c="gray", lw=1)
ax.set(xlabel="Combine 40 sensor top speed (yd/s)", ylabel="Game p99 top speed, first 3s (yd/s)")
fig.savefig(FIGURES / "01_combine_vs_game_top_speed.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Does sensor data beat the stopwatch?
#
# If sensor top speed / flying 20 predicts game speed better than the official `forty`,
# that's the headline "tracking adds context" result the judges asked for.

# %%
df.select(
    pl.corr("forty", "game_top_speed").alias("r_forty"),
    pl.corr("combine_top_speed", "game_top_speed").alias("r_sensor_top_speed"),
    pl.corr("sensor_flying_20", "game_top_speed").alias("r_flying_20"),
    pl.corr("combine_max_accel", "game_p90_accel").alias("r_accel"),
)
