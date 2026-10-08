# Big Data Bowl 2027

Working repo for the [NFL Big Data Bowl 2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027):
**uncover non-obvious links between 10 Hz NFL Scouting Combine sensor tracking and regular-season
NFL game performance.**

- 510 rookie prospects (2023-2025 classes), Combine drill tracking + 3 seasons of in-game tracking
- Writeup: <= 2,000 words, < 10 figures/tables, public Kaggle notebook
- Scoring: Football 30% / Data Science 30% / Writeup 20% / Viz 20%
- **Deadline: January 6, 2027, 11:59 PM UTC**

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

Get the data (one-time):

1. Join the competition and accept the rules on Kaggle.
2. Create an API token at <https://www.kaggle.com/settings> (API -> Create New Token).
3. Download (~2.3 GB) and convert to Parquet. It prompts for the token unless `KAGGLE_API_TOKEN` is set:

```bash
uv run python scripts/download_data.py
```

## Usage

```python
from bigdatabowl import load, features as F

profile = load.player_profile()                        # bio + combine testing + career, 1 row/player
attempts = F.combine_attempt_features(load.combine_tracking())   # 1 row per drill attempt
plays = F.game_play_features(load.game_tracking(2024))           # 1 row per player-play (first 3s post-snap)
season = F.game_player_season_features(plays).collect()          # robust per player-season aggregates
```

All loaders return lazy polars frames, so the 9M-row game tracking only materializes what you ask for.

## Layout

```
scripts/download_data.py   Kaggle download -> data/raw -> Parquet in data/processed
src/bigdatabowl/load.py    Lazy loaders for every table
src/bigdatabowl/features.py Speed / burst / braking / change-of-direction metrics (shared Combine + game)
notebooks/                 Analyses (jupytext percent-format .py; run in VS Code or convert to .ipynb)
docs/data.md               Condensed data dictionary + join map
docs/ideas.md              Ranked project ideas and football-data gotchas
tests/                     Synthetic-data checks on the feature math
```

## Data license

Competition data is CC BY-NC 4.0 and is git-ignored. Do not commit it.
