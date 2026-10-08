# Big Data Bowl 2027

Kaggle analytics hackathon: link Combine 10 Hz tracking to NFL game performance. See README.md
for setup, docs/data.md for schema, docs/ideas.md for direction.

## Commands
- `uv sync --extra dev` -- install
- `uv run pytest -q` -- feature-math tests (synthetic data, no download needed)
- `uv run ruff check .` -- lint
- `uv run python scripts/download_data.py` -- needs Kaggle token + accepted rules

## Conventions
- Package is `bigdatabowl`, NOT `bdb` -- `bdb` shadows Python's stdlib debugger module.
- Use polars LazyFrames; never `.collect()` full game tracking (9M rows) without filtering.
- Data under `data/` is CC BY-NC and git-ignored. Never commit CSV/Parquet.
- Prefer p95/p99 over raw max for tracking-derived peaks (sensor glitches).
- Signed accel/decel comes from `ds_dt` (diff of `s`); the raw `a` column is unsigned magnitude.
- Heading differences must be wrapped to [-180, 180) (`features._wrap_deg`).
