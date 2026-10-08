# Data overview

Official dictionary: <https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/data>.
This is a condensed working reference. 510 rookie prospects (2023-2025 draft classes),
9 CSVs, ~2.3 GB. License: CC BY-NC 4.0 -- data stays local (git-ignored).

| File | Rows | Grain | Notes |
|---|---|---|---|
| `players.csv` | 510 | `nfl_id` | Bio, college, draft round/pick (null = UDFA) |
| `combine_results.csv` | 510 | `nfl_id` | Measurables, stopwatch drills, NGS scores |
| `combine_tracking.csv` | 463K | `event_id` x `time` | 10 Hz drills; `entity_type` PLAYER/BALL |
| `player_career_successes.csv` | 510 | `nfl_id` | Snaps, games, starts, All-Pro, Pro Bowl |
| `player_play.csv` | 316K | `game_id, play_id, nfl_id` | 64 cols: context, EPA/WP, coverage, route, OL/DL metrics |
| `games.csv` | 1,002 | `game_id` | Season, week, REG/POST, teams |
| `game_tracking_{2023,2024,2025}.csv` | 1.4M / 3.0M / 4.3M | `game_id, play_id, nfl_id, time` | 10 Hz, rookie cohort only, has `event` tags |

## Joins

```
players --nfl_id--> combine_results, combine_tracking, player_career_successes, player_play
games   --game_id--> player_play, game_tracking_*
player_play --(game_id, play_id, nfl_id)--> game_tracking_*
```

## Tracking columns

Both sources: `x` (0-120 yd), `y` (0-53.3 yd), `s` speed yd/s, `a` accel yd/s^2, `dis` yd
since last frame, `dir` motion direction deg. Game tracking adds `o` (body orientation) and
`event` (`ball_snap`, `pass_forward`, `pass_arrived`, `first_contact`, `tackle`, ...).
Combine adds `drill_type` (`FORTY_YARD_DASH`, `THREE_CONE_DRILL`, `SHORT_SHUTTLE`,
`SKILL_DRILLS_{WR,DB,OL,DL,TE}`), `drill_name`, `attempt`.

## Useful `player_play` columns by position

- **WR/TE/RB**: `route_ran`, `target`, `separation_at_pass_forward`, `cushion`,
  `yards_after_catch`, `expected_yards_after_catch`
- **OL**: `pressure_allowed`, `time_to_pressure_allowed`, `peak_pressure_probability_allowed`,
  `pass_rushers_encountered`, `sack_allowed`
- **DL/EDGE**: `player_get_off`, `time_to_pressure`, `quick_pressure`, `sack`, `blitzing`
- **DB/LB**: `coverage_assignment`, `team_coverage_type`, `tackle`, `tackle_for_loss`
- **All**: `expected_points_added`, win-probability added, down/distance, formation

## Known quirks
- Column typo in source: `visitor_team_win_probility_added`.
- `three_cone`, `short_shuttle`, `bench_reps` null for opt-outs.
- `time` is an ISO-8601 string; `bigdatabowl.load` parses it to datetime.
