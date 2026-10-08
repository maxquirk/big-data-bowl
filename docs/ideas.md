# Project ideas

The prompt rewards going **narrow and deep** (one position group, one drill, one trait) and
being **usable week-to-week** by a team. Every idea below links Combine tracking to game
tracking/performance -- that's a hard pass/fail requirement.

## Tier 1 -- strongest pitches

### 1. Speed Retention Index ("does the Combine speed show up?")
Combine sensor top speed vs. in-game p99 top speed per player. Ratio = how much athleticism
is *usable* in pads/traffic. Questions: is retention position-specific? Does low retention
predict busts better than a slow 40? Starter: [`notebooks/01_speed_translation.py`](../notebooks/01_speed_translation.py).

### 2. Brake & Re-Accelerate (WR/DB change of direction)
From route/backpedal drills: max decel into the break, speed bled through the cut, time to
re-accelerate. Link to `separation_at_pass_forward` (WR) and separation allowed (DB). The
prompt literally lists this -- judges will see many, so the edge has to be in the metric
quality (e.g. normalize by entry speed, separate left vs right cuts).

### 3. First-Step Get-Off (trench)
DL/OL drill burst over first 0.5-1.0s vs. in-game `player_get_off`, `time_to_pressure`,
`peak_pressure_probability_allowed`. Cleanest causal story: first-step quickness is the same
movement in both contexts.

### 4. Stopwatch vs Sensor -- where does tracking add info?
Residualize sensor metrics on the official times (forty, shuttle, 3-cone) and test whether the
*residual* predicts NFL outcomes. "Two guys both ran 4.45 -- here's how to tell them apart."
Very front-office friendly.

## Tier 2 -- creative / differentiated

- **Movement fingerprints / comps** -- embed each prospect's drill trajectories (speed-curve
  shape, turn profile) and find nearest-neighbor NFL comps from earlier draft classes.
- **Acceleration profile curves** -- fit a mono-exponential sprint model (`v(t) = v_max (1 - e^{-t/tau})`)
  to the 40; `tau` is a cleaner "burst" constant than the 10-yd split. Compare tau in-game.
- **Fatigue / attempt-to-attempt consistency** -- variance across attempts as a proxy for
  movement reliability; does it predict snap share?
- **Draft value residuals** -- which sensor traits are underpriced by draft slot (pick vs
  `career_games_started` / snaps)?
- **Lateral vs linear athletes** -- cluster OL/DL by shuttle/3-cone sensor traces and map
  clusters to run-block vs pass-pro usage.

## Football-data gotchas to handle (judges score this)

- Game top speed depends on role/usage: a WR on go routes hits top speed more often than a
  slot WR. Condition on route/play type, or use high quantiles over many plays.
- Small samples: rookies with few snaps. Use shrinkage / mixed models (position random effects).
- Survivorship: only players who got on the field generate game tracking.
- `play_direction`: normalize x/y so offense always moves the same way before spatial work.
- Tracking glitches: prefer p95/p99 over raw max.

## Writeup constraints
<= 2,000 words, < 10 figures/tables, public Kaggle notebook attached. Deadline **Jan 6, 2027
11:59 PM UTC**. Scored: Football 30 / Data Science 30 / Writeup 20 / Viz 20.
