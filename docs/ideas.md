# Project ideas -- ranked by evidence

The prompt rewards going **narrow and deep** and being **usable week-to-week** by a team. Every
idea must link Combine tracking to game tracking or performance; that's a hard pass/fail.
Scripts behind every number are in [`research/`](../research) (run R1 before R2).

## Scorecard (2026-10-16)

All r values are **within position** (position means removed). Rough noise floor: with
100-200 tests at n~80-100, the best null |z| lands around 2.6-2.9. Treat anything below
z~3.3 as a lead, not a finding.

| Combine signal -> game outcome | n | Best result | Verdict |
|---|---|---|---|
| Sensor 40 top speed -> game top speed (all pos) | 396 | r=.33. The 40 drops out when both are in the model (p=.61 vs .005) | **Real** |
| Forty / broad / 10-yd -> DL `player_get_off` | 83-88 | r~.40, z~4 | **Real** (stopwatch) |
| Run-the-Hoop duration -> DL get-off | 94 | r=.40, z=4.1. Adds over forty+weight at p=.04 (n=73) | **Promising** |
| WR route drills -> route-adjusted separation | 114 | 1 hit in 204 tests (z=3.7, wrong-ish sign) | Noise |
| WR Combine route -> same route in games (decel, cut, accel) | 46-84 | Only GO-route top speed r=.33 | Speed only |
| DB transition drills -> in-game break on the throw | 100 | max z=2.7 of 126 tests | Noise |
| OL drills -> pressure allowed / peak pressure prob | 89 | max z=2.3 of 120 tests | Noise |
| Sensor speed -> career snaps beyond draft slot | 414 | Not significant. Signs hint that speed is *over*-drafted (forty p=.049) | Weak lead |

### Why change-of-direction doesn't translate: reliability
Attempt-to-attempt r (median across drills): top speed **.57** (.99 in the 40). Single-frame
peak decel **.27** (.38 smoothed). Peak cut load **.22** (.42 using p90). At 10 Hz, one-rep
accel/decel peaks are mostly noise. The game side is noisy too: DB break-accel split-half r=.22.
Exceptions: the DB backpedal-and-45 transition (decel .80, cut .77) and the gauntlet (~.6)
are genuinely repeatable.

Caveat: some "attempt pairs" are different tasks. Per the host, PASS_RUSH_DRILL left = rip
and right = swim, with no field marking which. Low test-retest there partly reflects
technique, not noise.

## One signal per position (R5, 2026-10-16)

[`research/r5_position_scan.py`](../research/r5_position_scan.py) matches each group's Combine
drills to the *same movement* in game tracking (speed<->speed, burst<->burst/drill time,
cuts<->cuts/drill time, NGS outcomes<->burst/drill time). Correlations are within position, on
smoothed metrics averaged over attempts. `p_adj` is a family-wise permutation p over every
matched test for that group (1,000 shuffles). "Replicates" = same sign and size when game
metrics are rebuilt from odd vs even game_ids. That checks game-side noise only; it uses the
same players, so it doesn't test player sampling.

| Group | Combine metric -> game metric | n | r | p_adj (tests) | Odd / even games | Tier |
|---|---|---|---|---|---|---|
| **EDGE** | 40 sensor top speed -> top speed in first 3 s of rush | 43 | +.54 | **.015** (279) | +.50 / +.48 | **Confirmed** |
| **IDL** | Run-the-Hoop time -> NGS get-off (slower hoop = slower get-off) | 41 | +.55 | **.022** (217) | +.46 / +.56 | **Confirmed** |
| **WR** | 40 sensor top speed -> route top speed (p90) | 76 | +.52 | .086 (378) | +.54 / +.57 | Strong lead |
| **S** | Backpedal-and-45 transition top speed -> in-game top speed | 44 | +.51 | .26 (192) | +.39 / +.47 | Lead |
| **CB** | Line-drill time -> ground covered 2 s after snap (faster drill = more ground) | 39 | -.39 | .55 (192) | -.40 / -.40 | Lead |
| **OL** | Long-pull drill time -> run-block top speed, first 3 s | 68 | -.43 | .68* | -.36 / -.43 | Lead |
| **TE** | Hook-route accel -> YAC over expected | 28 | +.50 | .48 (184) | +.35 / +.43 | Weak (tiny n) |

\*OL's matched-family winner (mirror-drill accel -> pass-set) did not replicate (-.28 / +.02),
so the pull-drill pair is shown instead. It is football-logical and replicates, but it only
passes the looser all-tests family (528 tests).

Notes:
- EDGE: the official 10-yd split is even stronger for the same outcome (r=-.63). The sensor
  top speed adds nothing beyond it (p=.23), so EDGE is "speed travels", not "sensor beats
  stopwatch".
- IDL: hoop time adds over weight + broad jump (adj R^2 .34 -> .44, p=.014). This is the
  cleanest "tracking adds information" result.
- WR: sensor top speed nearly beats the official 40 head-to-head (p=.056, with the 40 at
  p=.32 once both are in the model).
- LB / RB / FB / QB: <=3 players with Combine tracking, so none can be tested.

## Recommended pitch: "What survives the trip from Indy to Sunday"

A reliability-first framework. It turns the null results above into the contribution:

1. **Gate 1 -- Is it a trait?** Test-retest each sensor metric across attempts (and across
   the left/right versions of the same drill). Drop metrics below ~.5.
2. **Gate 2 -- Does it show up in games?** Match the Combine movement to the *same movement*
   in game tracking (top speed -> open-field max speed; hoop/get-off burst -> first 0.5 s after
   the snap; DB transition -> break on the throw). Report within-position r against the
   reliability ceiling sqrt(rel_combine x rel_game).
3. **Gate 3 -- Does it beat the stopwatch?** Incremental value over the official 40, shuttle,
   3-cone and weight.
4. **Deliverable:** a "Translatable Traits" prospect card showing only metrics that pass all
   three gates, with position percentiles. It's actionable for scouts ("ignore single-rep
   agility peaks; sensor top speed beats the 40"), and correlation-vs-ceiling makes it
   rigorous for the data-science judges.

Headline findings so far: **speed travels, first-step burst travels for DL, single-rep
agility doesn't**, and draft slot already prices speed.

## Next experiments (highest value first)

1. **DL first step from tracking, not NGS.** Compute in-game speed and displacement 0.3/0.5/1.0 s
   after the snap for EDGE/IDL across ~44K rushes. Compare to hoop / four-bag / wave burst.
   That gives far more plays than `player_get_off` alone, plus a reliability check on both sides.
2. **Sprint-model burst constant.** Fit `v(t) = v_max(1 - e^{-t/tau})` to each 40 and to
   in-game open-field sprints. `tau` uses the whole curve, so it should be far more reliable
   than a 10-yd split at 10 Hz (r=.49 vs official). Test whether tau beats the 10-yd split
   for DL get-off.
3. **Rip vs swim asymmetry.** Infer rush side from path curvature in PASS_RUSH_DRILL. Does
   a two-way rusher (small rip/swim gap) produce more pressure?
4. **Speed over-drafting.** Draft pick ~ speed (given position and production) vs snaps ~ pick.
   If speed moves pick more than it moves snaps, that's a front-office market inefficiency.
   Needs more power: use all 510 players and the NGS production score as a control.
5. **Multi-rep averaging.** Average agility metrics across *all* drills of a type. If the
   composite's reliability clears .6, re-test WR/DB translation before giving up on agility.

## Football-data gotchas (judges score this)

- Game top speed depends on usage: backup linemen with <100 snaps dominate the
  "low speed retention" list. Condition on role and route.
- Game tracking covers rookies only, so a DB's separation *allowed* is only measurable
  when the receiver is also in the cohort.
- `separation_at_pass_forward` adjusted for route, depth, cushion and alignment explains only
  R^2=.17. Separation is mostly scheme and coverage, not receiver athleticism.
- Small samples (~80-120 per position group): use shrinkage, report CIs, and correct for
  the number of tests.
- Survivorship: only players who got on the field generate game tracking.
- Combine `dis` undercounts and clips include standing time (see [data.md](data.md)).

## Writeup constraints
<= 2,000 words, < 10 figures/tables, public Kaggle notebook attached. Deadline **Jan 6, 2027
11:59 PM UTC**. Scored: Football 30 / Data Science 30 / Writeup 20 / Viz 20.
