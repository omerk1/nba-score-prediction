# Play-by-Play Possession Table — Build Log

Substrate for context-conditioned features (garbage-time-filtered and
luck-adjusted margin, clutch splits, lineup continuity, shot quality). No
feature reads it yet; `pbp.enabled` stays `false`. Code: `src/pbp/`,
`scripts/backfill_pbp.py`, `tests/test_pbp_possessions.py`. Data:
`data/raw/pbp.sqlite` (local, gitignored) — `pbp_events` (raw PlayByPlayV3
rows), `possessions`, `possession_game_summary`, `pbp_fetch_log`.

## 2026-09-17 — First season pulled (2024-25 regular season)

**Endpoint**: nba_api `PlayByPlayV3`. Schema identical on 2016-17 and 2024-25
games. ~493 events/game, 1.20 s/game including the 0.7 s rate-limit sleep;
1,225/1,225 games fetched, 0 errors, ~25 min. Full 8-season backfill is
therefore ~3.5 h, one overnight resumable run.

**Possession definition**: maximal run of core events (FGA, non-technical FT,
turnover, rebound) by one team within a period; ends at the terminal event or
the opponent's defensive rebound. Technical FTs tallied separately. Only
scoring rows update the running score — Instant Replay rows carry stale
scores and broke reconciliation on 17 games until excluded. Team-level rows
(shot-clock turnovers) put the team id in `personId`; normalised.

**Validation, 243,520 possessions**:

| Check | Result |
|---|---|
| Points reconcile with `game` box score | 1,225 / 1,225 games |
| Durations tile each period exactly | yes (tested) |
| Possessions per team-game | mean 99.4 (p5 92, p95 108) — matches official pace |
| Points per possession | 1.142 (league ORtg ≈ 114) |
| Turnover rate / oreb-per-possession | 14.4% / 12.7% |
| Outcome mix | made FG 39.6, miss 36.4, TOV 14.4, FT 9.2, period end 0.4 (%) |

**Lineups** (both five-man units known): 98.3% of possessions; 74.4% of games
fully complete, 2.3% below 90%. V3 gives no period starters and names the
incoming substitute by text only, so starters are inferred from first
appearances and names resolved against the game roster. Four resolution
failure modes were found and fixed on this season: shared surnames (`L. James`
via the initial column), three-way shared surnames (`Jay. Williams` via a
first-name prefix from nba_api's static player list), generational suffixes
(`Butler` for `Butler III`), and diacritics (`Schroder`/`Schröder`,
`Pöltl`/`Poeltl`). The residual (556 of 16,403 substitutions) is almost
entirely deep-bench players who enter and then generate no event of any kind,
which nothing in the feed can identify — concentrated in garbage time, which
most planned features filter out anyway. Fixable later by joining box-score
starters if lineup-conditioned features turn out to need it.

**Next**: build the first context-conditioned aggregates on this season,
pre-screen each on window-to-window persistence before any CV run, then
backfill the remaining seasons for the survivors.

## 2026-09-17 — First aggregates + persistence screen (2024-25 only)

`src/pbp/aggregates.py` (33 candidates), `scripts/pbp_persistence_screen.py`,
results in `outputs/pbp_persistence_screen.csv`. Each team's season is split
into consecutive N-game blocks; every aggregate is a pooled ratio over that
block's possessions (summed numerator / summed denominator, never a mean of
per-game ratios). `r_self` = pooled correlation between block b and b+1;
`r_next_margin` = block b's value against block b+1's points margin, which is
what a pre-game feature actually needs. Pairs never cross a season boundary.
Nothing here is a model result — this is a pre-CV screen.

**Bar**: beat `margin_pg` (raw points margin, the thing every existing rolling
feature already encodes) at `r_next_margin`. Ranked at N=10, all three block
sizes agree on the ordering:

| Stat | r_next_margin (5 / 10 / 20) | r_self (10) |
|---|---|---|
| net_rtg_luckadj_nogarbage | 0.511 / **0.592** / 0.602 | 0.588 |
| net_rtg_luckadj | 0.489 / 0.574 / 0.563 | 0.591 |
| net_rtg_nonclutch | 0.469 / 0.569 / 0.653 | 0.547 |
| net_rtg_nogarbage | 0.467 / 0.562 / 0.655 | 0.549 |
| margin_pg (bar) | 0.457 / 0.556 / 0.642 | 0.556 |
| efg | 0.286 / 0.362 / 0.438 | 0.526 |
| top_lineup_share | 0.119 / 0.235 / 0.343 | 0.385 |
| three_rate | 0.008 / 0.016 / 0.022 | 0.723 |
| pace48 | 0.009 / −0.005 / 0.039 | 0.581 |
| lead_changes | 0.022 / −0.024 / −0.058 | 0.008 |

Negative `r_next_margin` on `def_rtg`/`tov_rate` is the expected sign (lower
is better), not a failure.

**One candidate clears the bar**: luck-adjusted, garbage-filtered net rating
(own and opponent 3P/FT makes replaced with league-rate expectations, blowout
possessions dropped). Cluster-bootstrap over teams, 2,000 resamples:
Δr vs. `margin_pg` = **+0.054** at N=5 (95% CI [−0.002, +0.116], P(Δ>0)=0.97)
and +0.035 at N=10 (CI [−0.031, +0.102], P=0.85). Real but modest, and the
interval touches zero — this is a "worth a CV run", not a result. Decomposed:
the opponent-luck half does the work; a sweep of the own-luck weight gives
0.502 / **0.524** / 0.511 at weights 0.0 / 0.5 / 1.0 (N=5), so full removal of
a team's own shooting luck overshoots. Carry weight 0.5 into the ablation.
Garbage-filtering alone is worth only +0.011.

**Three groups fail, and the pattern is consistent**:

- **Style/shot-mix persists but predicts nothing.** `three_rate` (r_self 0.72),
  `pace48` (0.58), `mid_rate`, `rim_rate`, `oreb_per100` are among the most
  stable stats measured and all sit at |r_next_margin| < 0.06. They describe
  how a team plays, not how well. Already represented in
  `style_fingerprint_features` anyway.
- **Order-aware/momentum stats fail on persistence itself.** `lead_changes`
  (r_self 0.008), `largest_run_for` (0.079), `net_rtg_clutch` (0.120, and only
  70 pairs since clutch possessions are sparse at N=10). This is the direct
  test of the "context and what happened before/after" hypothesis and it comes
  back null: within-game sequence structure does not carry to the next block.
  `time_leading_share` is the exception at 0.497, but it correlates 0.85 with
  net rating — a restatement of "this team is better", not new information.
- **Lineup continuity persists but is dominated.** `top_lineup_share` /
  `top3_lineup_share` reach r_self 0.39-0.48 with r_next_margin 0.23, well
  under the bar; bootstrap Δ vs. `margin_pg` = −0.32 (CI [−0.51, −0.11]).

**Methodology note, important for reading the table**: `r_within` (persistence
after demeaning per team) is ≈0 for nearly every stat at every block size.
Pooled persistence is almost entirely between-team spread — good teams stay
good. That is fine for a pre-game feature, which exploits exactly that spread,
but it means none of these stats measure *change* within a team, and no
"team is heating up" reading of the pooled numbers is supported.

**Next**: one ablation on the single surviving candidate (luck-adjusted
garbage-filtered net rating, own-luck weight 0.5) as a rolling pre-game
feature, screened on the last 3 folds first. The other 7 seasons only need
backfilling if that clears — which is the point of screening before pulling
~3.5 h of data.

## 2026-09-18 — Shot quality and lineup-conditioned efficiency (both fail)

The two families the first screen left untested, flagged there as the gap in
its own coverage. `src/pbp/shots.py` (shot quality, reads the event stream
directly since the possession table keeps only each possession's last shot),
plus lineup-conditioned ratings in `aggregates.py`. Same bar as before: beat
`margin_pg` at `r_next_margin`. Cluster-bootstrap over teams, 1,000 resamples,
Δ against `margin_pg`:

| Candidate | r_next_margin (5/10/20) | r_self (10) | Δ vs margin at N=5 |
|---|---|---|---|
| net_rtg_luckadj_nogarbage (prior winner) | 0.511 / 0.592 / 0.602 | 0.588 | **+0.054** [−0.000, +0.113] |
| net_rtg_xadj, own weight 0.25 | — | — | +0.014 [−0.028, +0.063] |
| net_rtg_bench_lineups | 0.346 / 0.469 / 0.582 | 0.413 | −0.111 [−0.161, −0.046] |
| net_rtg_top5_lineups | 0.303 / 0.344 / 0.437 | 0.221 | −0.154 [−0.222, −0.088] |
| pps_vs_x (shot-making over expectation) | 0.256 / 0.344 / 0.433 | 0.562 | −0.200 [−0.297, −0.093] |
| opp_xpps (shot quality allowed) | −0.123 / −0.205 / −0.247 | 0.602 | −0.580 [−0.748, −0.352] |
| xpps (own shot selection) | 0.042 / −0.001 / −0.060 | 0.523 | — |
| net_rtg_top5_minus_bench | 0.019 / 0.011 / 0.011 | −0.018 | — |

**Shot quality: the skill/luck split is real, and neither half helps.**
`opp_xpps`, the quality of shot a defence concedes, has the highest defensive
persistence measured anywhere in this work (r_self 0.602, against 0.360 for
raw `def_rtg`), while `opp_pps_vs_x`, whether opponents then miss more than
their locations imply, sits at 0.170. That is the predicted skill-versus-luck
contrast and it came out cleanly. But shot selection on both ends is
orthogonal to *winning*: `xpps` predicts next-block margin at −0.001. Teams
differ stably in shot quality and those differences do not move the scoreboard
at this horizon.

**Extending the winning adjustment to all shots makes it worse.** Valuing
every field goal at its location expectation, rather than correcting only 3P
and FT against one league rate, drops `r_next_margin` from 0.592 to 0.420 at
full weight (Δ vs margin −0.105 at N=5, CI excludes zero). A weight sweep puts
the best variant at 0.25 own-luck weight and only +0.014, inside noise. The
reading: shooting above a location expectation is substantially skill, so
removing it discards signal, whereas three-point and free-throw variance
around a single league rate is mostly noise and removing it helps. The simpler
adjustment wins and stays the only survivor.

**Lineup-conditioned efficiency fails, including in an instructive direction.**
Restricting net rating to a team's five most-used lineups *lowers* persistence
(0.221 vs 0.549 overall) because the restriction cuts sample size and the
top-five set itself turns over between blocks. Bench-lineup rating outperforms
top-lineup rating (0.469 vs 0.344), the opposite of the intuition behind the
candidate, and correlates 0.85 with plain net rating, so it is mostly a
restatement. `net_rtg_top5_minus_bench`, the starter-versus-bench gap, has
r_self −0.018: not a stable team property at all.

**Pipeline cross-check (mechanical, not a finding)**: `pps` computed from raw
shot rows and `2 × efg` computed from the possession table's own counts agree
to 0.000000 across all 240 team-blocks. Two independent paths through the
data, so the possession table's shot accounting matches the event stream
exactly.

**Standing conclusion after both screens**: 45 candidates tested, one beats
raw points margin, by +0.054 with an interval that touches zero. Combined with
the near-zero within-team persistence found in the first screen, the evidence
is that this possession data carries little the existing feature set does not
already hold. The single ablation on the survivor is still worth running; a
learned encoder over the same possessions is not indicated by anything
measured here.
