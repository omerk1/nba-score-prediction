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
