# Injury Report PDF Extraction — Scope

Status: scoped 2026-09-20, not started. Fixes a live data defect and, as a
secondary goal, is the one place in this repo where document extraction by a
language model is a natural tool rather than a substitute for a better one.
Related: `docs/PIPELINE_AUDIT.md` (2026-09-17 addendum, the report-date
finding), `docs/MARKET_EDGE.md` (2026-08-17, the dropped-players finding).

## 1. The defects

Three separate problems in `src/news_scraping/scrapers/nba_injury_pdf.py`, in
descending order of impact.

**D1. The game-date column is never read.** `_parse_pdf` maps columns by name
and requests only `Team`, `PlayerName`, `CurrentStatus` and `Reason`. The PDF's
own `Game Date` column (index 0, documented in the module docstring) is
discarded, so every row is filed under the date of the report that contained
it. The scraper keeps only the last report of each day (`scrape_log.report_time`
is `11PM` or `11_45PM` for all 814 dates), and that report covers both the games
just played and the next day's games.

Measured against per-game player logs:

| listings | refer to a game on report date D | on D+1 |
|---|---:|---:|
| Questionable / Doubtful | 0.8% | 99% |
| Out | 62% | 38% |

`_add_injury_features` joins `injury_features` to games on **equal** dates. So a
game on date G is currently given the counts from the 11PM report of G, which is
published after G tipped off and mostly describes G+1. Not outcome leakage,
since injury reports encode roster decisions, but the feature is attached to the
wrong game.

**D2. Rows are silently dropped.** `docs/MARKET_EDGE.md` records the Lakers'
three clearly-listed Out players on 2025-10-21 vanishing from extraction, with
`n_out` falling back to 0, which is indistinguishable from a healthy roster.
Cause is table detection: `_extract_table` takes `page.extract_table()` when it
returns more than one row and otherwise falls back to a text-alignment strategy,
and a continuation page is accepted only when its first non-empty row has
exactly 4 cells. Anything else is skipped with no record.

**D3. No extraction telemetry.** Nothing records how many rows a page yielded,
whether a page was skipped, or whether a team name failed to resolve
(`logger.debug` only). A silent regression is invisible.

## 2. What does and does not need a language model

Stated plainly, because the first version of this scope overstated the case.

- **D1 needs no model at all.** Read `GameDate` on header pages, carry it
  forward across continuation pages exactly as `current_abbr` is already carried
  forward, store it as a new column. Roughly 15 lines. The cost is re-parsing
  814 documents, not the logic.
- **D2 is mostly geometry.** `pdfplumber` exposes word-level bounding boxes, so
  columns can be recovered by clustering on x-position and rows on y-position,
  without depending on ruling lines. Dedicated table-structure models exist too.
  A language model is one option here, not the only one.
- **What a model genuinely buys** is tolerance for format drift (the NBA changed
  the filename scheme in December 2025 and the column layout has shifted before)
  and coverage of the long tail without enumerating every layout. That is real
  but modest, and it costs determinism in a repo that values byte-identical
  reruns.

**Design conclusion: deterministic parser first, model as a fallback and as a
referee.** The model is invoked only for pages the deterministic path fails on
or disagrees about, which keeps the common path reproducible and the cost near
zero, while still producing a genuine extraction pipeline with a real
evaluation.

## 3. Plan

**Phase A — deterministic fix and telemetry (no model).**
1. Read and carry forward `GameDate`; add `game_date` (the game) alongside the
   existing column, renamed `report_date`, in `player_injuries`. Keep both so
   the old behaviour stays reconstructable.
2. Add an `extraction_log` table: per (report date, page), rows found, whether a
   header was detected, unresolved team names, and the strategy used.
3. Re-parse from cached PDFs where possible, else re-fetch the 814 reports.
4. Rebuild `injury_features` keyed on the game date, and point
   `_add_injury_features` at it.

**Phase B — labeled evaluation set.**
Hand-label every row of 20 PDFs, stratified across the old and new formats and
deliberately including 2025-10-21 (the known failure). Roughly 1,200 rows.
This is the ground truth both extractors are scored against.

**Phase C — model extractor as fallback and referee.**
Send page text (or the page image, tested both ways) with a strict schema, one
object per row: game date, team, player, status, reason. Cache by page hash.
Compare against the deterministic parser on the labeled set:

- row recall, the D2 metric, are listed players missing
- field accuracy per field, exact after normalization
- game-date assignment accuracy, the D1 metric
- disagreement rate on the full 814 documents, where the model flags pages the
  deterministic path likely got wrong

**Phase D — downstream ablation.**
The corrected dates change a live feature, so this goes through the normal
workflow: baseline against treatment under `--protocol cv`, logged to
`outputs/experiments_v2.csv`, written up in `docs/EXPERIMENTS.md`. Expect
movement on folds 3 to 5 only, since injury coverage starts in 2021-22.

## 4. Gates

| Phase | Gate to continue |
|---|---|
| A | Extracted game dates agree with the schedule for at least 99% of rows; total row count per report is unchanged or higher. |
| B | Two independent passes over the 20 PDFs agree, disagreements resolved by reading the PDF. |
| C | The model must beat the deterministic parser on row recall on the labeled set, or it is used only as a disagreement flag, not as an extractor. |
| D | Composite score does not regress on folds 1 and 2, where nothing should change. |

## 5. Risks

- **Re-fetching 814 PDFs** hits an NBA CDN. Use the existing backoff, cache
  every document to disk on first fetch so later phases never re-download, and
  accept that some old reports may have been removed.
- **Rebuilding `injury_features` changes a live feature.** Write to new rows
  keyed by scorer or a new column rather than overwriting, so the current
  champion stays reproducible until the ablation decides.
- **Another agent is working in this repo.** `injury_features.sqlite` is shared
  through the worktree; do phase A writes against a copy and swap only after the
  ablation.

## 6. Effort

Phase A is a few hours, most of it the re-fetch. Phase B is a focused hour of
labeling. Phase C is a few hours. Phase D is one CV run. Phases A and D carry
the model value; B and C carry the extraction-pipeline experience.

---

## Phase A results (2026-09-21)

Deterministic only, no model. `data/raw/injury_dates.sqlite` now holds
`player_injuries_dated` (game date, report date, team, player, status, reason,
date source) and `extraction_log` (per-report telemetry). The live
`injury_features.sqlite` is untouched.

| measure | before | after |
|---|---:|---:|
| rows extracted from 814 reports | 76,778 | 96,950 |
| rows with a usable game date | 47% | 100% |
| pages skipped whole | 1,461 | 27 |
| Clippers rows in the whole history | 0 | 2,920 |
| extracted date matches a scheduled game | not measured | 99.93% |

**D1, the game-date column, fixed.** Read and carried forward across rows and
continuation pages. Reports split 55.5% / 44.5% between the report's own date
and the next day, confirming that the old equal-date join attached most rows to
the wrong game.

**D2, dropped rows, fixed, and it was worse than recorded.** Three compounding
causes. (a) The reports write "LA Clippers" where nba_api says "Los Angeles
Clippers", so *every* Clippers listing in the entire history failed to resolve
and vanished; that team has looked healthy since 2021. (b) From 2023-24 the
ruled-table extraction returns only a header row, so the text-alignment fallback
takes over and emits only the columns with content on that page; the parser
accepted exactly four columns and skipped everything else whole, losing 1,461
pages. Columns are now located by finding the status column from its contents
and deriving the rest from its position, which handles three, four and five
column pages and makes the team column optional. (c) The documented
2025-10-21 Lakers case now returns LeBron James, Maxi Kleber and Adou Thiero,
where the old table has no rows at all.

**Dating the newer reports.** The 2023-24+ fallback drops the date column
entirely, so those rows are dated from the schedule: a report dated D covers D
and D+1, and a team scheduled on exactly one of them is unambiguous. Scored
against the 37,881 rows whose PDFs do carry a real date:

| branch | rows | accuracy |
|---|---:|---:|
| unambiguous | 34,397 | 1.0000 |
| back-to-back tie-break | 3,413 | 0.9786 |
| unresolved | 71 | n/a |

**A rule that was measured, not reasoned.** The first tie-break assumed a late
report previews the next day and chose D+1. Scored, that branch was 2.1%
accurate; the answer is D at 97.9%, taking overall agreement from 91.0% to
99.6%. The wrong rule had a plausible justification and would have shipped
without the check.

**The gate did not catch any of this.** "Extracted dates match a scheduled game"
passed at 99.8% on the broken version, because rows that were silently dropped
never produced a date to test. What worked was telemetry on what got discarded,
plus scoring the rule against ground truth. Recorded here because it is the
transferable lesson, not the row counts.

**The 71 unresolved rows are correct extractions**, not failures: 2021-12-18
listings for games postponed during the COVID outbreak, which therefore have no
scheduled game.

**Still open in phase A**: rebuilding `injury_features` on the corrected dates
and pointing `_add_injury_features` at them. That changes a live feature, so it
goes with phase D's ablation rather than ahead of it.
