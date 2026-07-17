# Polymarket Data API `/trades` — Step 1 undocumented-capability probe

Tested empirically on 2026-07-08 against `https://data-api.polymarket.com/trades`,
using known-capped markets from the 50-game round (primarily
`nba-hou-orl-2026-02-26`, condition id
`0x7f4e6bbb540d399d875cc70d5251e7b1fd32f32e886295d866dd93bbd222da6d`, and the
4 highest-volume capped games: `nba-lac-gsw-2026-03-02`, `nba-bos-mil-2026-03-02`,
`nba-den-okc-2026-02-27`, `nba-cle-det-2026-02-27`).

## 1. Ascending order — NOT AVAILABLE

Tried `sort`, `order`, `orderBy`, `sortBy`, `sortDirection`, `ascending`,
`sortOrder`, each with values `asc`/`ASC`/`true`/`timestamp_asc`/`oldest`.
All returned HTTP 200 with results **identical** to the unparameterized
request (newest-first, unchanged). None of these keys are recognized by the
API — it silently ignores unknown query params rather than erroring, so this
took direct response-content comparison (not just status code) to rule out.
No ascending-order option exists.

## 2. Time-window filters — NOT AVAILABLE

Tried `from`, `to`, `before`, `after`, `startTs`, `endTs`, `timestamp_gt`,
`timestamp_gte`, `takerOrderFilledAtGt`, `minTimestamp`, `maxTimestamp`
(including the previously-tried `after`/`before` from the 50-game round,
re-confirmed here). All silently ignored — identical results with and
without. No time-windowed pagination is possible against this endpoint.

## 3. Query splitting — WORKS (this is the fix)

- `asset` (single outcome token) — silently ignored, confirmed again (results
  identical regardless of which of the market's two token ids is passed).
  Does not split anything.
- `filterType` — this key IS recognized by the backend (passing an invalid
  value returns `{"error":"invalid filterType TRADES. must be: [CASH TOKENS]"}`,
  confirming a real enum), but passing valid values `CASH` / `TOKENS` returns
  **identical** result sets to each other and to the unfiltered request. It
  does not partition trades or grant a separate offset budget.
- `takerOnly` — also a real recognized param (invalid value gives a
  `strconv.ParseBool` error, confirming it's parsed as a bool), but `true`
  and `false` both return results identical to baseline. No effect, no
  separate budget.
- **`side` (`BUY` / `SELL`) — genuinely filters, and each value gets its own
  independent offset-cap budget of ~3000-3500.** Confirmed on 5 games
  (1 original + 4 highest-volume capped games, up to $6.85M volume):
  - `side=BUY` and `side=SELL` return disjoint, non-overlapping trade sets
    (0 overlap in all samples checked).
  - Paginating `side=BUY` alone still hits the same offset-cap behavior
    (HTTP 400 `"max historical activity offset of 3000 exceeded"` once
    offset > 3000) — BUY volume dominates every game tested, so BUY alone
    remains capped in all 5 samples (missing up to +120 min *after* tip-off
    on the BUY side alone in the worst case, `nba-cle-det-2026-02-27`).
  - Paginating `side=SELL` alone **exhausted naturally under its own
    budget in all 5 samples tested**, i.e. did not hit the cap at all,
    because SELL is the minority side in every NBA moneyline market
    observed (bettors overwhelmingly buy the outcome they favor rather than
    sell it). SELL alone reached back thousands of minutes before tip-off
    (2.6–8.3 days) in every sample.
  - Since the two queries are disjoint (no reconciliation/de-dup needed
    beyond a plain concatenation) and SELL is never capped in the observed
    data, the **union of `side=BUY` + `side=SELL` gives full pre-game +
    in-game coverage** even though BUY alone is still capped: the time
    window BUY is missing (recent pregame → up to ~2h into the game, on the
    highest-volume games) is exactly the window SELL still has data for.
  - Net effect: this at minimum doubles the usable per-market budget
    (~3000 → ~7000 offset-equivalent), and in practice removes the cap
    entirely for these markets because the two sides are asymmetric enough
    that the minority side always finishes under its own cap.

## Decision

Side-splitting (`side=BUY` + `side=SELL`, each paginated to its own cap,
merged) is implemented as the default/only fetch strategy in
`data_api.fetch_all_trades` / `fetch_trades_until_before`, replacing the
single unfiltered-pagination approach. This resolved full pre-game +
in-game coverage on every one of the 5 test markets, including the highest-
volume capped game in the 50-game batch ($6.85M `nba-lac-gsw-2026-03-02`).

**Goldsky subgraph fallback (Step 2) was not implemented** — per the task's
own conditional ("only if Step 1 is insufficient"), and because Step 1's
side-split fix passed validation on every capped game it was tested against
including the worst cases by volume. See the Step 5 before/after table for
the full-50-game confirmation. If a future, even-higher-volume game someday
has BOTH sides individually exceed ~3500 trades, the subgraph fallback
described in the design doc would still be the right next step — the
`data_api.py` module now records `buy_capped`/`sell_capped` per game so that
condition is detectable and visible (as `was_capped=True` in games.csv) if
it ever occurs, without silently truncating data.
