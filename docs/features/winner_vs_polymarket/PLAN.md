# Using Polymarket to find value on Winner — plan

**Goal**: find Winner bets that are +EV when priced at Polymarket's fair value. The model is not involved: it doesn't beat the market (`docs/MARKET_EDGE.md`), but the market may beat Winner.

**Bar**: Winner's odds are 1.70–1.85 (break-even 54–59%). A bet counts only at **EV ≥ +3%** against Polymarket's fair price. That covers the cost of Winner's margin plus ~2% of noise in Polymarket's own price (recorded price vs. what you could actually trade at).

## Done

- **Main spread/total, afternoon capture** — negative (2026-11-13, `docs/MARKET_EDGE.md`). Median gap 0.4 pts, 1 of ~320 game-markets positive (+1%). Winner sets these lines off the sharp market.
- **Step 2, Winner's other markets** — closed, nothing to test (2026-11-13 live probe of the list page's "Other" filter and a game page). Winner's NBA offer is spread, total, half-time/full-time and the odd player-points matchup. There are no moneyline, first-half, alternate-line or prop markets. The extras carry 60% (HT/FT) and 24% (player matchup) margin, vs. 11% on spread/total, which no price gap can overcome.

## Steps, in order

### 1. Slate bets priced on Polymarket — live

Replaces the model-edge picks (53-68, −22.1u by 2026-11-19) with the Winner sides that cost the least against Polymarket's fair value (`src/serving/polymarket_pricing.py`, `notifications.polymarket_pricing`).

- **Bar:** expected value ≥ −7% at Winner's odds, ~1–2 bets a night. Every side is expected to lose (Winner's margin); this picks the cheapest. It is not an edge.
- Model picks stay on the slate, labelled "for info, not a bet". Edge calibration stays off.
- The morning report grades these bets against their expected value. If the actual return runs well below expected over ~100 bets, the pricing is off.

### 2. Winner's other markets vs. Polymarket — closed (see Done)

### 3. Evening capture after injury news (slow, ~3–4 weeks)

The afternoon capture sees both markets after they've had all day to agree. Gaps, if they exist, open when news breaks and Winner reprices slower than Polymarket.

**Status: collecting.** `scripts/evening_snapshot.py`, a 22:30 LaunchAgent (runs only if the Mac is awake; no wake schedule). The daily job logs an "afternoon" row per game and the evening job an "evening" row, both priced on Polymarket at capture time, to `outputs/line_snapshots.csv`.

- 22:30 Israel time (15:30 ET), after the NBA's afternoon injury report. Earlier than tip-off on purpose: most tips are 02:00–05:30 Israel time, which is no use for manual betting.
- After ~200 games: compare the Winner-vs-Polymarket gap per capture time.

**Go/no-go**: same +3% / 10% rule. Also check whether the evening gaps cluster on games with injury-status changes since the afternoon. Clustering there means the mechanism is real.

### 4. If step 3 passes: a real edge on the slate

- Raise step 1's bar from −7% to +3% EV, on evening prices.
- Bet real money on these as an edge only after ≥ 100 graded bets show positive ROI at Winner's actual odds.

### 5. Deferred: move the Polymarket collector to Data API v2

`src/polymarket_prices/data_api.py` hits the retired v1 `/trades` endpoint (HTTP 410). This blocks only historical backfill and `market_benchmark` for 2026-27, not steps 1–4 (live prices come from the Gamma/CLOB endpoints). Do it only if a step above needs trade-level history.
