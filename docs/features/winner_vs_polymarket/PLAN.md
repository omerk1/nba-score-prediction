# Using Polymarket to find value on Winner — plan

**Goal**: find Winner bets that are +EV when priced at Polymarket's fair value. The model is not involved: it doesn't beat the market (`docs/MARKET_EDGE.md`), but the market may beat Winner.

**Bar**: Winner's odds are 1.70–1.85 (break-even 54–59%). A bet counts only at **EV ≥ +3%** against Polymarket's fair price. That covers the cost of Winner's margin plus ~2% of noise in Polymarket's own price (recorded price vs. what you could actually trade at).

## Done

- **Main spread/total, afternoon capture** — negative (2026-11-13, `docs/MARKET_EDGE.md`). Median gap 0.4 pts, 1 of ~320 game-markets positive (+1%). Winner sets these lines off the sharp market.
- **Step 2, Winner's other markets** — closed, nothing to test (2026-11-13 live probe of the list page's "Other" filter and a game page). Winner's NBA offer is spread, total, half-time/full-time and the odd player-points matchup. There are no moneyline, first-half, alternate-line or prop markets. The extras carry 60% (HT/FT) and 24% (player matchup) margin, vs. 11% on spread/total, which no price gap can overcome.

## Steps, in order

### 1. Turn on edge calibration in the slate (independent hygiene)

- Flip `notifications.edge_calibration.enabled` to `true`. The raw spread/total edges hit 49.7% while claiming 67% (2026-11-04 entry).
- **Threshold: no change.** An edge is already `p − Winner break-even`, so `recommend_min_edge: 0` still means "beats Winner's margin". With the learned weight near 0, spread/total recommendations mostly disappear. That is the intended result.
- Winner lists no NBA moneyline, so every slate pick is a spread/total. With calibration on, the slate shows no bets until the model earns a positive weight.

### 2. Winner's other markets vs. Polymarket — closed (see Done)

### 3. Evening capture after injury news (slow, ~3–4 weeks)

The afternoon capture sees both markets after they've had all day to agree. Gaps, if they exist, open when news breaks and Winner reprices slower than Polymarket.

1. Add a second Winner capture at ~22:30 Israel time (15:30 ET), after the NBA's afternoon injury report. Earlier than tip-off on purpose: most tips are 02:00–05:30 Israel time, which is no use for manual betting.
2. Price Polymarket at the same moment, and log both captures to the same file.
3. Collect ~200 games, then rerun the comparison per capture time.

**Go/no-go**: same +3% / 10% rule. Also check whether the evening gaps cluster on games with injury-status changes since the afternoon. Clustering there means the mechanism is real.

### 4. If step 2 or 3 passes: paper-trade before real money

- Add a Polymarket-priced edge to the slate, separate from the model edge, with its own threshold starting at +3% EV and tuned only on graded paper bets.
- Live Polymarket prices come from the Gamma/CLOB endpoints, which still work.
- Log and grade every flagged bet. Bet real money only after ≥ 100 graded bets show positive ROI at Winner's actual odds.

### 5. Deferred: move the Polymarket collector to Data API v2

`src/polymarket_prices/data_api.py` hits the retired v1 `/trades` endpoint (HTTP 410). This blocks only historical backfill and `market_benchmark` for 2026-27, not steps 2–4. Do it only if a step above needs trade-level history.
