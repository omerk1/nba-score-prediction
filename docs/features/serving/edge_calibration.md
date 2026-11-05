# Edge calibration and edge tracking

**Problem**: spread/total cover probabilities price the raw prediction against the line with a fixed residual curve, treating every point of disagreement with the market as real. Strong bets (edge ≥ 5%) claimed ~67% and hit ~50%, live (29-28) and on the held-out 2025-26 season (405-410). Evidence: `docs/MARKET_EDGE.md`, 2026-11-04 entry.

**Decision**: the slate keeps showing the model's own edges (raw, as before). The goal is to track honestly whether those edges are real, not to hide them. Calibrated pricing is built but ships **off**.

**Tracking (always on)**:

- Each rec logs the unshrunk `*_probability_raw` / `*_edge_raw` next to the priced ones. They're identical while calibration is off.
- Morning report record = the model's picks (raw edge ≥ 5%), continuous with every row logged before this change: 32-39, −14.5u over 98 games on 2026-11-05.
- Morning report "Is the model beating the line?": per market, over graded games of the currently loaded model, the slope of (actual − line) on (prediction − line) through the origin, i.e. the share of the model's disagreement with the line that came true. Near 0 means the edges are noise; clearly above 0 means they're real.

**Calibrated pricing (`notifications.edge_calibration.enabled`, default off)**:

- Prices from `line + weight × (prediction − line)`, `weight = clip(slope − z·se, 0, 1)` with z = 1, and 0 below `min_games` = 50.
- Only spread/total cover probabilities and edges move. Point predictions, heatmap and moneyline are unchanged, and logged predictions stay raw, so the weight is always learned from unshrunk predictions.
- A retrained model starts at weight 0 (filtered on `model_trained_at`). A fitting failure prices at the line.
- When on, the morning report adds a row for what the calibrated slate recommended.

**Numbers on 2026-11-05**:

| Data | Spread slope | Total slope |
|---|---|---|
| Live, 98 / 91 games | −0.52 ± 0.30 | +0.28 ± 0.33 |
| Held-out 2025-26, 1,217 games | −0.07 ± 0.09 | +0.04 ± 0.10 |

Both give weight 0. If enabled today, every side would price near 50%, below the ~55.6% break-even, and the slate would show no bets. Rough scale (se ≈ 13 / (4·√n)): a model whose true slope is 0.5 would get a weight of ~0.2 after 100 graded games and ~0.3 after 300.

**Not done**: moneyline probabilities aren't calibrated against the market (earlier isotonic work in `docs/MARKET_EDGE.md` found no edge there either).
