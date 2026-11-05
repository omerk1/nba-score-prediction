# Edge calibration

**Problem**: spread/total cover probabilities priced the raw prediction against the line with a fixed residual curve, treating every point of disagreement with the market as real. Strong bets (edge ≥ 5%) claimed ~67% and hit ~50%, live (29-28) and on the held-out 2025-26 season (405-410). Evidence: `docs/MARKET_EDGE.md`, 2026-11-04 entry.

**Change** (`src/serving/edge_calibration.py`, `notifications.edge_calibration` in `configs/config.yaml`):

- Per market, over graded games of the currently loaded model, slope of (actual − line) on (prediction − line) through the origin: the share of the model's disagreement that came true.
- Pricing uses `line + weight × (prediction − line)`. `weight = clip(slope − z·se, 0, 1)` with z = 1, and 0 below `min_games` = 50 graded games.
- The point prediction, margin heatmap and moneyline probability are unchanged; only spread/total cover probabilities and their edges move. The logged `predicted_diff`/`predicted_total` stay raw, so the weight is always learned from unshrunk predictions.
- A retrained model starts at weight 0 (filtered on `model_trained_at`). A calibration failure prices at the line, never at the raw prediction.
- Morning report: new "Is the model beating the line?" section with the slope ± se, games counted, and the weight the slate prices at.
- Tracking stays on the model, not the slate: each rec also logs the unshrunk `*_probability_raw` / `*_edge_raw`, and the morning report's record covers the model's own picks (raw edge ≥ 5%), continuous with every row logged before this change (32-39, −14.5u over 98 games at launch). A second row shows what the calibrated slate recommended.

**State at launch (2026-11-05)**: weight 0 for both markets.

| Data | Spread slope | Total slope |
|---|---|---|
| Live, 98 / 91 games | −0.52 ± 0.30 | +0.28 ± 0.33 |
| Held-out 2025-26, 1,217 games | −0.07 ± 0.09 | +0.04 ± 0.10 |

At weight 0 every side prices near 50%, below the ~55.6% break-even, so the slate shows no recommended bets until the model's record earns them. Rough scale (se ≈ 13 / (4·√n) at today's spreads): a model whose true slope is 0.5 would be priced at weight ~0.2 after 100 graded games and ~0.3 after 300.

**Not done**: moneyline probabilities aren't calibrated against the market (no line to shrink toward in the same way; earlier isotonic work in `docs/MARKET_EDGE.md` found no edge there either).
