# HTML slate

**Daily job**: one Telegram message, a self-contained HTML page
(`outputs/slates/<day>.html`, gitignored) with a short caption. The page
lists recommended bets first (strong, then marginal), then a per-game
drill-down. Each bet card and game card shows who is out (raw ESPN snapshot
for the slate day via `src/serving/injury_lookup.py`; "unavailable" when the
day has no snapshot). If the page send fails, the full text slate is sent
instead. A hosted link was not built: the page is private betting analysis
and Telegram's document upload needs no hosting.

**Recommended bet** = priced side with any positive model edge over
break-even (`notifications.recommend_min_edge: 0.0`); edge >= 5% is flagged
strong. One definition (`src/serving/recommended_bets.py`).
