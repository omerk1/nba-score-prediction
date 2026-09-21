"""Play-by-play (PBP) data: raw event collection from nba_api's PlayByPlayV3
and reduction of the event stream into a per-possession table.

The possession table is the substrate for context-conditioned aggregates
(garbage-time-filtered margin, luck-adjusted margin, clutch splits, lineup
continuity, shot-quality). Nothing in this package feeds the model directly --
features built on it go through the usual ablation-gated workflow.
"""
