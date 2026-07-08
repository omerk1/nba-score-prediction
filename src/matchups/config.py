"""
Lightweight config access for the A7 style-matchup module.

Rationale: configs/config.yaml's `Config` pydantic model (src/utils/config_loader.py)
is a frozen schema that doesn't know about `style_matchup` and per project rules we do
not modify any source file outside src/matchups/. So this module reads the
`style_matchup` block directly from the YAML (raw dict, no validation) and reuses
`load_config()` for `injury_features.severity_weights` per the project's config-reuse
instruction (avoid a second injury_severity_multipliers block).
"""

from pathlib import Path
from typing import Any

import yaml

from src.utils.config_loader import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# Read-only symlinked/shared DBs — never open for write from this module.
NBA_API_DB = str(PROJECT_ROOT / "data" / "raw" / "nba_api.sqlite")
INJURY_DB = str(PROJECT_ROOT / "data" / "raw" / "injury_features.sqlite")

# Our own additive cache DB, kept under outputs/ (a permitted write location).
CACHE_DB = str(PROJECT_ROOT / "outputs" / "a7_matchups_cache.sqlite")

DEFAULT_STYLE_MATCHUP_CONFIG: dict[str, Any] = {
    "fingerprint_window": 20,
    "decay_halflife": 5,
    # Halflife (in consecutive qualifying team-games) for calibration.py's decay-weighted
    # injury-impact calibration -- down-weights Out-events by their position in a
    # continuous absence streak (fix for the perimeter_specialist sign-flip artifact
    # caused by long single-player absences dominating the sample). See
    # calibration.py's DEFAULT_DECAY_HALFLIFE_GAMES and docs/A7_PHASE_LOG.md
    # "Decay-weighted injury-impact calibration" for the halflife grid explored.
    "injury_calibration_decay_halflife_games": 20,
    "encoding": "hand_picked",
    "pca_n_components": 5,
    "similarity_method": "cosine",
    "similarity_threshold": 0.70,
    "knn_k": 30,
    "min_confidence_sample": 10,
    "full_confidence_sample": 50,
    "low_confidence_fallback": "h2h",
    "archetype_method": "percentile",
    # NOTE: design doc default was a single shared 0.75/0.25 cut applied to all four
    # archetypes. Phase 0 explored each archetype's thresholds INDEPENDENTLY (a grid
    # per archetype, not one shared knob) plus a genuinely different taxonomy
    # (KMeans clustering on raw PPG/AST/REB/BLK/STL/FG%, k=4..8 — see phase log).
    # Clustering mostly recovered a playing-time tier split (bench/rotation/starter/
    # star), not stylistic groups, because player_stats_cache has no minutes/usage-
    # rate stats to separate "how much" from "how" — so the percentile approach
    # (already era-adjusted via season-relative ranking) was kept as primary.
    # Per-archetype tuning: facilitator/scorer loosened 0.75/0.25 -> 0.65/0.35
    # (design default left them at 1-2 player-seasons total, no statistical power).
    # rim_protector kept at 0.75/0.75 (908 player-seasons, already ample).
    # perimeter_specialist loosened blk 0.25->0.30, stl 0.75->0.70 (31 -> 80
    # player-seasons; still selective for a "low rim protection, high steal" wing).
    # Added `combo` (NEW — design doc's facilitator/scorer are mutually exclusive by
    # construction, which drops genuine dual-threat playmaker-scorers). Set high
    # (0.85/0.85, n=496) since PPG+AST both scale with playing time/usage, so a
    # lower bar (e.g. 0.70) mostly just re-selects "played a lot of minutes" rather
    # than a distinct stylistic archetype.
    # Considered and REJECTED: a third "versatile_defender" mid-band bucket between
    # rim_protector and perimeter_specialist. Tested BLK/STL percentile bands
    # (0.40-0.75) — captured a diffuse 11-15% "everyone in the middle" group with no
    # distinct separation, not a real archetype; not added given only 6 stats
    # available (no minutes/usage/matchup data to define a genuine third defensive
    # profile).
    "archetype_percentiles": {
        "facilitator": {"ast_pct": 0.65, "ppg_pct": 0.35},
        "scorer": {"ppg_pct": 0.65, "ast_pct": 0.35},
        # combo: the Critique & Bug-Fix Pass redefined this using usage_rate + assist-RATE
        # (usage_pct/ast_rate_pct, from player_importance's minutes_per_game/usage_rate,
        # available 2018-19 onward) as the PRIMARY definition -- a direct measurement of
        # "high-usage dual-threat playmaker", replacing the original v1 workaround (raw
        # ppg_pct/ast_pct both >=0.85, deliberately set high specifically to avoid
        # re-selecting players who simply accumulate a lot of raw PPG/AST from playing
        # heavy minutes). Empirically nearly reproduces the old population at this
        # threshold (377 vs 376 player-seasons, ~79% overlap, near-identical usage_rate
        # mean) -- see phase log. ppg_pct/ast_pct are KEPT as the fallback definition for
        # the 2016-17/2017-18 seasons player_importance doesn't cover (classify_archetypes
        # uses usage_pct/ast_rate_pct whenever available, else falls back to these).
        "combo": {"ppg_pct": 0.85, "ast_pct": 0.85, "usage_pct": 0.80, "ast_rate_pct": 0.80},
        "rim_protector": {"blk_pct": 0.75, "reb_pct": 0.75},
        "perimeter_specialist": {"blk_pct": 0.30, "stl_pct": 0.70},
    },
    "injury_impact": {
        "facilitator": {"assist_rate": -0.15, "pace_score": -0.1},
        "scorer": {"three_pt_reliance": 0.1, "paint_activity": 0.1},
        "combo": {},  # populated empirically in Phase 0 across all 5 fingerprint metrics
        "rim_protector": {"defensive_rating": 2.5, "paint_activity": -0.15},
        "perimeter_specialist": {"defensive_rating": 1.5},
    },
}


def load_style_matchup_config() -> dict[str, Any]:
    """Read the `style_matchup` block from configs/config.yaml (raw dict).

    Falls back to hardcoded design-doc defaults for any missing keys so the
    module works even before config.yaml has been amended.
    """
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f) or {}
    cfg = dict(DEFAULT_STYLE_MATCHUP_CONFIG)
    cfg.update(raw.get("style_matchup", {}))
    return cfg


def severity_weights() -> dict[str, float]:
    """Reuse injury_features.severity_weights from the main config (do not duplicate)."""
    main_cfg = load_config()
    sw = main_cfg.injury_features.severity_weights
    return {"severe": sw.severe, "moderate": sw.moderate, "minor": sw.minor}


def doubtful_weight() -> float:
    main_cfg = load_config()
    return main_cfg.injury_features.doubtful_weight
