"""
Deterministic formula-based injury impact scorer.

Replaces the LLM when scorer = "formula" in config.
Impact = sum of (importance × status_weight) for each injured player.
Severity weights are configurable via injury_features.severity_weights in config.yaml.

team_deficit captures non-linear roster depletion:
    team_deficit = missing_quality / available_quality
The ratio grows super-linearly as more important players are missing.
"""

import re

_GLEAGUE_RE = re.compile(r'gleague|g league|two.?way|on assignment', re.IGNORECASE)
_SEVERE_RE = re.compile(r'\b(surgery|fracture|torn|tear|rupture|acl|ligament)\b', re.IGNORECASE)
_MODERATE_RE = re.compile(r'\b(sprain|strain|bone bruise|bonebruise)\b', re.IGNORECASE)


def is_gleague(reason: str) -> bool:
    return bool(_GLEAGUE_RE.search(reason or ''))


def classify_severity(reason: str, severity_weights) -> float:
    r = reason or ''
    if _SEVERE_RE.search(r):
        return severity_weights.severe
    if _MODERATE_RE.search(r):
        return severity_weights.moderate
    return severity_weights.minor


def compute_team_deficit(players: list[dict], importance_map: dict[str, float], severity_weights, doubtful_weight: float) -> float:
    """
    Non-linear roster depletion metric.

    missing_quality / available_quality grows super-linearly when multiple
    important players are absent — losing a 2nd star hurts more than the first
    because the denominator (remaining quality) shrinks too.
    """
    total_quality = sum(importance_map.values())
    if total_quality <= 0:
        return 0.0

    missing_quality = 0.0
    for p in players:
        if is_gleague(p.get('reason', '')):
            continue
        importance = importance_map.get(p['player_name'], 0.0)
        status = p.get('status', '')
        if status == 'Out':
            missing_quality += importance * classify_severity(p.get('reason', ''), severity_weights)
        elif status == 'Doubtful':
            missing_quality += importance * classify_severity(p.get('reason', ''), severity_weights) * doubtful_weight

    available_quality = max(total_quality - missing_quality, 0.01)
    return round(missing_quality / available_quality, 4)


def score_team(players: list[dict], importance_map: dict[str, float], severity_weights, doubtful_weight: float = 0.8) -> dict:
    """
    Compute team injury impact without an LLM.

    Args:
        players: [{"player_name": str, "status": str, "reason": str}]
        importance_map: {player_name: importance_score (0–1)}
        severity_weights: SeverityWeightsConfig
        doubtful_weight: fraction of Out severity applied to Doubtful players

    Returns:
        {"n_out": int, "n_questionable": int, "team_deficit": float}
    """
    n_out = sum(1 for p in players if p.get("status") == "Out")
    n_questionable = sum(1 for p in players if p.get("status") in ("Questionable", "Day-To-Day"))

    return {
        "n_out": n_out,
        "n_questionable": n_questionable,
        "team_deficit": compute_team_deficit(players, importance_map, severity_weights, doubtful_weight),
    }
