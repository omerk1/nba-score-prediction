"""
Deterministic formula-based injury impact scorer.

Replaces the LLM when scorer = "formula" in config.
Impact = sum of (importance × status_weight) for each injured player.
Weights are configurable via injury_features.formula_weights in config.yaml.
"""


def score_team(players: list[dict], importance_map: dict[str, float], weights) -> dict:
    """
    Compute team injury impact without an LLM.

    Args:
        players: [{"player_name": str, "status": str, "reason": str, "days_out": int}]
        importance_map: {player_name: importance_score (0–1)}
        weights: FormulaWeightsConfig with out_weight and questionable_weight

    Returns:
        {"impact_score": float, "n_out": int, "n_questionable": int, "star_out": bool}
    """
    n_out = sum(1 for p in players if p.get("status") == "Out")
    n_questionable = sum(1 for p in players if p.get("status") in ("Questionable", "Day-To-Day"))

    impact = 0.0
    for p in players:
        importance = importance_map.get(p["player_name"], 0.1)
        status = p.get("status", "")
        if status == "Out":
            impact += importance * weights.out_weight
        elif status in ("Questionable", "Day-To-Day"):
            impact += importance * weights.questionable_weight

    star_out = any(
        p.get("status") == "Out" and importance_map.get(p["player_name"], 0) > 0.5
        for p in players
    )

    return {
        "impact_score": round(impact, 3),
        "n_out": n_out,
        "n_questionable": n_questionable,
        "star_out": star_out,
    }
