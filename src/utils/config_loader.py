from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict

# --- Schema Definitions ---


class DataPathsConfig(BaseModel):
    raw_db: str
    processed: str
    features: str
    models: str


class DatasetsLoadingConfig(BaseModel):
    data_start_date: str
    train_start_date: str
    train_end_date: str
    validation_start_date: str
    validation_end_date: str
    test_start_date: str
    test_end_date: Optional[str] = None
    allowed_season_types: Optional[list[str]] = None
    context_season_types: Optional[list[str]] = None


class CVFoldConfig(BaseModel):
    """One expanding-window CV fold: train = datasets_loading.train_start_date
    (shared, fixed across every fold) through train_end_date; validation = the
    next season; test = the season after that. `cv.folds` in configs/config.yaml
    must be ordered oldest -> newest -- validated mechanically (not just by
    convention) by src/evaluation/cv_harness.py's validate_fold_definitions.
    See CLAUDE.md's "Project Rules (ML experimentation)" section."""

    name: str
    train_end_date: str
    validation_start_date: str
    validation_end_date: str
    test_start_date: str
    test_end_date: str


class CVConfig(BaseModel):
    folds: list[CVFoldConfig] = []


class FeaturesConfig(BaseModel):
    rolling_windows: list[int]
    naive_rolling_baseline: int
    min_games_played: int
    h2h_margin_window: int
    h2h_win_rate_window: int
    targets: list[str]
    exclude: list[str]


class TuningConfig(BaseModel):
    n_trials: int
    depth: list[int]
    learning_rate: list[float]
    l2_leaf_reg: list[float]
    min_data_in_leaf: list[int]
    subsample: list[float]
    colsample_bylevel: list[float]


class TargetFormulation(str, Enum):
    home_away = "home_away"
    diff_total = "diff_total"


class ModelConfig(BaseModel):
    random_state: int
    iterations: int
    early_stopping_rounds: int
    tuning: Optional[TuningConfig] = None  # absent when not tuning
    # EXPERIMENTS.md section 3.3 (target reformulation): home_away (old status
    # quo) fits MultiRMSE on [PTS_home, PTS_away] directly; diff_total fits on
    # [POINT_DIFF, TOTAL_POINTS] instead, aligning the training loss with what
    # the composite metric actually rewards (diff-dominant, total at half
    # weight -- see ScorePredictor._to_training_targets for the mechanism).
    # ADOPTED as the new committed default (configs/config.yaml: diff_total) --
    # beat home_away on val_score in all 5 CV folds individually. Schema
    # default here stays home_away (the conservative fallback if this field is
    # ever omitted from a config), same convention as
    # StyleMatchupConfig.raw_features_enabled's schema-vs-yaml split.
    target_formulation: TargetFormulation = TargetFormulation.home_away
    target_lambda_weight: float = 0.5


class EloTuningConfig(BaseModel):
    n_trials: int
    k_factor: list[float]
    home_advantage: list[float]
    season_regression: list[float]


class EloFeaturesConfig(BaseModel):
    enabled: bool
    initial_rating: float
    k_factor: float
    home_advantage: float
    mov_multiplier: bool
    season_regression: float
    tuning: Optional[EloTuningConfig] = None


class ImportanceWeightsConfig(BaseModel):
    minutes_share: float
    usage_rate: float
    pts_share: float


class SeverityWeightsConfig(BaseModel):
    severe: float
    moderate: float
    minor: float


class InjuryScorer(str, Enum):
    formula = "formula"
    llm = "llm"


class InjuryMissingValueStrategy(str, Enum):
    zero_fill = "zero_fill"
    native_nan = "native_nan"


class InjuryFeaturesConfig(BaseModel):
    enabled: bool
    scorer: InjuryScorer
    db_path: str
    llm_model: str
    api_calls_per_minute: int
    parallel_workers: int
    pdf_era_start: str
    importance_weights: ImportanceWeightsConfig
    severity_weights: SeverityWeightsConfig
    doubtful_weight: float
    # E4 (EXPERIMENTS.md, session rs_20260808_1): does imputation strategy for
    # missing injury rows matter more than the feature itself being weak?
    # zero_fill (status quo) vs. native_nan (let CatBoost's own missing-value
    # handling operate) -- gated so the ablation ships disabled-from-adoption
    # by default per CLAUDE.md's ablation-gated feature workflow.
    missing_value_strategy: InjuryMissingValueStrategy = InjuryMissingValueStrategy.zero_fill


class StyleMatchupConfig(BaseModel):
    """A7 module (src/matchups/). See docs/a7_style_matchup_design.md and
    docs/a7_phase_log.md for how these values were chosen. injury_impact keys
    are archetype names; each maps to a dict of {fingerprint_metric: delta},
    so it's left as a plain nested dict rather than a fixed per-archetype
    model. `enabled` gates feature_builder.py's _add_style_matchup_features
    (mirrors EloFeaturesConfig/InjuryFeaturesConfig's own `enabled` field) —
    added by the KNN-score integration test. `raw_features_enabled` gates
    the separate, independently-toggleable _add_style_fingerprint_features
    (added by the raw-fingerprint feature redesign: raw per-team fingerprint
    components + explicit home-vs-away differentials, no KNN similarity search
    involved — a different feature set from `enabled`'s KNN-lookup score, not a
    replacement for it)."""

    enabled: bool
    raw_features_enabled: bool = False
    # Track C pace/possession swap-in test (docs/NEW_DATA_FEASIBILITY.md): gates
    # feature_builder.py's addition of official_pace/official_poss (nba_api's own
    # PACE/POSS, src/matchups/pace_possession.py) alongside pace_score -- a
    # separate flag from raw_features_enabled so this candidate can be ablated
    # independently of the already-adopted 18 raw-fingerprint columns. Requires
    # raw_features_enabled=true (checked in feature_builder.py) since these are
    # added inside the same method. Ships disabled by default per CLAUDE.md's
    # ablation-gated feature workflow.
    official_pace_enabled: bool = False
    fingerprint_window: int
    decay_halflife: float
    encoding: str
    similarity_method: str
    similarity_threshold: float
    knn_k: int
    min_confidence_sample: int
    full_confidence_sample: int
    low_confidence_fallback: str
    archetype_method: str
    injury_impact_calibrated: bool
    injury_impact: dict[str, dict[str, float]]


class OnOffSplitsConfig(BaseModel):
    """Player on/off court splits (see docs/on_off_splits_decisions.md /
    docs/on_off_splits_log.md). Data source is nba_api's TeamPlayerOnOffSummary
    endpoint, called with DateTo set to enforce a leakage-safe historical cutoff
    (LastNGames was tested and found to NOT compose with date filters — see the
    decisions doc). `enabled` gates feature_builder.py's
    _add_on_off_splits_features, mirroring InjuryFeaturesConfig/StyleMatchupConfig's
    own `enabled` field."""

    enabled: bool
    db_path: str
    checkpoint_cadence_days: int
    min_on_off_minutes: float = 0.0


class SeasonMotivationConfig(BaseModel):
    """Season motivation / seeding-incentive features (see
    docs/SEASON_MOTIVATION_DECISIONS.md). No `db_path` of its own — reads
    `data_paths.raw_db` (standings/schedule, from `game`) and
    `injury_features.db_path` (`player_importance`/`player_injuries`)
    directly. `enabled` gates `_add_season_motivation_features`."""

    enabled: bool = False
    # Gates motivation_score/games_to_clinch_*/recent_minutes_trend_score --
    # the original Phase 1 design, which did NOT clear the ablation bar (see
    # log FINAL SUMMARY). Separate from `enabled` so a signal that DID clear
    # the bar (preferred_opponent_delta) can ship without these non-adopted
    # columns. Defaults True (opt-out, not opt-in) so nothing changes unless
    # explicitly set False.
    motivation_score_enabled: bool = True
    playoff_line_seed: int = 10
    direct_playoff_seed: Optional[int] = None
    direct_playoff_weight: float = 0.5
    roster_behavior_weight: float = 1.0
    min_importance_games: int = 5
    recent_trend_lookback_weeks: int = 4
    # Behavior-based signals (independently toggleable, same convention as
    # StyleMatchupConfig's enabled/raw_features_enabled pair) -- each requires
    # `enabled` above to be true AND its own flag, so a signal that clears the
    # ablation bar can be turned on without re-enabling everything else.
    performance_vs_expectation_enabled: bool = False
    performance_vs_expectation_window: int = 10
    opponent_adjusted_form_enabled: bool = False
    opponent_adjusted_form_window: int = 10
    preferred_opponent_delta_enabled: bool = False
    preferred_opponent_delta_window_games: int = 20


class Config(BaseModel):
    """
    Main Configuration Object.
    Pydantic automatically handles nested dicts to objects.
    """

    model_config = ConfigDict(frozen=True)  # Makes config immutable

    data_paths: DataPathsConfig
    datasets_loading: DatasetsLoadingConfig
    features: FeaturesConfig
    model: ModelConfig
    elo_features: Optional[EloFeaturesConfig] = None
    injury_features: Optional[InjuryFeaturesConfig] = None
    style_matchup: Optional[StyleMatchupConfig] = None
    on_off_splits: Optional[OnOffSplitsConfig] = None
    season_motivation: Optional[SeasonMotivationConfig] = None
    cv: Optional[CVConfig] = None


# --- Loader Functions ---


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """
    Load configuration from a YAML file.
    Supports automatic validation and nested object creation.
    """
    if config_path is None:
        # PROJECT_ROOT/configs/config.yaml
        config_path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path.absolute()}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f) or {}

    # Pydantic validates the whole tree here
    return Config.model_validate(config_dict)


def get_config_value(obj: Any, path: str, default: Any = None) -> Any:
    """
    Cleaner implementation of the dot-notation getter.
    Example: get_config_value(cfg, "data_paths.raw_db")
    """
    try:
        for part in path.split("."):
            # Works for both Pydantic objects and standard dicts
            if isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        return obj
    except (AttributeError, KeyError, TypeError):
        return default
