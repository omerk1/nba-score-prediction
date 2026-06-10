import yaml
from enum import Enum
from pathlib import Path
from typing import Optional, Any

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


class ModelConfig(BaseModel):
    random_state: int
    iterations: int
    early_stopping_rounds: int
    tuning: Optional[TuningConfig] = None  # absent when not tuning


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
    injury_features: Optional[InjuryFeaturesConfig] = None


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

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f) or {}

    # Pydantic validates the whole tree here
    return Config.model_validate(config_dict)


def get_config_value(obj: Any, path: str, default: Any = None) -> Any:
    """
    Cleaner implementation of the dot-notation getter.
    Example: get_config_value(cfg, "data_paths.raw_db")
    """
    try:
        for part in path.split('.'):
            # Works for both Pydantic objects and standard dicts
            if isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        return obj
    except (AttributeError, KeyError, TypeError):
        return default