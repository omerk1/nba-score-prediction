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


class FeaturesConfig(BaseModel):
    rolling_window: int
    min_games_played: int
    h2h_margin_window: int = 3
    h2h_win_rate_window: int = 5
    targets: list[str]
    exclude: list[str]


class ModelConfig(BaseModel):
    random_state: int = 42


class ImportanceWeightsConfig(BaseModel):
    minutes_share: float = 0.4
    usage_rate: float = 0.4
    pts_share: float = 0.2


class FormulaWeightsConfig(BaseModel):
    out_weight: float = -1.0
    questionable_weight: float = -0.5


class InjuryScorer(str, Enum):
    formula = "formula"
    llm = "llm"


class InjuryFeaturesConfig(BaseModel):
    enabled: bool = False
    scorer: InjuryScorer = InjuryScorer.formula
    db_path: str = "data/raw/injury_features.sqlite"
    llm_model: str = "gemini-1.5-flash"
    api_calls_per_minute: int = 14
    parallel_workers: int = 1
    pdf_era_start: str = "2021-10-01"
    importance_weights: ImportanceWeightsConfig = ImportanceWeightsConfig()
    formula_weights: FormulaWeightsConfig = FormulaWeightsConfig()


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