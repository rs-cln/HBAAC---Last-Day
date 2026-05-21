"""Single source of truth configuration for the HBAAC pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def load_config(
    config_path: Optional[Path | str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    run_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Load YAML config and apply optional overrides."""

    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = _deep_update(cfg, overrides)
    if run_mode is not None:
        cfg.setdefault("run_control", {})["run_mode"] = run_mode
    assert_horizon_dates(cfg)
    return cfg


def resolve_path(path_value: str | Path, root: Path = ROOT) -> Path:
    """Resolve config paths relative to project root."""

    path = Path(path_value)
    return path if path.is_absolute() else root / path


def config_path(cfg: Mapping[str, Any], section: str, key: str) -> Path:
    """Resolve a path from the active config."""

    return resolve_path(cfg[section][key])


def assert_horizon_dates(cfg: Mapping[str, Any]) -> None:
    """Validate competition horizon/date constants."""

    comp = cfg["competition"]
    train_end = pd.Timestamp(comp["train_end"])
    validation_start = pd.Timestamp(comp["validation_start"])
    validation_end = pd.Timestamp(comp["validation_end"])
    evaluation_start = pd.Timestamp(comp["evaluation_start"])
    evaluation_end = pd.Timestamp(comp["evaluation_end"])
    public_horizon = int(comp["public_horizon"])
    forecast_horizon = int(comp["forecast_horizon"])

    if validation_start != train_end + pd.Timedelta(days=1):
        raise AssertionError("validation_start must be one day after train_end.")
    if evaluation_start != validation_end + pd.Timedelta(days=1):
        raise AssertionError("evaluation_start must be one day after validation_end.")
    validation_days = (validation_end - validation_start).days + 1
    evaluation_days = (evaluation_end - evaluation_start).days + 1
    if validation_days != public_horizon:
        raise AssertionError("validation date span must equal public_horizon.")
    if evaluation_days != public_horizon:
        raise AssertionError("evaluation date span must equal public_horizon.")
    if validation_days + evaluation_days != forecast_horizon:
        raise AssertionError("validation + evaluation spans must equal forecast_horizon.")


def print_active_config(cfg: Optional[Mapping[str, Any]] = None) -> None:
    """Print active config in a readable format."""

    active = cfg if cfg is not None else DEFAULT_CONFIG
    print(pformat(active, sort_dicts=False))


DEFAULT_CONFIG = load_config()

PATHS_CFG = DEFAULT_CONFIG["paths"]
RUN_CFG = DEFAULT_CONFIG["run_control"]
COMP_CFG = DEFAULT_CONFIG["competition"]
METRIC_CFG = DEFAULT_CONFIG["metric"]
FEATURE_CFG = DEFAULT_CONFIG["features"]
VALIDATION_CFG = DEFAULT_CONFIG["validation"]
TRAINING_CFG = DEFAULT_CONFIG["training"]

LEGACY_RAW_DIR = resolve_path(PATHS_CFG["legacy_raw_dir"])
DATA_DIR = ROOT / "data"
RAW_DIR = resolve_path(Path(PATHS_CFG["raw_train_path"]).parent)
PROCESSED_DIR = resolve_path(PATHS_CFG["processed_dir"])
FEATURE_DIR = resolve_path(PATHS_CFG["features_dir"])
MODEL_DIR = resolve_path(PATHS_CFG["models_dir"])
REPORT_DIR = resolve_path(PATHS_CFG["reports_dir"])
SUBMISSION_DIR = resolve_path(PATHS_CFG["submissions_dir"])
ARCHIVE_DIR = resolve_path(PATHS_CFG["archive_dir"])
BEFORE_ENSEMBLE_DIR = SUBMISSION_DIR / "before_ensemble"
NOTEBOOK_DIR = resolve_path(PATHS_CFG["notebooks_dir"])

TRAIN_FILE = Path(PATHS_CFG["raw_train_path"]).name
SAMPLE_SUBMISSION_FILE = Path(PATHS_CFG["sample_submission_path"]).name

TRAIN_START = str(COMP_CFG["train_start"])
TRAIN_END = str(COMP_CFG["train_end"])
FORECAST_START = str(COMP_CFG["validation_start"])
FORECAST_END = str(COMP_CFG["evaluation_end"])
VALIDATION_START = str(COMP_CFG["validation_start"])
VALIDATION_END = str(COMP_CFG["validation_end"])
EVALUATION_START = str(COMP_CFG["evaluation_start"])
EVALUATION_END = str(COMP_CFG["evaluation_end"])
FORECAST_HORIZON = int(COMP_CFG["forecast_horizon"])
SUBMISSION_HORIZON = int(COMP_CFG["public_horizon"])

RANDOM_SEED = int(RUN_CFG["random_seed"])
RUN_MODE = str(RUN_CFG["run_mode"])
SMOKE_N_SKUS = int(RUN_CFG["smoke_n_skus"])
SMOKE_TOP_PROFIT_SKUS = int(RUN_CFG["smoke_top_profit_skus"])
SMOKE_SKU_SELECTION_METHOD = str(RUN_CFG["smoke_sku_selection_method"])
BASELINE_METHOD = str(RUN_CFG["baseline_method"])
ENABLE_LIGHTGBM = bool(RUN_CFG["enable_lightgbm"])
ENABLE_RECURSIVE_INFERENCE = bool(RUN_CFG["enable_recursive_inference"])

EPSILON = float(METRIC_CFG["zero_scale_epsilon"])
METRIC_NAME = str(METRIC_CFG["metric_name"])
NEGATIVE_PROFIT_WEIGHT_POLICY = str(METRIC_CFG["negative_profit_weight_policy"])

TARGET_VARIANTS = [
    "net_daily_qty",
    "net_daily_qty_clip0",
    "gross_positive_qty",
]
DEFAULT_TARGET = str(RUN_CFG["target_variant"])

LAGS = [int(x) for x in FEATURE_CFG["lags"]]
ROLLING_WINDOWS = [int(x) for x in FEATURE_CFG["rolling_windows"]]
RECENT_WINDOWS = [int(x) for x in FEATURE_CFG["recent_windows"]]
CV_FOLDS: List[Dict[str, str]] = [dict(fold) for fold in VALIDATION_CFG["cv_folds"]]


@dataclass
class PipelinePaths:
    """Resolved project paths used by scripts and notebook orchestration."""

    cfg: Mapping[str, Any] = field(default_factory=lambda: DEFAULT_CONFIG)
    root: Path = ROOT

    @property
    def legacy_raw_dir(self) -> Path:
        return resolve_path(self.cfg["paths"]["legacy_raw_dir"], self.root)

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return resolve_path(Path(self.cfg["paths"]["raw_train_path"]).parent, self.root)

    @property
    def processed_dir(self) -> Path:
        return resolve_path(self.cfg["paths"]["processed_dir"], self.root)

    @property
    def feature_dir(self) -> Path:
        return resolve_path(self.cfg["paths"]["features_dir"], self.root)

    @property
    def model_dir(self) -> Path:
        return resolve_path(self.cfg["paths"]["models_dir"], self.root)

    @property
    def report_dir(self) -> Path:
        return resolve_path(self.cfg["paths"]["reports_dir"], self.root)

    @property
    def submission_dir(self) -> Path:
        return resolve_path(self.cfg["paths"]["submissions_dir"], self.root)

    @property
    def archive_dir(self) -> Path:
        return resolve_path(self.cfg["paths"]["archive_dir"], self.root)

    @property
    def before_ensemble_dir(self) -> Path:
        return self.submission_dir / "before_ensemble"

    @property
    def notebook_dir(self) -> Path:
        return resolve_path(self.cfg["paths"]["notebooks_dir"], self.root)

    @property
    def output_submission_path(self) -> Path:
        return resolve_path(
            self.cfg["submission"].get(
                "output_submission_path",
                self.cfg["paths"]["output_submission_path"],
            ),
            self.root,
        )

    @property
    def train_path(self) -> Path:
        preferred = resolve_path(self.cfg["paths"]["raw_train_path"], self.root)
        legacy = self.legacy_raw_dir / TRAIN_FILE
        return preferred if preferred.exists() else legacy

    @property
    def sample_submission_path(self) -> Path:
        preferred = resolve_path(self.cfg["paths"]["sample_submission_path"], self.root)
        legacy = self.legacy_raw_dir / SAMPLE_SUBMISSION_FILE
        return preferred if preferred.exists() else legacy


def mode_config(run_mode: str, cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, object]:
    """Return mode-specific options from config."""

    active = cfg or DEFAULT_CONFIG
    normalized = run_mode.lower().strip()
    if normalized not in {"smoke", "full"}:
        raise ValueError("RUN_MODE must be either 'smoke' or 'full'.")
    training = active["training"][normalized]
    run_control = active["run_control"]
    return {
        "run_mode": normalized,
        "n_skus": int(run_control["smoke_n_skus"]) if normalized == "smoke" else None,
        "top_profit_skus": int(run_control["smoke_top_profit_skus"])
        if normalized == "smoke"
        else None,
        "num_boost_round": int(training["num_boost_round"]),
        "early_stopping_rounds": int(training["early_stopping_rounds"]),
        "max_train_rows": training.get("max_train_rows"),
    }


def horizon_groups() -> List[Tuple[str, int, int]]:
    """Direct model horizon groups."""

    return [
        ("h1_7", 1, 7),
        ("h8_14", 8, 14),
        ("h15_28", 15, 28),
        ("h29_42", 29, 42),
        ("h43_56", 43, 56),
    ]

