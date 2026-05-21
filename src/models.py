"""Model training helpers for recursive, direct, and hurdle LightGBM models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import model_feature_columns
from .utils import get_logger, write_json

LOGGER = get_logger(__name__)


def _require_lightgbm():
    try:
        import lightgbm as lgb
    except (ImportError, OSError) as exc:
        raise ImportError(
            "LightGBM is required for model training. Install requirements.txt and "
            "ensure the OpenMP runtime is available. On macOS this usually means "
            "installing libomp."
        ) from exc
    return lgb


def recency_weights(dates: pd.Series, half_life_days: float = 365.0) -> np.ndarray:
    """Exponential recency weights where recent observations get more weight."""

    max_date = pd.to_datetime(dates).max()
    age_days = (max_date - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
    return np.power(0.5, age_days / half_life_days)


def make_sample_weights(
    frame: pd.DataFrame,
    weight_col: str = "profit_weight",
    date_col: str = "Date",
    recency_half_life_days: float = 365.0,
) -> np.ndarray:
    """Combine profit and recency weights for training rows."""

    profit = frame[weight_col].fillna(0.0).to_numpy(dtype=float) if weight_col in frame else 1.0
    if np.ndim(profit) == 0:
        profit = np.ones(len(frame), dtype=float)
    recency = recency_weights(frame[date_col], half_life_days=recency_half_life_days)
    weights = profit * recency
    mean = weights.mean()
    if mean <= 0 or not np.isfinite(mean):
        return np.ones(len(frame), dtype=float)
    return weights / mean


def default_lgbm_params(objective: str = "tweedie", seed: int = config.RANDOM_SEED) -> Dict[str, object]:
    """Default LightGBM parameters with objective-specific settings."""

    params: Dict[str, object] = {
        "objective": objective,
        "metric": "rmse" if objective == "regression" else objective,
        "learning_rate": 0.035,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.01,
        "lambda_l2": 0.05,
        "max_bin": 255,
        "seed": seed,
        "verbosity": -1,
        "num_threads": -1,
    }
    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.3
    if objective == "poisson":
        params["metric"] = "poisson"
    return params


@dataclass
class TrainedModel:
    """Lightweight model artifact metadata."""

    model_name: str
    target_col: str
    feature_cols: List[str]
    categorical_cols: List[str]
    objective: str
    model_path: Path
    metadata_path: Path


def _prepare_xy(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    objective: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    x = frame[list(feature_cols)].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(0.0)
    y = frame[target_col].astype(float).clip(lower=0)
    if objective == "regression_log1p":
        y = np.log1p(y)
    return x, y


def train_recursive_lgbm(
    feature_frame: pd.DataFrame,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
    target_col: str = config.DEFAULT_TARGET,
    objective: str = "tweedie",
    params: Optional[Dict[str, object]] = None,
    model_name: Optional[str] = None,
    model_dir: Path = config.MODEL_DIR / "recursive",
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 100,
    seed: int = config.RANDOM_SEED,
) -> Tuple[object, TrainedModel, pd.DataFrame]:
    """Train one global recursive LightGBM model."""

    lgb = _require_lightgbm()
    work = feature_frame.sort_values(["ItemCode", "Date"]).copy()
    train_mask = work["Date"] <= train_end
    valid_mask = (work["Date"] > train_end) & (work["Date"] <= valid_end)
    train_df = work.loc[train_mask].copy()
    valid_df = work.loc[valid_mask].copy()
    if valid_df.empty:
        raise ValueError("Validation frame is empty for recursive model training.")

    feature_cols = model_feature_columns(work, target_col)
    categorical_cols = [c for c in ["ItemCode_cat", "Demand_Type_cat", "abc_group_cat"] if c in feature_cols]
    lgb_objective = "regression" if objective == "regression_log1p" else objective
    model_params = default_lgbm_params(lgb_objective, seed=seed)
    if params:
        model_params.update(params)

    x_train, y_train = _prepare_xy(train_df, feature_cols, target_col, objective)
    x_valid, y_valid = _prepare_xy(valid_df, feature_cols, target_col, objective)
    w_train = make_sample_weights(train_df)
    w_valid = make_sample_weights(valid_df)

    train_data = lgb.Dataset(
        x_train,
        label=y_train,
        weight=w_train,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )
    valid_data = lgb.Dataset(
        x_valid,
        label=y_valid,
        weight=w_valid,
        categorical_feature=categorical_cols,
        reference=train_data,
        free_raw_data=False,
    )
    evals_result: Dict[str, Dict[str, List[float]]] = {}
    booster = lgb.train(
        model_params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[valid_data],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.record_evaluation(evals_result),
        ],
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    name = model_name or f"recursive_lgbm_{objective}"
    model_path = model_dir / f"{name}.txt"
    booster.save_model(str(model_path))
    metadata_path = model_dir / f"{name}.json"
    metadata = {
        "model_name": name,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "objective": objective,
        "params": model_params,
        "best_iteration": booster.best_iteration,
        "train_end": str(train_end.date()),
        "valid_end": str(valid_end.date()),
    }
    write_json(metadata_path, metadata)
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": booster.feature_importance(importance_type="gain"),
            "importance_split": booster.feature_importance(importance_type="split"),
            "model_name": name,
        }
    ).sort_values("importance_gain", ascending=False)
    artifact = TrainedModel(
        model_name=name,
        target_col=target_col,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        objective=objective,
        model_path=model_path,
        metadata_path=metadata_path,
    )
    return booster, artifact, importance


def predict_booster(model: object, x: pd.DataFrame, objective: str, num_iteration: Optional[int] = None) -> np.ndarray:
    """Predict and invert log transform when needed."""

    pred = np.asarray(model.predict(x, num_iteration=num_iteration), dtype=float)
    if objective == "regression_log1p":
        pred = np.expm1(pred)
    return np.clip(pred, 0.0, None)


def make_direct_training_frame(
    feature_frame: pd.DataFrame,
    target_col: str,
    min_horizon: int,
    max_horizon: int,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Build direct-horizon training rows with horizon as a feature."""

    work = feature_frame.sort_values(["ItemCode", "Date"]).copy()
    frames = []
    for horizon in range(min_horizon, max_horizon + 1):
        shifted = work.copy()
        shifted["direct_target"] = (
            shifted.groupby("ItemCode", observed=True)[target_col].shift(-horizon)
        )
        shifted["horizon"] = horizon
        frames.append(shifted.dropna(subset=["direct_target"]))
    out = pd.concat(frames, ignore_index=True)
    if max_rows is not None and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=config.RANDOM_SEED).sort_values(["ItemCode", "Date"])
    return out


def train_direct_lgbm(
    feature_frame: pd.DataFrame,
    train_end: pd.Timestamp,
    target_col: str,
    horizon_name: str,
    min_horizon: int,
    max_horizon: int,
    model_dir: Path = config.MODEL_DIR / "direct",
    num_boost_round: int = 800,
    early_stopping_rounds: int = 80,
    max_rows: Optional[int] = None,
) -> Tuple[object, TrainedModel, pd.DataFrame]:
    """Train one direct model for a horizon group."""

    direct = make_direct_training_frame(
        feature_frame,
        target_col=target_col,
        min_horizon=min_horizon,
        max_horizon=max_horizon,
        max_rows=max_rows,
    )
    direct["target_for_model"] = direct["direct_target"].clip(lower=0)
    valid_start = train_end - pd.Timedelta(days=56)
    train = direct.loc[direct["Date"] <= valid_start].copy()
    valid = direct.loc[(direct["Date"] > valid_start) & (direct["Date"] <= train_end)].copy()
    if train.empty or valid.empty:
        raise ValueError(f"Not enough data for direct model {horizon_name}.")

    tmp = pd.concat([train, valid], ignore_index=True)
    feature_cols = model_feature_columns(tmp, "target_for_model")
    if "horizon" not in feature_cols:
        feature_cols.append("horizon")
    lgb = _require_lightgbm()
    x_train, y_train = _prepare_xy(train, feature_cols, "target_for_model", "tweedie")
    x_valid, y_valid = _prepare_xy(valid, feature_cols, "target_for_model", "tweedie")
    params = default_lgbm_params("tweedie")
    params["tweedie_variance_power"] = 1.3
    train_data = lgb.Dataset(x_train, label=y_train, weight=make_sample_weights(train), free_raw_data=False)
    valid_data = lgb.Dataset(x_valid, label=y_valid, weight=make_sample_weights(valid), reference=train_data, free_raw_data=False)
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[valid_data],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    name = f"direct_lgbm_{horizon_name}"
    model_path = model_dir / f"{name}.txt"
    booster.save_model(str(model_path))
    metadata_path = model_dir / f"{name}.json"
    write_json(
        metadata_path,
        {
            "model_name": name,
            "target_col": target_col,
            "feature_cols": feature_cols,
            "objective": "tweedie",
            "min_horizon": min_horizon,
            "max_horizon": max_horizon,
            "best_iteration": booster.best_iteration,
        },
    )
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": booster.feature_importance(importance_type="gain"),
            "importance_split": booster.feature_importance(importance_type="split"),
            "model_name": name,
        }
    ).sort_values("importance_gain", ascending=False)
    artifact = TrainedModel(
        model_name=name,
        target_col=target_col,
        feature_cols=feature_cols,
        categorical_cols=[],
        objective="tweedie",
        model_path=model_path,
        metadata_path=metadata_path,
    )
    return booster, artifact, importance


@dataclass
class HurdleModel:
    """Classifier plus positive-demand regressor."""

    classifier: object
    regressor: object
    feature_cols: List[str]

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        proba = self.classifier.predict_proba(x[self.feature_cols])[:, 1]
        qty = np.expm1(self.regressor.predict(x[self.feature_cols]))
        return np.clip(proba * qty, 0.0, None)


def train_hurdle_lgbm(
    feature_frame: pd.DataFrame,
    train_end: pd.Timestamp,
    target_col: str = config.DEFAULT_TARGET,
    model_dir: Path = config.MODEL_DIR / "hurdle",
    max_rows: Optional[int] = None,
) -> Tuple[HurdleModel, Path]:
    """Train a simple LightGBM hurdle model."""

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except (ImportError, OSError) as exc:
        raise ImportError(
            "LightGBM is required for hurdle training and needs the OpenMP runtime."
        ) from exc

    train = feature_frame.loc[feature_frame["Date"] <= train_end].copy()
    if max_rows is not None and len(train) > max_rows:
        train = train.sample(n=max_rows, random_state=config.RANDOM_SEED)
    feature_cols = model_feature_columns(train, target_col)
    x = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train[target_col].astype(float).clip(lower=0)
    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=63,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(x, (y > 0).astype(int), sample_weight=make_sample_weights(train))
    positive = y > 0
    reg = LGBMRegressor(
        objective="regression",
        n_estimators=600,
        learning_rate=0.035,
        num_leaves=63,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    reg.fit(
        x.loc[positive],
        np.log1p(y.loc[positive]),
        sample_weight=make_sample_weights(train.loc[positive]),
    )
    model = HurdleModel(classifier=clf, regressor=reg, feature_cols=feature_cols)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "hurdle_lgbm.joblib"
    joblib.dump(model, path)
    return model, path
