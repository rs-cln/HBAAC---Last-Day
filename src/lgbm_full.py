"""Full-SKU LightGBM training, recursive inference, and CV utilities."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config
from .baselines import generate_all_baselines
from .metrics import (
    score_horizon_slices_from_matrices,
    wrmsse_score_from_matrices,
)
from .postprocess import ensemble_forecasts
from .utils import get_logger, write_metadata


LOGGER = get_logger(__name__)

MAC_SAFE_LGBM_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.04,
    "num_leaves": 63,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.01,
    "lambda_l2": 0.05,
    "max_bin": 255,
    "num_threads": 4,
    "force_col_wise": True,
    "deterministic": True,
    "seed": config.RANDOM_SEED,
    "feature_pre_filter": False,
    "verbosity": -1,
}

LAGS = [1, 2, 3, 7, 14, 21, 28, 35, 56, 84, 91, 112, 182, 364]
ROLLING_WINDOWS = [7, 14, 28, 56, 90, 180, 365]
CALENDAR_FEATURES = [
    "day_of_week",
    "week_of_year",
    "month",
    "day_of_month",
    "day_of_year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "quarter",
    "days_to_month_end",
]
STATIC_FEATURES = [
    "item_code_cat",
    "profit_weight_scaled",
    "profit_rank_log",
    "abc_code",
]
INTERMITTENT_FEATURES = [
    "days_since_last_sale",
    "recent_sales_intensity",
    "demand_burst_score",
]


@dataclass
class BoosterSpec:
    """Serializable metadata for a trained LightGBM-like predictor."""

    model_name: str
    objective: str
    feature_names: List[str]
    model_path: Path
    auxiliary_model_path: Optional[Path] = None
    best_iteration: Optional[int] = None


@dataclass
class FoldResult:
    """CV output for one model/fold."""

    fold: str
    model_name: str
    forecast: pd.DataFrame
    pred_matrix: np.ndarray
    scores: pd.DataFrame
    detail: pd.DataFrame
    diagnostics: Dict[str, object]


def require_lightgbm():
    """Import LightGBM with a clear error if libomp is unavailable."""

    try:
        import lightgbm as lgb
    except Exception as exc:
        raise ImportError(
            "LightGBM is unavailable. On macOS install libomp first, then retry."
        ) from exc
    return lgb


def feature_names() -> List[str]:
    """Return the fixed feature order used by the full runner."""

    names = list(STATIC_FEATURES) + list(CALENDAR_FEATURES)
    names.extend(f"lag_{lag}" for lag in LAGS)
    for window in ROLLING_WINDOWS:
        names.extend(
            [
                f"rolling_mean_{window}",
                f"rolling_sum_{window}",
                f"rolling_std_{window}",
                f"rolling_max_{window}",
                f"positive_rate_{window}",
            ]
        )
    names.extend(INTERMITTENT_FEATURES)
    return names


def load_panel_for_lgbm(
    panel_path: Path,
    target_col: str,
) -> pd.DataFrame:
    """Load only the columns needed for full LightGBM training."""

    panel = pd.read_parquet(panel_path, columns=["Date", "ItemCode", target_col])
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel["ItemCode"] = panel["ItemCode"].astype(str)
    panel[target_col] = panel[target_col].astype("float32").clip(lower=0)
    return panel


def make_wide_matrix(
    panel: pd.DataFrame,
    skus: Sequence[str],
    target_col: str,
) -> pd.DataFrame:
    """Create a dense date x SKU target matrix."""

    wide = (
        panel.pivot_table(
            index="Date",
            columns="ItemCode",
            values=target_col,
            aggfunc="sum",
        )
        .sort_index()
        .reindex(columns=list(skus), fill_value=0.0)
        .fillna(0.0)
    )
    full_index = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(full_index, fill_value=0.0)
    return wide.astype("float32")


def make_static_arrays(
    skus: Sequence[str],
    weight_table: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Build SKU-level features aligned to ``skus``."""

    weights = weight_table.set_index("ItemCode").reindex(list(skus))
    profit_weight = weights["profit_weight"].fillna(0.0).to_numpy(dtype=np.float32)
    profit_rank = weights["profit_rank"].fillna(len(skus) + 1).to_numpy(dtype=np.float32)
    abc = weights["abc_group"].fillna("ABC_C").map(
        {"ABC_A": 0.0, "ABC_B": 1.0, "ABC_C": 2.0}
    )
    return {
        "item_code_cat": np.arange(len(skus), dtype=np.float32),
        "profit_weight_scaled": (profit_weight * len(skus)).astype(np.float32),
        "profit_rank_log": np.log1p(profit_rank).astype(np.float32),
        "abc_code": abc.fillna(2.0).to_numpy(dtype=np.float32),
    }


def calendar_values(date: pd.Timestamp) -> Dict[str, float]:
    """Calendar features available before a forecast date."""

    days_in_month = date.days_in_month
    return {
        "day_of_week": float(date.dayofweek),
        "week_of_year": float(date.isocalendar().week),
        "month": float(date.month),
        "day_of_month": float(date.day),
        "day_of_year": float(date.dayofyear),
        "is_weekend": float(date.dayofweek >= 5),
        "is_month_start": float(date.is_month_start),
        "is_month_end": float(date.is_month_end),
        "quarter": float(date.quarter),
        "days_to_month_end": float(days_in_month - date.day),
    }


def _last_positive_before(values: np.ndarray, t_idx: int, lookback: int = 365) -> np.ndarray:
    """Return last positive row index before ``t_idx`` within a bounded lookback."""

    if t_idx <= 0:
        return np.full(values.shape[1], -1, dtype=np.int32)
    start = max(0, t_idx - lookback)
    positive = values[start:t_idx] > 0
    reverse_has_positive = positive[::-1].argmax(axis=0)
    has_positive = positive.any(axis=0)
    out = t_idx - 1 - reverse_has_positive.astype(np.int32)
    out[~has_positive] = -1
    return out.astype(np.int32)


def build_feature_block(
    values: np.ndarray,
    date_index: Sequence[pd.Timestamp],
    t_idx: int,
    static_arrays: Mapping[str, np.ndarray],
    names: Sequence[str],
) -> np.ndarray:
    """Build all SKU features for one date from history strictly before it."""

    n_skus = values.shape[1]
    block = np.empty((n_skus, len(names)), dtype=np.float32)
    date = pd.Timestamp(date_index[t_idx])
    col = 0

    for name in STATIC_FEATURES:
        block[:, col] = static_arrays[name]
        col += 1
    cal = calendar_values(date)
    for name in CALENDAR_FEATURES:
        block[:, col] = cal[name]
        col += 1
    for lag in LAGS:
        block[:, col] = values[t_idx - lag] if t_idx - lag >= 0 else 0.0
        col += 1
    rolling_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for window in ROLLING_WINDOWS:
        start = max(0, t_idx - window)
        segment = values[start:t_idx]
        if len(segment) == 0:
            mean = np.zeros(n_skus, dtype=np.float32)
            total = np.zeros(n_skus, dtype=np.float32)
            std = np.zeros(n_skus, dtype=np.float32)
            max_value = np.zeros(n_skus, dtype=np.float32)
            positive_rate = np.zeros(n_skus, dtype=np.float32)
        else:
            total = segment.sum(axis=0, dtype=np.float32)
            mean = total / np.float32(len(segment))
            std = segment.std(axis=0, dtype=np.float32)
            max_value = segment.max(axis=0)
            positive_rate = (segment > 0).mean(axis=0, dtype=np.float32)
        rolling_cache[window] = (mean, total, std, max_value)
        block[:, col] = mean
        block[:, col + 1] = total
        block[:, col + 2] = std
        block[:, col + 3] = max_value
        block[:, col + 4] = positive_rate
        col += 5
    last_positive = _last_positive_before(values, t_idx)
    days_since = np.where(last_positive >= 0, t_idx - last_positive, 9999)
    block[:, col] = days_since.astype(np.float32)
    col += 1
    sum_28 = rolling_cache[28][1]
    sum_180 = rolling_cache[180][1]
    max_90 = rolling_cache[90][3]
    mean_90 = rolling_cache[90][0]
    block[:, col] = sum_28 / (sum_180 + config.EPSILON)
    col += 1
    block[:, col] = max_90 / (mean_90 + config.EPSILON)

    if not np.isfinite(block).all():
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
    return block


def build_matrix_for_dates(
    wide: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    static_arrays: Mapping[str, np.ndarray],
    names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature and target matrices for selected dates."""

    date_index = list(pd.to_datetime(wide.index))
    position = {date: idx for idx, date in enumerate(date_index)}
    values = wide.to_numpy(dtype=np.float32, copy=False)
    dates = [pd.Timestamp(date) for date in dates]
    n_skus = values.shape[1]
    x = np.empty((len(dates) * n_skus, len(names)), dtype=np.float32)
    y = np.empty(len(dates) * n_skus, dtype=np.float32)
    for i, date in enumerate(dates):
        t_idx = position[date]
        start = i * n_skus
        end = start + n_skus
        x[start:end] = build_feature_block(
            values=values,
            date_index=date_index,
            t_idx=t_idx,
            static_arrays=static_arrays,
            names=names,
        )
        y[start:end] = values[t_idx]
    return x, y


def date_range_for_training(
    train_end: pd.Timestamp,
    train_window_days: int,
    internal_valid_days: int = 56,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Return recent train and internal validation date ranges."""

    train_end = pd.Timestamp(train_end)
    valid_start = train_end - pd.Timedelta(days=internal_valid_days - 1)
    train_start = train_end - pd.Timedelta(days=train_window_days - 1)
    train_dates = pd.date_range(train_start, valid_start - pd.Timedelta(days=1), freq="D")
    valid_dates = pd.date_range(valid_start, train_end, freq="D")
    return train_dates, valid_dates


def make_row_weights(
    dates: Sequence[pd.Timestamp],
    skus: Sequence[str],
    weight_table: pd.DataFrame,
    reference_date: pd.Timestamp,
    half_life_days: float = 365.0,
) -> np.ndarray:
    """Profit-weighted and recency-weighted row weights."""

    weights = weight_table.set_index("ItemCode").reindex(list(skus))
    sku_weight = weights["profit_weight"].fillna(0.0).to_numpy(dtype=np.float32)
    sku_weight = np.float32(0.05) + sku_weight * np.float32(len(skus))
    age_days = np.array(
        [(pd.Timestamp(reference_date) - pd.Timestamp(date)).days for date in dates],
        dtype=np.float32,
    )
    recency = np.power(np.float32(0.5), age_days / np.float32(half_life_days))
    row_weights = np.repeat(recency, len(skus)) * np.tile(sku_weight, len(dates))
    mean = float(row_weights.mean())
    if mean <= 0 or not np.isfinite(mean):
        return np.ones(len(row_weights), dtype=np.float32)
    return (row_weights / mean).astype(np.float32)


def lgbm_params(objective: str, seed: int, tweedie_power: float = 1.3) -> Dict[str, Any]:
    """Objective-specific LightGBM parameters."""

    params = dict(MAC_SAFE_LGBM_PARAMS)
    params["seed"] = int(seed)
    params["bagging_seed"] = int(seed)
    params["feature_fraction_seed"] = int(seed)
    if objective == "regression_log1p":
        params["objective"] = "regression"
        params["metric"] = "rmse"
    elif objective == "tweedie":
        params["objective"] = "tweedie"
        params["metric"] = "tweedie"
        params["tweedie_variance_power"] = tweedie_power
    elif objective == "poisson":
        params["objective"] = "poisson"
        params["metric"] = "poisson"
    elif objective == "binary":
        params["objective"] = "binary"
        params["metric"] = "binary_logloss"
    else:
        params["objective"] = objective
        params["metric"] = "rmse"
    return params


def train_booster(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    w_valid: np.ndarray,
    names: Sequence[str],
    objective: str,
    num_boost_round: int,
    early_stopping_rounds: int,
    seed: int,
    tweedie_power: float = 1.3,
) -> Any:
    """Train one LightGBM booster."""

    lgb = require_lightgbm()
    y_train_model = y_train
    y_valid_model = y_valid
    if objective == "regression_log1p":
        y_train_model = np.log1p(y_train)
        y_valid_model = np.log1p(y_valid)
    params = lgbm_params(objective, seed=seed, tweedie_power=tweedie_power)
    train_data = lgb.Dataset(
        x_train,
        label=y_train_model,
        weight=w_train,
        feature_name=list(names),
        free_raw_data=True,
    )
    valid_data = lgb.Dataset(
        x_valid,
        label=y_valid_model,
        weight=w_valid,
        feature_name=list(names),
        reference=train_data,
        free_raw_data=True,
    )
    return lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[valid_data],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )


def train_hurdle_boosters(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    w_valid: np.ndarray,
    names: Sequence[str],
    num_boost_round: int,
    early_stopping_rounds: int,
    seed: int,
) -> Tuple[Any, Any]:
    """Train binary occurrence and positive-quantity boosters."""

    classifier = train_booster(
        x_train=x_train,
        y_train=(y_train > 0).astype(np.float32),
        w_train=w_train,
        x_valid=x_valid,
        y_valid=(y_valid > 0).astype(np.float32),
        w_valid=w_valid,
        names=names,
        objective="binary",
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
    )
    positive_train = y_train > 0
    positive_valid = y_valid > 0
    if positive_train.sum() == 0 or positive_valid.sum() == 0:
        raise ValueError("Hurdle regressor needs positive train and validation rows.")
    regressor = train_booster(
        x_train=x_train[positive_train],
        y_train=y_train[positive_train],
        w_train=w_train[positive_train],
        x_valid=x_valid[positive_valid],
        y_valid=y_valid[positive_valid],
        w_valid=w_valid[positive_valid],
        names=names,
        objective="regression_log1p",
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed + 17,
    )
    return classifier, regressor


def predict_booster_matrix(
    booster: Any,
    x: np.ndarray,
    objective: str,
) -> np.ndarray:
    """Predict non-negative quantities from a booster."""

    pred = np.asarray(
        booster.predict(x, num_iteration=booster.best_iteration),
        dtype=np.float32,
    )
    if objective == "regression_log1p":
        pred = np.expm1(pred)
    return np.clip(pred, 0.0, None).astype(np.float32)


def predict_hurdle_matrix(classifier: Any, regressor: Any, x: np.ndarray) -> np.ndarray:
    """Predict hurdle expected demand."""

    prob = np.asarray(
        classifier.predict(x, num_iteration=classifier.best_iteration),
        dtype=np.float32,
    )
    qty = np.expm1(
        np.asarray(
            regressor.predict(x, num_iteration=regressor.best_iteration),
            dtype=np.float32,
        )
    )
    return np.clip(prob * qty, 0.0, None).astype(np.float32)


def recursive_predict_matrix(
    model: Any,
    wide_history: pd.DataFrame,
    static_arrays: Mapping[str, np.ndarray],
    names: Sequence[str],
    start_date: pd.Timestamp,
    horizon: int,
    objective: str,
    auxiliary_model: Optional[Any] = None,
) -> np.ndarray:
    """Recursive h1..horizon prediction matrix with previous predictions as lags."""

    values = wide_history.to_numpy(dtype=np.float32, copy=True)
    date_index = list(pd.to_datetime(wide_history.index))
    preds = np.empty((horizon, values.shape[1]), dtype=np.float32)
    for h in range(1, horizon + 1):
        current_date = pd.Timestamp(start_date) + pd.Timedelta(days=h - 1)
        date_index.append(current_date)
        padded = np.vstack([values, np.zeros((1, values.shape[1]), dtype=np.float32)])
        x = build_feature_block(
            values=padded,
            date_index=date_index,
            t_idx=len(date_index) - 1,
            static_arrays=static_arrays,
            names=names,
        )
        if objective == "hurdle":
            if auxiliary_model is None:
                raise ValueError("Hurdle inference requires an auxiliary regressor.")
            pred = predict_hurdle_matrix(model, auxiliary_model, x)
        else:
            pred = predict_booster_matrix(model, x, objective)
        preds[h - 1] = pred
        values = np.vstack([values, pred.reshape(1, -1).astype(np.float32)])
        if not np.isfinite(pred).all() or (pred < 0).any():
            raise AssertionError("Recursive LightGBM produced invalid predictions.")
    return preds


def zero_sundays(pred_matrix: np.ndarray, start_date: pd.Timestamp) -> np.ndarray:
    """Set Sunday horizons to zero for distributor closure pattern post-processing."""

    out = pred_matrix.copy()
    dates = pd.date_range(pd.Timestamp(start_date), periods=len(out), freq="D")
    sunday = np.array([date.dayofweek == 6 for date in dates], dtype=bool)
    out[sunday] = 0.0
    return out


def forecast_matrix_to_long(
    pred_matrix: np.ndarray,
    skus: Sequence[str],
    start_date: pd.Timestamp,
    model_name: str,
) -> pd.DataFrame:
    """Convert horizon x SKU predictions to long forecast format."""

    horizon = pred_matrix.shape[0]
    dates = pd.date_range(start_date, periods=horizon, freq="D")
    out = pd.DataFrame(
        {
            "ItemCode": np.tile(np.asarray(skus, dtype=object), horizon),
            "Date": np.repeat(dates.to_numpy(), len(skus)),
            "horizon": np.repeat(np.arange(1, horizon + 1), len(skus)),
            "y_pred": pred_matrix.reshape(-1).astype(float),
        }
    )
    out["model_name"] = model_name
    return out[["model_name", "ItemCode", "Date", "horizon", "y_pred"]]


def actual_matrix_for_dates(
    wide: pd.DataFrame,
    start_date: pd.Timestamp,
    horizon: int,
) -> np.ndarray:
    """Return actual target matrix for a contiguous horizon."""

    dates = pd.date_range(start_date, periods=horizon, freq="D")
    missing = [str(date.date()) for date in dates if date not in wide.index]
    if missing:
        raise AssertionError(f"Actual matrix is missing dates: {missing[:5]}")
    return wide.loc[dates].to_numpy(dtype=np.float32, copy=False)


def forecast_to_scoring_frame(
    forecast: pd.DataFrame,
    wide_actual: pd.DataFrame,
    start_date: pd.Timestamp,
    horizon: int,
    target_col: str,
) -> pd.DataFrame:
    """Attach actuals to a long forecast for saved OOF predictions."""

    actual = actual_matrix_for_dates(wide_actual, start_date, horizon)
    skus = list(wide_actual.columns)
    actual_long = pd.DataFrame(
        {
            "ItemCode": np.tile(np.asarray(skus, dtype=object), horizon),
            "Date": np.repeat(
                pd.date_range(start_date, periods=horizon, freq="D").to_numpy(),
                len(skus),
            ),
            "horizon": np.repeat(np.arange(1, horizon + 1), len(skus)),
            target_col: actual.reshape(-1).astype(float),
        }
    )
    return actual_long.merge(
        forecast.drop(columns=["model_name"], errors="ignore"),
        on=["ItemCode", "Date", "horizon"],
        how="left",
    )


def score_prediction_matrix(
    pred_matrix: np.ndarray,
    wide_actual: pd.DataFrame,
    wide_history: pd.DataFrame,
    valid_start: pd.Timestamp,
    weight_table: pd.DataFrame,
    model_name: str,
    fold_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Score one fold with official profit-weighted WRMSSE."""

    actual = actual_matrix_for_dates(wide_actual, valid_start, pred_matrix.shape[0])
    history = wide_history.to_numpy(dtype=np.float32, copy=False)
    score, diag, detail = wrmsse_score_from_matrices(
        history_values=history,
        actual_values=actual,
        pred_values=pred_matrix,
        item_codes=list(wide_actual.columns),
        weight_table=weight_table,
    )
    horizon_scores = score_horizon_slices_from_matrices(
        history_values=history,
        actual_values=actual,
        pred_values=pred_matrix,
        item_codes=list(wide_actual.columns),
        weight_table=weight_table,
    )
    rows = []
    for _, row in horizon_scores.iterrows():
        rows.append(
            {
                "fold": fold_name,
                "model_name": model_name,
                "slice": row["slice"],
                "wrmsse": row["wrmsse"],
                "zero_scale_skus": row["zero_scale_skus"],
                "scored_skus": row["scored_skus"],
                "weight_sum_scored": row["weight_sum_scored"],
            }
        )
    rows.append(
        {
            "fold": fold_name,
            "model_name": model_name,
            "slice": "official_aggregate",
            "wrmsse": score,
            "zero_scale_skus": diag["zero_scale_skus"],
            "scored_skus": diag["scored_skus"],
            "weight_sum_scored": diag["weight_sum_scored"],
        }
    )
    detail["fold"] = fold_name
    detail["model_name"] = model_name
    return pd.DataFrame(rows), detail, diag


def train_recursive_for_fold(
    wide: pd.DataFrame,
    skus: Sequence[str],
    weight_table: pd.DataFrame,
    fold: Mapping[str, str],
    objective: str,
    train_window_days: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    seed: int,
    model_dir: Path,
    tweedie_power: float = 1.3,
) -> Tuple[Any, Optional[Any], BoosterSpec]:
    """Train one recursive objective for one fold."""

    names = feature_names()
    static_arrays = make_static_arrays(skus, weight_table)
    train_end = pd.Timestamp(fold["train_end"])
    train_dates, valid_dates = date_range_for_training(
        train_end,
        train_window_days=train_window_days,
    )
    LOGGER.info(
        "Building full LGBM matrices for %s %s: %s train days, %s valid days",
        fold["fold"],
        objective,
        len(train_dates),
        len(valid_dates),
    )
    x_train, y_train = build_matrix_for_dates(wide, train_dates, static_arrays, names)
    x_valid, y_valid = build_matrix_for_dates(wide, valid_dates, static_arrays, names)
    w_train = make_row_weights(train_dates, skus, weight_table, train_end)
    w_valid = make_row_weights(valid_dates, skus, weight_table, train_end)
    LOGGER.info(
        "Training %s on %.2fM rows, validating on %.2fM rows",
        objective,
        len(y_train) / 1_000_000,
        len(y_valid) / 1_000_000,
    )
    start = time.perf_counter()
    auxiliary = None
    if objective == "hurdle":
        model, auxiliary = train_hurdle_boosters(
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
            x_valid=x_valid,
            y_valid=y_valid,
            w_valid=w_valid,
            names=names,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
        )
    else:
        model = train_booster(
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
            x_valid=x_valid,
            y_valid=y_valid,
            w_valid=w_valid,
            names=names,
            objective=objective,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
            tweedie_power=tweedie_power,
        )
    LOGGER.info("Finished %s training in %.1fs", objective, time.perf_counter() - start)
    del x_train, y_train, x_valid, y_valid, w_train, w_valid
    gc.collect()

    model_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{objective}_{fold['fold']}"
    model_path = model_dir / f"recursive_lgbm_{suffix}.txt"
    auxiliary_path = None
    model.save_model(str(model_path))
    if auxiliary is not None:
        auxiliary_path = model_dir / f"recursive_lgbm_{suffix}_regressor.txt"
        auxiliary.save_model(str(auxiliary_path))
    spec = BoosterSpec(
        model_name=f"recursive_lgbm_{objective}",
        objective=objective,
        feature_names=list(names),
        model_path=model_path,
        auxiliary_model_path=auxiliary_path,
        best_iteration=getattr(model, "best_iteration", None),
    )
    return model, auxiliary, spec


def evaluate_fold_model(
    model: Any,
    auxiliary: Optional[Any],
    spec: BoosterSpec,
    wide: pd.DataFrame,
    skus: Sequence[str],
    weight_table: pd.DataFrame,
    fold: Mapping[str, str],
) -> FoldResult:
    """Recursive forecast and official WRMSSE scoring for one fold."""

    static_arrays = make_static_arrays(skus, weight_table)
    train_end = pd.Timestamp(fold["train_end"])
    valid_start = pd.Timestamp(fold["valid_start"])
    history_wide = wide.loc[wide.index <= train_end]
    pred_matrix = recursive_predict_matrix(
        model=model,
        wide_history=history_wide,
        static_arrays=static_arrays,
        names=spec.feature_names,
        start_date=valid_start,
        horizon=config.FORECAST_HORIZON,
        objective=spec.objective,
        auxiliary_model=auxiliary,
    )
    pred_matrix = zero_sundays(pred_matrix, valid_start)
    scores, detail, diag = score_prediction_matrix(
        pred_matrix=pred_matrix,
        wide_actual=wide,
        wide_history=history_wide,
        valid_start=valid_start,
        weight_table=weight_table,
        model_name=spec.model_name,
        fold_name=str(fold["fold"]),
    )
    forecast = forecast_matrix_to_long(
        pred_matrix,
        skus,
        start_date=valid_start,
        model_name=spec.model_name,
    )
    return FoldResult(
        fold=str(fold["fold"]),
        model_name=spec.model_name,
        forecast=forecast,
        pred_matrix=pred_matrix,
        scores=scores,
        detail=detail,
        diagnostics=diag,
    )


def baseline_cv_forecast(
    wide: pd.DataFrame,
    skus: Sequence[str],
    fold: Mapping[str, str],
    target_col: str,
    baseline_method: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate the configured simple baseline for one CV fold."""

    train_end = pd.Timestamp(fold["train_end"])
    valid_start = pd.Timestamp(fold["valid_start"])
    if baseline_method == "simple_pipeline_sunday_zero":
        history_values = wide.loc[wide.index <= train_end].tail(56).to_numpy(
            dtype=np.float32,
            copy=False,
        )
        means = history_values.mean(axis=0) if len(history_values) else np.zeros(len(skus))
        matrix = np.tile(means.astype(np.float32), (config.FORECAST_HORIZON, 1))
        dates = pd.date_range(valid_start, periods=config.FORECAST_HORIZON, freq="D")
        sunday = np.array([date.dayofweek == 6 for date in dates], dtype=bool)
        matrix[sunday] = 0.0
        forecast = forecast_matrix_to_long(
            matrix,
            skus,
            start_date=valid_start,
            model_name=baseline_method,
        )
        return forecast, matrix

    history = wide.loc[wide.index <= train_end].reset_index(names="Date").melt(
        id_vars="Date",
        var_name="ItemCode",
        value_name=target_col,
    )
    baselines = generate_all_baselines(
        history=history,
        skus=list(skus),
        forecast_start=valid_start,
        target_col=target_col,
        horizon=config.FORECAST_HORIZON,
    )
    if baseline_method not in baselines:
        raise KeyError(f"Missing baseline method: {baseline_method}")
    forecast = baselines[baseline_method]
    matrix = (
        forecast.pivot(index="horizon", columns="ItemCode", values="y_pred")
        .reindex(index=range(1, config.FORECAST_HORIZON + 1), columns=list(skus))
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    return forecast, matrix


def score_baseline_fold(
    wide: pd.DataFrame,
    skus: Sequence[str],
    weight_table: pd.DataFrame,
    fold: Mapping[str, str],
    target_col: str,
    baseline_method: str,
) -> FoldResult:
    """Score the configured full baseline on one fold."""

    forecast, matrix = baseline_cv_forecast(
        wide=wide,
        skus=skus,
        fold=fold,
        target_col=target_col,
        baseline_method=baseline_method,
    )
    history_wide = wide.loc[wide.index <= pd.Timestamp(fold["train_end"])]
    scores, detail, diag = score_prediction_matrix(
        pred_matrix=matrix,
        wide_actual=wide,
        wide_history=history_wide,
        valid_start=pd.Timestamp(fold["valid_start"]),
        weight_table=weight_table,
        model_name=baseline_method,
        fold_name=str(fold["fold"]),
    )
    return FoldResult(
        fold=str(fold["fold"]),
        model_name=baseline_method,
        forecast=forecast,
        pred_matrix=matrix,
        scores=scores,
        detail=detail,
        diagnostics=diag,
    )


def tune_two_way_blend(
    fold_results: Sequence[FoldResult],
    weight_table: pd.DataFrame,
    wide: pd.DataFrame,
    folds: Sequence[Mapping[str, str]],
    candidate_names: Sequence[str],
    optimize_slice: str = "h29_56",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Grid-search non-negative ensemble weights on available fold forecasts."""

    result_lookup = {(res.fold, res.model_name): res for res in fold_results}
    rows = []
    best_score = np.inf
    best_weights: Dict[str, float] = {}
    grids: List[Dict[str, float]] = []
    if len(candidate_names) == 1:
        grids = [{candidate_names[0]: 1.0}]
    elif len(candidate_names) == 2:
        for w0 in np.linspace(0, 1, 21):
            grids.append(
                {
                    candidate_names[0]: float(w0),
                    candidate_names[1]: float(1.0 - w0),
                }
            )
    else:
        rng = np.random.default_rng(config.RANDOM_SEED)
        for _ in range(80):
            draw = rng.dirichlet(np.ones(len(candidate_names)))
            grids.append(dict(zip(candidate_names, draw.astype(float))))
        for name in candidate_names:
            grids.append({candidate: float(candidate == name) for candidate in candidate_names})

    for trial, weights in enumerate(grids):
        fold_scores = []
        for fold in folds:
            matrices = []
            model_weights = []
            for name in candidate_names:
                result = result_lookup.get((str(fold["fold"]), name))
                if result is None:
                    continue
                model_weights.append(float(weights.get(name, 0.0)))
                matrices.append(result.pred_matrix)
            total_weight = float(sum(model_weights))
            if total_weight <= 0:
                continue
            pred = sum(
                matrix * np.float32(weight / total_weight)
                for matrix, weight in zip(matrices, model_weights)
            )
            history_wide = wide.loc[wide.index <= pd.Timestamp(fold["train_end"])]
            actual = actual_matrix_for_dates(
                wide,
                pd.Timestamp(fold["valid_start"]),
                config.FORECAST_HORIZON,
            )
            if optimize_slice == "h1_28":
                score_slice = slice(0, 28)
            elif optimize_slice == "h29_56":
                score_slice = slice(28, 56)
            elif optimize_slice == "h1_56":
                score_slice = slice(None)
            else:
                raise ValueError("optimize_slice must be h1_28, h29_56, or h1_56.")
            score, _, _ = wrmsse_score_from_matrices(
                history_values=history_wide.to_numpy(dtype=np.float32, copy=False),
                actual_values=actual[score_slice],
                pred_values=pred[score_slice],
                item_codes=list(wide.columns),
                weight_table=weight_table,
            )
            fold_scores.append(score)
        if not fold_scores:
            continue
        avg_score = float(np.mean(fold_scores))
        row = {"trial": trial, "avg_wrmsse": avg_score}
        row.update({f"weight_{name}": float(weights.get(name, 0.0)) for name in candidate_names})
        rows.append(row)
        if avg_score < best_score:
            best_score = avg_score
            best_weights = dict(weights)
    return best_weights, pd.DataFrame(rows).sort_values("avg_wrmsse")


def write_feature_importance(
    model: Any,
    feature_names_: Sequence[str],
    model_name: str,
    path: Path,
) -> pd.DataFrame:
    """Save LightGBM gain/split feature importance."""

    importance = pd.DataFrame(
        {
            "model_name": model_name,
            "feature": list(feature_names_),
            "importance_gain": model.feature_importance(importance_type="gain"),
            "importance_split": model.feature_importance(importance_type="split"),
        }
    ).sort_values(["model_name", "importance_gain"], ascending=[True, False])
    path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(path, index=False)
    return importance


def save_forecast_artifact(
    forecast: pd.DataFrame,
    path: Path,
    run_mode: str,
    model_name: str,
) -> None:
    """Save a long forecast with metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(path, index=False)
    write_metadata(path, run_mode, {"model_name": model_name, "rows": len(forecast)})


def make_ensemble_forecast(
    forecasts: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Weighted ensemble and h1..h56 safety assertions."""

    merged = ensemble_forecasts(dict(forecasts), dict(weights))
    got_horizons = set(merged["horizon"].astype(int).unique())
    expected_horizons = set(range(1, config.FORECAST_HORIZON + 1))
    if got_horizons != expected_horizons:
        raise AssertionError(f"Ensemble horizons {sorted(got_horizons)} != 1..56.")
    if (merged["y_pred"] < 0).any() or not np.isfinite(merged["y_pred"]).all():
        raise AssertionError("Ensemble forecast contains invalid predictions.")
    return merged
