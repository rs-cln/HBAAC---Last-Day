"""Baseline forecasters and CV scoring."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config
from .metrics import score_horizon_slices, wrmsse_score


def make_forecast_dates(start_date: pd.Timestamp, horizon: int = config.FORECAST_HORIZON) -> pd.DataFrame:
    """Create future dates with 1-based horizon."""

    dates = pd.date_range(start=start_date, periods=horizon, freq="D")
    return pd.DataFrame({"Date": dates, "horizon": np.arange(1, horizon + 1)})


def _history_matrix(
    history: pd.DataFrame,
    item_col: str,
    date_col: str,
    target_col: str,
    skus: Sequence[str],
) -> pd.DataFrame:
    pivot = history.pivot_table(
        index=date_col, columns=item_col, values=target_col, aggfunc="sum"
    ).sort_index()
    return pivot.reindex(columns=skus, fill_value=0.0).fillna(0.0)


def _long_forecast(
    values: np.ndarray,
    skus: Sequence[str],
    forecast_dates: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    frames = []
    for j, sku in enumerate(skus):
        frame = forecast_dates.copy()
        frame["ItemCode"] = sku
        frame["y_pred"] = values[:, j]
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["model_name"] = model_name
    return out[["model_name", "ItemCode", "Date", "horizon", "y_pred"]]


def all_zero_forecast(skus: Sequence[str], forecast_dates: pd.DataFrame) -> pd.DataFrame:
    values = np.zeros((len(forecast_dates), len(skus)), dtype=float)
    return _long_forecast(values, skus, forecast_dates, "all_zero")


def last_mean_forecast(
    history: pd.DataFrame,
    skus: Sequence[str],
    forecast_dates: pd.DataFrame,
    target_col: str,
    window: int,
) -> pd.DataFrame:
    matrix = _history_matrix(history, "ItemCode", "Date", target_col, skus)
    means = matrix.tail(window).mean(axis=0).to_numpy(dtype=float)
    values = np.tile(means, (len(forecast_dates), 1))
    return _long_forecast(values, skus, forecast_dates, f"last_{window}_mean")


def last_28_repeat_forecast(
    history: pd.DataFrame,
    skus: Sequence[str],
    forecast_dates: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    matrix = _history_matrix(history, "ItemCode", "Date", target_col, skus)
    last = matrix.tail(28).to_numpy(dtype=float)
    if last.shape[0] == 0:
        values = np.zeros((len(forecast_dates), len(skus)), dtype=float)
    else:
        reps = int(np.ceil(len(forecast_dates) / last.shape[0]))
        values = np.tile(last, (reps, 1))[: len(forecast_dates)]
    return _long_forecast(values, skus, forecast_dates, "last_28_repeat")


def day_of_week_mean_forecast(
    history: pd.DataFrame,
    skus: Sequence[str],
    forecast_dates: pd.DataFrame,
    target_col: str,
    lookback_days: int = 365,
) -> pd.DataFrame:
    hist = history.copy()
    cutoff = hist["Date"].max() - pd.Timedelta(days=lookback_days)
    hist = hist.loc[hist["Date"] > cutoff]
    hist["dow"] = hist["Date"].dt.dayofweek
    means = (
        hist.groupby(["ItemCode", "dow"], observed=True)[target_col]
        .mean()
        .rename("dow_mean")
        .reset_index()
    )
    fallback = hist.groupby("ItemCode", observed=True)[target_col].mean().rename("fallback")
    frames = []
    for _, row in forecast_dates.iterrows():
        dow = int(row["Date"].dayofweek)
        pred = pd.DataFrame({"ItemCode": skus})
        pred = pred.merge(means.loc[means["dow"] == dow, ["ItemCode", "dow_mean"]], on="ItemCode", how="left")
        pred = pred.merge(fallback.reset_index(), on="ItemCode", how="left")
        pred["y_pred"] = pred["dow_mean"].fillna(pred["fallback"]).fillna(0.0)
        pred["Date"] = row["Date"]
        pred["horizon"] = int(row["horizon"])
        frames.append(pred[["ItemCode", "Date", "horizon", "y_pred"]])
    out = pd.concat(frames, ignore_index=True)
    out["model_name"] = "day_of_week_mean"
    return out[["model_name", "ItemCode", "Date", "horizon", "y_pred"]]


def seasonal_lag_364_forecast(
    history: pd.DataFrame,
    skus: Sequence[str],
    forecast_dates: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    hist = history[["ItemCode", "Date", target_col]].sort_values(["ItemCode", "Date"]).copy()
    fallback = (
        hist.groupby("ItemCode", observed=True)
        .tail(56)
        .groupby("ItemCode", observed=True)[target_col]
        .mean()
    )
    frames = []
    for _, row in forecast_dates.iterrows():
        lag_date = row["Date"] - pd.Timedelta(days=364)
        pred = pd.DataFrame({"ItemCode": skus})
        lag = hist.loc[hist["Date"] == lag_date, ["ItemCode", target_col]].rename(
            columns={target_col: "lag_value"}
        )
        pred = pred.merge(lag, on="ItemCode", how="left")
        pred["fallback"] = pred["ItemCode"].map(fallback).fillna(0.0)
        pred["y_pred"] = pred["lag_value"].fillna(pred["fallback"]).fillna(0.0)
        pred["Date"] = row["Date"]
        pred["horizon"] = int(row["horizon"])
        frames.append(pred[["ItemCode", "Date", "horizon", "y_pred"]])
    out = pd.concat(frames, ignore_index=True)
    out["model_name"] = "seasonal_lag_364"
    return out[["model_name", "ItemCode", "Date", "horizon", "y_pred"]]


def croston_forecast_series(y: np.ndarray, alpha: float = 0.1, variant: str = "croston") -> float:
    """Return constant Croston/SBA forecast for one series."""

    y = np.asarray(y, dtype=float)
    demand = y[y > 0]
    if len(demand) == 0:
        return 0.0
    first_positive = int(np.argmax(y > 0))
    z = y[first_positive]
    p = max(first_positive + 1, 1)
    interval = 1
    for value in y[first_positive + 1 :]:
        interval += 1
        if value > 0:
            z = alpha * value + (1 - alpha) * z
            p = alpha * interval + (1 - alpha) * p
            interval = 0
    forecast = z / max(p, config.EPSILON)
    if variant == "sba":
        forecast *= 1 - alpha / 2
    return float(max(forecast, 0.0))


def tsb_forecast_series(y: np.ndarray, alpha: float = 0.1, beta: float = 0.1) -> float:
    """Return constant TSB-style forecast for one intermittent series."""

    y = np.asarray(y, dtype=float)
    occurrences = (y > 0).astype(float)
    positives = y[y > 0]
    if len(positives) == 0:
        return 0.0
    p = occurrences[0]
    z = positives[0]
    for value, occurred in zip(y[1:], occurrences[1:]):
        p = beta * occurred + (1 - beta) * p
        if occurred:
            z = alpha * value + (1 - alpha) * z
    return float(max(p * z, 0.0))


def intermittent_baseline_forecast(
    history: pd.DataFrame,
    skus: Sequence[str],
    forecast_dates: pd.DataFrame,
    target_col: str,
    method: str,
) -> pd.DataFrame:
    matrix = _history_matrix(history, "ItemCode", "Date", target_col, skus)
    preds = []
    for sku in skus:
        y = matrix[sku].to_numpy(dtype=float)
        if method == "croston":
            preds.append(croston_forecast_series(y, variant="croston"))
        elif method == "sba":
            preds.append(croston_forecast_series(y, variant="sba"))
        elif method == "tsb":
            preds.append(tsb_forecast_series(y))
        else:
            raise ValueError(f"Unknown intermittent method: {method}")
    values = np.tile(np.array(preds, dtype=float), (len(forecast_dates), 1))
    return _long_forecast(values, skus, forecast_dates, f"{method}_baseline")


def simple_raw_pipeline_forecast(
    history: pd.DataFrame,
    skus: Sequence[str],
    forecast_dates: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Simple baseline inspired by the current notebook: mean demand with Sunday zeroing."""

    # 1. Calculate 56-day and 28-day means
    matrix = _history_matrix(history, "ItemCode", "Date", target_col, skus)
    means_56 = matrix.tail(56).mean(axis=0).to_numpy(dtype=float)
    means_28 = matrix.tail(28).mean(axis=0).to_numpy(dtype=float)

    # 2. Identify Top 100 SKUs and Class A SKUs by profit net
    profit_col = "profit_net" if "profit_net" in history.columns else target_col
    sku_profits = history.groupby("ItemCode")[profit_col].sum().clip(lower=0)
    sku_profits = sku_profits.reindex(skus, fill_value=0.0).to_numpy(dtype=float)
    
    # Blending for Top 100 SKUs by profit
    top_100_idx = np.argsort(-sku_profits)[:100]
    is_top_100 = np.zeros(len(skus), dtype=bool)
    is_top_100[top_100_idx] = True
    
    final_means = np.where(is_top_100, 0.95 * means_56 + 0.05 * means_28, means_56)
    
    # 3. Create baseline long forecast
    values = np.tile(final_means, (len(forecast_dates), 1))
    base = _long_forecast(values, skus, forecast_dates, "simple_pipeline_sunday_zero")
    
    # 4. Sunday zeroing
    sunday_mask = base["Date"].dt.dayofweek == 6
    base.loc[sunday_mask, "y_pred"] = 0.0
    
    # 5. Month-start/end adjustments for Class A SKUs (top 80% cum profit)
    total_profit = sku_profits.sum()
    class_a_mask = np.zeros(len(skus), dtype=bool)
    if total_profit > 0:
        sorted_idx = np.argsort(-sku_profits)
        sorted_profits = sku_profits[sorted_idx]
        cum_profits = np.cumsum(sorted_profits) / total_profit
        
        for idx, val in zip(sorted_idx, cum_profits):
            if val <= 0.80:
                class_a_mask[idx] = True
            else:
                if not class_a_mask.any():
                    class_a_mask[sorted_idx[0]] = True
                break
                
    class_a_items = {skus[i] for i in range(len(skus)) if class_a_mask[i]}
    
    # Check if month boundary (month-start or month-end)
    is_month_boundary = base["Date"].dt.is_month_start | base["Date"].dt.is_month_end
    
    # Apply 0.95 factor for Class A boundary dates
    adjust_mask = is_month_boundary & base["ItemCode"].isin(class_a_items)
    base.loc[adjust_mask, "y_pred"] *= 0.95
    
    return base


def generate_all_baselines(
    history: pd.DataFrame,
    skus: Sequence[str],
    forecast_start: pd.Timestamp,
    target_col: str,
    horizon: int = config.FORECAST_HORIZON,
) -> Dict[str, pd.DataFrame]:
    """Generate all configured baseline forecasts."""

    forecast_dates = make_forecast_dates(forecast_start, horizon=horizon)
    baselines = [
        all_zero_forecast(skus, forecast_dates),
        last_mean_forecast(history, skus, forecast_dates, target_col, 28),
        last_mean_forecast(history, skus, forecast_dates, target_col, 56),
        last_mean_forecast(history, skus, forecast_dates, target_col, 90),
        last_28_repeat_forecast(history, skus, forecast_dates, target_col),
        day_of_week_mean_forecast(history, skus, forecast_dates, target_col),
        seasonal_lag_364_forecast(history, skus, forecast_dates, target_col),
        intermittent_baseline_forecast(history, skus, forecast_dates, target_col, "croston"),
        intermittent_baseline_forecast(history, skus, forecast_dates, target_col, "sba"),
        intermittent_baseline_forecast(history, skus, forecast_dates, target_col, "tsb"),
        simple_raw_pipeline_forecast(history, skus, forecast_dates, target_col),
    ]
    return {frame["model_name"].iloc[0]: frame for frame in baselines}


def score_baselines_for_folds(
    panel: pd.DataFrame,
    weight_table: pd.DataFrame,
    target_col: str = config.DEFAULT_TARGET,
    folds: Iterable[Dict[str, str]] = config.CV_FOLDS,
    skus: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score baseline forecasts on rolling 56-day folds."""

    work = panel.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    if skus is None:
        skus = sorted(work["ItemCode"].unique())
    scores = []
    predictions = []
    for fold in folds:
        train_end = pd.Timestamp(fold["train_end"])
        valid_start = pd.Timestamp(fold["valid_start"])
        valid_end = pd.Timestamp(fold["valid_end"])
        history = work.loc[(work["Date"] <= train_end) & (work["ItemCode"].isin(skus))]
        actual = work.loc[
            (work["Date"] >= valid_start)
            & (work["Date"] <= valid_end)
            & (work["ItemCode"].isin(skus)),
            ["ItemCode", "Date", target_col],
        ]
        baselines = generate_all_baselines(history, skus, valid_start, target_col)
        for name, pred in baselines.items():
            joined = actual.merge(pred, on=["ItemCode", "Date"], how="left")
            joined["y_pred"] = joined["y_pred"].fillna(0.0).clip(lower=0)
            joined["horizon"] = (joined["Date"] - valid_start).dt.days + 1
            score, diag, detail = wrmsse_score(
                joined,
                history,
                weight_table,
                target_col=target_col,
                pred_col="y_pred",
            )
            horizon_scores = score_horizon_slices(
                joined,
                history,
                weight_table,
                target_col=target_col,
                pred_col="y_pred",
            )
            for _, row in horizon_scores.iterrows():
                scores.append(
                    {
                        "fold": fold["fold"],
                        "model_name": name,
                        "slice": row["slice"],
                        "wrmsse": row["wrmsse"],
                        "zero_scale_skus": row["zero_scale_skus"],
                        "scored_skus": row["scored_skus"],
                        "weight_sum_scored": row["weight_sum_scored"],
                    }
                )
            scores.append(
                {
                    "fold": fold["fold"],
                    "model_name": name,
                    "slice": "official_aggregate",
                    "wrmsse": score,
                    "zero_scale_skus": diag["zero_scale_skus"],
                    "scored_skus": diag["scored_skus"],
                    "weight_sum_scored": diag["weight_sum_scored"],
                }
            )
            joined["fold"] = fold["fold"]
            joined["model_name"] = name
            predictions.append(joined)
    return pd.DataFrame(scores), pd.concat(predictions, ignore_index=True)
