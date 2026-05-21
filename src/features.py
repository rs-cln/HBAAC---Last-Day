"""Leakage-safe feature engineering for daily SKU demand."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from . import config


VN_HOLIDAYS = pd.to_datetime(
    [
        "2021-01-01",
        "2022-01-01",
        "2023-01-01",
        "2024-01-01",
        "2025-01-01",
        "2021-02-10",
        "2021-02-11",
        "2021-02-12",
        "2021-02-13",
        "2021-02-14",
        "2022-01-31",
        "2022-02-01",
        "2022-02-02",
        "2022-02-03",
        "2022-02-04",
        "2023-01-22",
        "2023-01-23",
        "2023-01-24",
        "2023-01-25",
        "2023-01-26",
        "2024-02-08",
        "2024-02-09",
        "2024-02-10",
        "2024-02-11",
        "2024-02-12",
        "2025-01-27",
        "2025-01-28",
        "2025-01-29",
        "2025-01-30",
        "2025-01-31",
        "2021-04-21",
        "2022-04-10",
        "2023-04-29",
        "2024-04-18",
        "2025-04-07",
        "2021-04-30",
        "2021-05-01",
        "2022-04-30",
        "2022-05-01",
        "2023-04-30",
        "2023-05-01",
        "2024-04-30",
        "2024-05-01",
        "2025-04-30",
        "2025-05-01",
        "2021-09-02",
        "2022-09-02",
        "2023-09-02",
        "2024-09-02",
        "2025-09-02",
        "2025-10-06",
    ]
)

TET_DATES = pd.to_datetime(
    ["2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10", "2025-01-29", "2026-02-17"]
)


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Add deterministic calendar features known for future dates."""

    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    iso = dates.dt.isocalendar()
    out["day_of_week"] = dates.dt.dayofweek.astype(np.int16)
    out["week_of_year"] = iso.week.astype(np.int16)
    out["month"] = dates.dt.month.astype(np.int16)
    out["day_of_month"] = dates.dt.day.astype(np.int16)
    out["day_of_year"] = dates.dt.dayofyear.astype(np.int16)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(np.int8)
    out["is_month_start"] = dates.dt.is_month_start.astype(np.int8)
    out["is_month_end"] = dates.dt.is_month_end.astype(np.int8)
    out["quarter"] = dates.dt.quarter.astype(np.int16)
    out["year"] = dates.dt.year.astype(np.int16)
    out["days_to_month_end"] = (dates.dt.days_in_month - dates.dt.day).astype(np.int16)
    out["is_vn_holiday"] = dates.isin(VN_HOLIDAYS).astype(np.int8)
    out["days_to_tet"] = _days_to_nearest_event(dates, TET_DATES, direction="next")
    out["days_since_tet"] = _days_to_nearest_event(dates, TET_DATES, direction="previous")
    return out


def _days_to_nearest_event(
    dates: pd.Series,
    events: pd.DatetimeIndex,
    direction: str,
    clip_value: int = 90,
) -> np.ndarray:
    dates_np = dates.to_numpy(dtype="datetime64[D]")
    events_np = events.to_numpy(dtype="datetime64[D]")
    result = np.full(len(dates_np), clip_value + 1, dtype=np.int16)
    for event in events_np:
        if direction == "next":
            diff = (event - dates_np).astype("timedelta64[D]").astype(np.int32)
        else:
            diff = (dates_np - event).astype("timedelta64[D]").astype(np.int32)
        valid = (diff >= 0) & (diff < result)
        result = np.where(valid, diff, result)
    return np.clip(result, 0, clip_value).astype(np.int16)


def add_lag_rolling_features(
    panel: pd.DataFrame,
    target_col: str,
    lags: Iterable[int] = config.LAGS,
    windows: Iterable[int] = config.ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Add lag and rolling features using only shifted target history."""

    out = panel.sort_values(["ItemCode", "Date"]).copy()
    group = out.groupby("ItemCode", observed=True)[target_col]
    for lag in lags:
        out[f"lag_{lag}"] = group.shift(lag)

    shifted = group.shift(1)
    shifted_by_sku = shifted.groupby(out["ItemCode"], observed=True)
    for window in windows:
        roll = shifted_by_sku.rolling(window=window, min_periods=1)
        out[f"rolling_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
        out[f"rolling_sum_{window}"] = roll.sum().reset_index(level=0, drop=True)
        out[f"rolling_std_{window}"] = roll.std().reset_index(level=0, drop=True)
        out[f"rolling_min_{window}"] = roll.min().reset_index(level=0, drop=True)
        out[f"rolling_max_{window}"] = roll.max().reset_index(level=0, drop=True)
        out[f"rolling_median_{window}"] = roll.median().reset_index(level=0, drop=True)
        out[f"rolling_p90_{window}"] = (
            shifted_by_sku.rolling(window=window, min_periods=1)
            .quantile(0.90)
            .reset_index(level=0, drop=True)
        )
        positive = shifted.where(shifted > 0, 0.0)
        positive_roll = positive.groupby(out["ItemCode"], observed=True).rolling(
            window=window, min_periods=1
        )
        positive_count = (
            (shifted > 0)
            .astype(float)
            .groupby(out["ItemCode"], observed=True)
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        observed_count = (
            shifted.notna()
            .astype(float)
            .groupby(out["ItemCode"], observed=True)
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        out[f"positive_days_count_{window}"] = positive_count
        out[f"zero_days_count_{window}"] = observed_count - positive_count
        out[f"mean_when_positive_{window}"] = (
            positive_roll.sum().reset_index(level=0, drop=True)
            / (positive_count + config.EPSILON)
        )

    out = add_intermitent_features(out, target_col)
    return out


def add_intermitent_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Add leakage-safe intermittent-demand features."""

    out = df.sort_values(["ItemCode", "Date"]).copy()
    shifted = out.groupby("ItemCode", observed=True)[target_col].shift(1).fillna(0.0)
    out["zero_streak"] = _zero_streak_by_group(out["ItemCode"], shifted)
    out["days_since_last_sale_feature"] = _days_since_last_positive_by_group(
        out["ItemCode"], shifted
    )
    for window in config.RECENT_WINDOWS:
        pos_col = f"positive_days_count_{window}"
        sum_col = f"rolling_sum_{window}"
        if pos_col in out.columns:
            out[f"positive_rate_last_{window}"] = out[pos_col] / float(window)
        if sum_col in out.columns:
            out[f"sum_last_{window}"] = out[sum_col]
    if "rolling_sum_28" in out.columns and "rolling_sum_180" in out.columns:
        out["recent_sales_intensity"] = out["rolling_sum_28"] / (
            out["rolling_sum_180"] + config.EPSILON
        )
    if "rolling_max_90" in out.columns and "rolling_mean_90" in out.columns:
        out["demand_burst_score"] = out["rolling_max_90"] / (
            out["rolling_mean_90"] + config.EPSILON
        )
    return out


def _zero_streak_by_group(items: pd.Series, shifted: pd.Series) -> np.ndarray:
    values = np.zeros(len(shifted), dtype=np.int16)
    current_item = None
    streak = 0
    for i, (item, value) in enumerate(zip(items.to_numpy(), shifted.to_numpy())):
        if item != current_item:
            current_item = item
            streak = 0
        if value <= 0:
            streak += 1
        else:
            streak = 0
        values[i] = min(streak, np.iinfo(np.int16).max)
    return values


def _days_since_last_positive_by_group(items: pd.Series, shifted: pd.Series) -> np.ndarray:
    values = np.zeros(len(shifted), dtype=np.int16)
    current_item = None
    days = 9999
    for i, (item, value) in enumerate(zip(items.to_numpy(), shifted.to_numpy())):
        if item != current_item:
            current_item = item
            days = 9999
        if value > 0:
            days = 1
        else:
            days = min(days + 1, 9999)
        values[i] = min(days, 9999)
    return values


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged/last-known price features without same-day leakage."""

    out = df.sort_values(["ItemCode", "Date"]).copy()
    if "daily_valid_price" not in out.columns:
        out["daily_valid_price"] = np.nan
    group = out.groupby("ItemCode", observed=True)["daily_valid_price"]
    out["last_valid_price"] = group.ffill().groupby(out["ItemCode"], observed=True).shift(1)
    for lag in [1, 7, 28]:
        out[f"price_lag_{lag}"] = group.shift(lag)
    out["sku_median_price"] = group.transform(
        lambda s: s.shift(1).expanding(min_periods=1).median()
    )
    global_median = out["daily_valid_price"].dropna().median()
    if not np.isfinite(global_median):
        global_median = 0.0
    for col in ["last_valid_price", "price_lag_1", "price_lag_7", "price_lag_28", "sku_median_price"]:
        out[col] = out[col].fillna(global_median)
    out["price_vs_sku_median"] = out["last_valid_price"] / (
        out["sku_median_price"] + config.EPSILON
    )
    out["price_changed_recently"] = (
        (out["price_lag_1"] - out["price_lag_28"]).abs()
        > (0.01 * out["sku_median_price"].abs() + config.EPSILON)
    ).astype(np.int8)
    return out


def merge_static_features(
    features: pd.DataFrame,
    demand_profile: Optional[pd.DataFrame] = None,
    weight_table: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge static SKU features used by models and reports."""

    out = features.copy()
    if demand_profile is not None:
        drop = [c for c in ["Demand_Type", "abc_group", "profit_weight", "profit_rank"] if c in out.columns]
        out = out.drop(columns=drop, errors="ignore")
        keep = [c for c in demand_profile.columns if c != "Date"]
        out = out.merge(demand_profile[keep], on="ItemCode", how="left")
    if weight_table is not None:
        keep = [
            c
            for c in ["ItemCode", "profit_weight", "profit_rank", "abc_group"]
            if c in weight_table.columns
        ]
        existing = [c for c in keep if c != "ItemCode" and c in out.columns]
        out = out.drop(columns=existing, errors="ignore")
        out = out.merge(weight_table[keep], on="ItemCode", how="left")
    return out


def build_feature_frame(
    panel: pd.DataFrame,
    target_col: str = config.DEFAULT_TARGET,
    demand_profile: Optional[pd.DataFrame] = None,
    weight_table: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build complete training feature frame."""

    out = add_calendar_features(panel)
    out = add_lag_rolling_features(out, target_col=target_col)
    out = add_price_features(out)
    out = merge_static_features(out, demand_profile=demand_profile, weight_table=weight_table)
    out["ItemCode_cat"] = out["ItemCode"].astype("category").cat.codes.astype(np.int32)
    for cat_col in ["Demand_Type", "abc_group"]:
        if cat_col in out.columns:
            out[f"{cat_col}_cat"] = out[cat_col].astype("category").cat.codes.astype(np.int16)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out


def model_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """Return model feature columns, excluding target/leakage fields."""

    excluded = {
        "Date",
        "ItemCode",
        target_col,
        "net_daily_qty",
        "net_daily_qty_clip0",
        "gross_positive_qty",
        "direct_target",
        "target_for_model",
        "sales_amount_net",
        "cost_amount_net",
        "profit_net",
        "daily_valid_price",
        "daily_valid_unit_cost",
        "first_sale_date",
        "last_sale_date",
        "first_transaction_date",
        "last_transaction_date",
    }
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols
