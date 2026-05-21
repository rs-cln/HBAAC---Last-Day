"""Demand pattern classification and SKU-level intermittent-demand features."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import config


def adi_cv2(values: np.ndarray) -> Dict[str, float]:
    """Compute Syntetos-Boylan ADI and CV2 from a demand series."""

    values = np.asarray(values, dtype=float)
    positive = values[values > 0]
    positive_days = int(len(positive))
    if positive_days == 0:
        return {
            "positive_days": 0.0,
            "adi": np.inf,
            "cv2": np.inf,
            "mean_when_positive": 0.0,
            "std_when_positive": 0.0,
        }
    mean_pos = float(np.mean(positive))
    std_pos = float(np.std(positive, ddof=0))
    cv2 = float((std_pos / (mean_pos + config.EPSILON)) ** 2)
    return {
        "positive_days": float(positive_days),
        "adi": float(len(values) / positive_days),
        "cv2": cv2,
        "mean_when_positive": mean_pos,
        "std_when_positive": std_pos,
    }


def classify_syntetos_boylan(adi: float, cv2: float) -> str:
    """Classify demand according to Syntetos-Boylan thresholds."""

    if not np.isfinite(adi) or not np.isfinite(cv2):
        return "NoDemand"
    if adi < 1.32 and cv2 < 0.49:
        return "Smooth"
    if adi >= 1.32 and cv2 < 0.49:
        return "Intermittent"
    if adi < 1.32 and cv2 >= 0.49:
        return "Erratic"
    return "Lumpy"


def _interval_stats(dates: pd.Series) -> Dict[str, float]:
    positive_dates = pd.to_datetime(dates).sort_values().drop_duplicates()
    if len(positive_dates) <= 1:
        return {"avg_interval_between_sales": np.nan, "median_interval_between_sales": np.nan}
    intervals = positive_dates.diff().dt.days.dropna()
    return {
        "avg_interval_between_sales": float(intervals.mean()),
        "median_interval_between_sales": float(intervals.median()),
    }


def build_demand_classification(
    panel: pd.DataFrame,
    target_col: str = config.DEFAULT_TARGET,
    weight_table: Optional[pd.DataFrame] = None,
    as_of_date: Optional[pd.Timestamp] = None,
    recent_windows: Iterable[int] = config.RECENT_WINDOWS,
) -> pd.DataFrame:
    """Build demand class and recent activity features for each SKU."""

    required = {"Date", "ItemCode", target_col}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    work = panel[["Date", "ItemCode", target_col]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    as_of = pd.Timestamp(as_of_date) if as_of_date is not None else work["Date"].max()
    work = work.loc[work["Date"] <= as_of].sort_values(["ItemCode", "Date"])

    rows: List[Dict[str, object]] = []
    for sku, grp in work.groupby("ItemCode", observed=True):
        y = grp[target_col].to_numpy(dtype=float)
        all_stats = adi_cv2(y)
        active = grp.loc[grp["Date"] >= grp["Date"].min()]
        active_stats = adi_cv2(active[target_col].to_numpy(dtype=float))
        recent_365 = grp.loc[grp["Date"] > as_of - pd.Timedelta(days=365)]
        recent_stats = adi_cv2(recent_365[target_col].to_numpy(dtype=float))
        positive = grp.loc[grp[target_col] > 0]
        first_date = grp["Date"].min()
        last_sale = positive["Date"].max() if not positive.empty else pd.NaT
        row: Dict[str, object] = {
            "ItemCode": sku,
            "ADI_all": all_stats["adi"],
            "CV2_all": all_stats["cv2"],
            "ADI_active": active_stats["adi"],
            "CV2_active": active_stats["cv2"],
            "ADI_recent_365": recent_stats["adi"],
            "CV2_recent_365": recent_stats["cv2"],
            "Demand_Type": classify_syntetos_boylan(active_stats["adi"], active_stats["cv2"]),
            "avg_qty_when_positive": all_stats["mean_when_positive"],
            "first_sale_date": positive["Date"].min() if not positive.empty else pd.NaT,
            "last_sale_date": last_sale,
            "days_since_last_sale": (
                int((as_of - last_sale).days) if pd.notna(last_sale) else 9999
            ),
            "first_transaction_date": first_date,
            "sku_age_days": int((as_of - first_date).days + 1),
            "zero_ratio": float((grp[target_col] <= 0).mean()),
        }
        row.update(_interval_stats(positive["Date"]))
        for window in recent_windows:
            sub = grp.loc[grp["Date"] > as_of - pd.Timedelta(days=window)]
            row[f"positive_days_last_{window}"] = int((sub[target_col] > 0).sum())
            row[f"sum_last_{window}"] = float(sub[target_col].sum())
            row[f"positive_rate_last_{window}"] = float((sub[target_col] > 0).mean()) if len(sub) else 0.0
        rows.append(row)

    out = pd.DataFrame(rows)
    out["avg_interval_between_sales"] = out["avg_interval_between_sales"].fillna(9999.0)
    out["median_interval_between_sales"] = out["median_interval_between_sales"].fillna(9999.0)

    if weight_table is not None:
        keep = [
            c
            for c in ["ItemCode", "profit_weight", "profit_rank", "abc_group", "profit_clipped"]
            if c in weight_table.columns
        ]
        out = out.merge(weight_table[keep], on="ItemCode", how="left")
        out["profit_weight"] = out["profit_weight"].fillna(0.0)
        out["abc_group"] = out["abc_group"].fillna("ABC_C")
    return out

