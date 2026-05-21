"""Official WRMSSE metric and reporting helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config


def compute_rmsse_scale(
    history: pd.DataFrame,
    target_col: str,
    item_col: str = "ItemCode",
    date_col: str = "Date",
) -> pd.DataFrame:
    """Compute the official per-SKU RMSSE denominator from training history."""

    required = {item_col, date_col, target_col}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history missing columns: {sorted(missing)}")
    ordered = history.sort_values([item_col, date_col])
    diffs = ordered.groupby(item_col, observed=True)[target_col].diff()
    scale = (
        diffs.pow(2)
        .groupby(ordered[item_col], observed=True)
        .mean()
        .rename("rmsse_scale")
        .reset_index()
    )
    scale["rmsse_scale"] = scale["rmsse_scale"].fillna(0.0)
    scale["zero_scale"] = scale["rmsse_scale"] <= 0
    return scale


def _prepare_weight_table(
    weight_table: pd.DataFrame,
    item_col: str = "ItemCode",
) -> pd.DataFrame:
    if "profit_weight" not in weight_table.columns:
        raise ValueError("weight_table must contain profit_weight.")
    out = weight_table[[item_col, "profit_weight"]].copy()
    total = float(out["profit_weight"].sum())
    if total <= 0:
        raise ValueError("Sum of profit_weight must be positive.")
    out["profit_weight"] = out["profit_weight"] / total
    return out


def rmsse_by_sku(
    actual_pred: pd.DataFrame,
    history: pd.DataFrame,
    weight_table: pd.DataFrame,
    target_col: str,
    pred_col: str = "y_pred",
    item_col: str = "ItemCode",
    date_col: str = "Date",
    epsilon: float = config.EPSILON,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Return per-SKU RMSSE details and zero-scale diagnostics."""

    required = {item_col, target_col, pred_col}
    missing = required - set(actual_pred.columns)
    if missing:
        raise ValueError(f"actual_pred missing columns: {sorted(missing)}")
    if (actual_pred[pred_col] < 0).any():
        raise ValueError("Predictions must be non-negative before WRMSSE scoring.")

    err = actual_pred.copy()
    err["squared_error"] = (err[target_col] - err[pred_col]).pow(2)
    mse = (
        err.groupby(item_col, observed=True)["squared_error"]
        .mean()
        .rename("forecast_mse")
        .reset_index()
    )
    scale = compute_rmsse_scale(history, target_col, item_col=item_col, date_col=date_col)
    weights = _prepare_weight_table(weight_table, item_col=item_col)
    scored = mse.merge(scale, on=item_col, how="left").merge(weights, on=item_col, how="left")
    scored["rmsse_scale"] = scored["rmsse_scale"].fillna(0.0)
    scored["zero_scale"] = scored["rmsse_scale"] <= 0
    scored["profit_weight"] = scored["profit_weight"].fillna(0.0)
    zero_scale_count = int(scored["zero_scale"].sum())
    scored["rmsse_scale_used"] = np.where(
        scored["zero_scale"], epsilon, scored["rmsse_scale"]
    )
    scored["rmsse"] = np.sqrt(scored["forecast_mse"] / scored["rmsse_scale_used"])
    scored["weighted_rmsse"] = scored["profit_weight"] * scored["rmsse"]
    diagnostics = {
        "zero_scale_skus": zero_scale_count,
        "scored_skus": int(scored[item_col].nunique()),
        "weight_sum_scored": float(scored["profit_weight"].sum()),
        "epsilon_used_for_zero_scale": epsilon if zero_scale_count else 0.0,
    }
    return scored, diagnostics


def wrmsse_score(
    actual_pred: pd.DataFrame,
    history: pd.DataFrame,
    weight_table: pd.DataFrame,
    target_col: str,
    pred_col: str = "y_pred",
    item_col: str = "ItemCode",
    date_col: str = "Date",
    epsilon: float = config.EPSILON,
) -> Tuple[float, Dict[str, object], pd.DataFrame]:
    """Compute official weighted RMSSE."""

    detail, diagnostics = rmsse_by_sku(
        actual_pred=actual_pred,
        history=history,
        weight_table=weight_table,
        target_col=target_col,
        pred_col=pred_col,
        item_col=item_col,
        date_col=date_col,
        epsilon=epsilon,
    )
    score = float(detail["weighted_rmsse"].sum())
    diagnostics["wrmsse"] = score
    return score, diagnostics, detail


def wrmsse_score_from_matrices(
    history_values: np.ndarray,
    actual_values: np.ndarray,
    pred_values: np.ndarray,
    item_codes: Sequence[str],
    weight_table: pd.DataFrame,
    epsilon: float = config.EPSILON,
) -> Tuple[float, Dict[str, object], pd.DataFrame]:
    """Compute official WRMSSE from dense matrices.

    Rows are dates or forecast horizons and columns are SKUs aligned to
    ``item_codes``. This is equivalent to :func:`wrmsse_score` but avoids
    repeatedly expanding dense panel history to long format during full CV.
    """

    history = np.asarray(history_values, dtype=np.float64)
    actual = np.asarray(actual_values, dtype=np.float64)
    pred = np.asarray(pred_values, dtype=np.float64)
    if history.ndim != 2 or actual.ndim != 2 or pred.ndim != 2:
        raise ValueError("history_values, actual_values, and pred_values must be 2D.")
    if actual.shape != pred.shape:
        raise ValueError(f"actual shape {actual.shape} != pred shape {pred.shape}.")
    if history.shape[1] != actual.shape[1] or history.shape[1] != len(item_codes):
        raise ValueError("Matrix column counts must match item_codes.")
    if (pred < 0).any():
        raise ValueError("Predictions must be non-negative before WRMSSE scoring.")

    diffs = np.diff(history, axis=0)
    scale = np.mean(np.square(diffs), axis=0) if len(history) > 1 else np.zeros(history.shape[1])
    mse = np.mean(np.square(actual - pred), axis=0)

    weights = _prepare_weight_table(weight_table)
    aligned_weights = (
        weights.set_index("ItemCode")
        .reindex(pd.Index(item_codes, name="ItemCode"))["profit_weight"]
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    zero_scale = scale <= 0
    scale_used = np.where(zero_scale, epsilon, scale)
    rmsse = np.sqrt(mse / scale_used)
    weighted = aligned_weights * rmsse
    detail = pd.DataFrame(
        {
            "ItemCode": list(item_codes),
            "forecast_mse": mse,
            "rmsse_scale": scale,
            "zero_scale": zero_scale,
            "profit_weight": aligned_weights,
            "rmsse_scale_used": scale_used,
            "rmsse": rmsse,
            "weighted_rmsse": weighted,
        }
    )
    diagnostics = {
        "zero_scale_skus": int(zero_scale.sum()),
        "scored_skus": int(len(item_codes)),
        "weight_sum_scored": float(aligned_weights.sum()),
        "epsilon_used_for_zero_scale": epsilon if int(zero_scale.sum()) else 0.0,
        "wrmsse": float(weighted.sum()),
    }
    return diagnostics["wrmsse"], diagnostics, detail


def score_horizon_slices_from_matrices(
    history_values: np.ndarray,
    actual_values: np.ndarray,
    pred_values: np.ndarray,
    item_codes: Sequence[str],
    weight_table: pd.DataFrame,
    epsilon: float = config.EPSILON,
) -> pd.DataFrame:
    """Score h1-28, h29-56, and all horizons from dense matrices."""

    slices = {
        "h1_28": (slice(0, 28), actual_values[:28], pred_values[:28]),
        "h29_56": (slice(28, 56), actual_values[28:56], pred_values[28:56]),
        "h1_56": (slice(None), actual_values, pred_values),
    }
    rows = []
    for name, (_, actual_slice, pred_slice) in slices.items():
        score, diag, _ = wrmsse_score_from_matrices(
            history_values=history_values,
            actual_values=actual_slice,
            pred_values=pred_slice,
            item_codes=item_codes,
            weight_table=weight_table,
            epsilon=epsilon,
        )
        rows.append(
            {
                "slice": name,
                "wrmsse": score,
                "zero_scale_skus": diag["zero_scale_skus"],
                "scored_skus": diag["scored_skus"],
                "weight_sum_scored": diag["weight_sum_scored"],
            }
        )
    return pd.DataFrame(rows)


def score_horizon_slices(
    actual_pred: pd.DataFrame,
    history: pd.DataFrame,
    weight_table: pd.DataFrame,
    target_col: str,
    pred_col: str = "y_pred",
    horizon_col: str = "horizon",
) -> pd.DataFrame:
    """Score h1-28, h29-56, and all horizons for one fold."""

    if horizon_col not in actual_pred.columns:
        raise ValueError("actual_pred must include a horizon column.")
    slices = {
        "h1_28": actual_pred.loc[actual_pred[horizon_col].between(1, 28)],
        "h29_56": actual_pred.loc[actual_pred[horizon_col].between(29, 56)],
        "h1_56": actual_pred,
    }
    rows = []
    for name, frame in slices.items():
        score, diag, _ = wrmsse_score(
            frame,
            history,
            weight_table,
            target_col=target_col,
            pred_col=pred_col,
        )
        rows.append(
            {
                "slice": name,
                "wrmsse": score,
                "zero_scale_skus": diag["zero_scale_skus"],
                "scored_skus": diag["scored_skus"],
                "weight_sum_scored": diag["weight_sum_scored"],
            }
        )
    return pd.DataFrame(rows)


def segment_wrmsse_report(
    per_sku_detail: pd.DataFrame,
    sku_segments: Optional[pd.DataFrame] = None,
    item_col: str = "ItemCode",
) -> pd.DataFrame:
    """Report WRMSSE for top-profit, demand classes, and long-tail segments."""

    detail = per_sku_detail.copy()
    if sku_segments is not None:
        keep_cols = [
            c
            for c in ["Demand_Type", "abc_group", "profit_rank", "profit_weight"]
            if c in sku_segments.columns
        ]
        segment_cols = [item_col] + keep_cols
        detail = detail.drop(columns=[c for c in keep_cols if c in detail.columns], errors="ignore")
        detail = detail.merge(sku_segments[segment_cols], on=item_col, how="left")

    rows = [
        {
            "segment": "overall",
            "sku_count": int(detail[item_col].nunique()),
            "weight_sum": float(detail["profit_weight"].sum()),
            "wrmsse": float(detail["weighted_rmsse"].sum()),
        }
    ]

    if "profit_rank" in detail.columns:
        for n in [50, 100, 500, 1000]:
            frame = detail.loc[detail["profit_rank"] <= n]
            rows.append(_segment_row(f"top_profit_{n}", frame, item_col))
        frame = detail.loc[detail["profit_rank"] > 1000]
        rows.append(_segment_row("long_tail_profit_rank_gt_1000", frame, item_col))

    if "Demand_Type" in detail.columns:
        for name, frame in detail.groupby("Demand_Type", dropna=False, observed=True):
            rows.append(_segment_row(f"demand_type_{name}", frame, item_col))

    if "abc_group" in detail.columns:
        for name, frame in detail.groupby("abc_group", dropna=False, observed=True):
            rows.append(_segment_row(f"abc_{name}", frame, item_col))

    return pd.DataFrame(rows)


def _segment_row(segment: str, frame: pd.DataFrame, item_col: str) -> Dict[str, object]:
    if frame.empty:
        return {"segment": segment, "sku_count": 0, "weight_sum": 0.0, "wrmsse": np.nan}
    return {
        "segment": segment,
        "sku_count": int(frame[item_col].nunique()),
        "weight_sum": float(frame["profit_weight"].sum()),
        "wrmsse": float(frame["weighted_rmsse"].sum()),
    }


def rmse(actual_pred: pd.DataFrame, target_col: str, pred_col: str = "y_pred") -> float:
    """Secondary diagnostic only."""

    return float(np.sqrt(np.mean((actual_pred[target_col] - actual_pred[pred_col]) ** 2)))


def assert_metric_inputs(
    history: pd.DataFrame,
    actual_pred: pd.DataFrame,
    target_col: str,
    item_col: str = "ItemCode",
    horizon_values: Optional[Iterable[int]] = None,
) -> None:
    """Validate common metric input mistakes early."""

    if target_col not in history.columns or target_col not in actual_pred.columns:
        raise AssertionError(f"{target_col} must exist in history and actual_pred.")
    if item_col not in history.columns or item_col not in actual_pred.columns:
        raise AssertionError(f"{item_col} must exist in history and actual_pred.")
    if horizon_values is not None:
        got = set(actual_pred["horizon"].unique())
        expected = set(horizon_values)
        if got != expected:
            raise AssertionError(f"Expected horizons {sorted(expected)}, got {sorted(got)}.")
