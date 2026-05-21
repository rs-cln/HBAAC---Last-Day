"""Forecast inference loops and submission construction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from . import config
from .features import build_feature_frame
from .models import predict_booster
from .utils import finite_nonnegative_frame, write_metadata


def future_dates(
    start_date: str = config.FORECAST_START,
    horizon: int = config.FORECAST_HORIZON,
) -> pd.DataFrame:
    """Return future dates and 1-based horizons."""

    dates = pd.date_range(start=pd.Timestamp(start_date), periods=horizon, freq="D")
    return pd.DataFrame({"Date": dates, "horizon": np.arange(1, horizon + 1)})


def recursive_forecast_lgbm(
    model: object,
    history_panel: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = config.DEFAULT_TARGET,
    objective: str = "tweedie",
    demand_profile: Optional[pd.DataFrame] = None,
    weight_table: Optional[pd.DataFrame] = None,
    start_date: str = config.FORECAST_START,
    horizon: int = config.FORECAST_HORIZON,
    sku_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Recursive 56-day inference using previous predictions as future lags."""

    history = history_panel.copy()
    history["Date"] = pd.to_datetime(history["Date"])
    if sku_order is None:
        sku_order = sorted(history["ItemCode"].unique())
    sku_order = list(sku_order)
    predictions = []

    for _, row in future_dates(start_date, horizon).iterrows():
        current_date = row["Date"]
        horizon_value = int(row["horizon"])
        future_rows = pd.DataFrame({"ItemCode": sku_order, "Date": current_date})
        for col in history.columns:
            if col in future_rows.columns:
                continue
            if col == target_col:
                future_rows[col] = 0.0
            elif pd.api.types.is_numeric_dtype(history[col]):
                future_rows[col] = 0.0
            else:
                future_rows[col] = np.nan
        feature_source = pd.concat([history, future_rows[history.columns]], ignore_index=True)
        features = build_feature_frame(
            feature_source,
            target_col=target_col,
            demand_profile=demand_profile,
            weight_table=weight_table,
        )
        current_features = features.loc[features["Date"] == current_date].copy()
        x = current_features[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_pred = predict_booster(model, x, objective=objective)
        pred_frame = current_features[["ItemCode", "Date"]].copy()
        pred_frame["horizon"] = horizon_value
        pred_frame["y_pred"] = y_pred
        predictions.append(pred_frame)

        append_rows = future_rows.copy()
        append_rows[target_col] = y_pred
        history = pd.concat([history, append_rows[history.columns]], ignore_index=True)

    out = pd.concat(predictions, ignore_index=True)
    out["y_pred"] = out["y_pred"].clip(lower=0)
    return out[["ItemCode", "Date", "horizon", "y_pred"]]


def make_submission_from_56day_forecast(
    forecast: pd.DataFrame,
    sample_submission: pd.DataFrame,
    output_path: Optional[Path] = None,
    run_mode: str = "smoke",
) -> pd.DataFrame:
    """Map h1-h28 to validation rows and h29-h56 to evaluation rows."""

    required_horizons = set(range(1, config.FORECAST_HORIZON + 1))
    got_horizons = set(forecast["horizon"].astype(int).unique())
    if got_horizons != required_horizons:
        raise AssertionError(
            f"Forecast must contain horizons 1..56. Got {sorted(got_horizons)}."
        )
    if forecast.duplicated(["ItemCode", "horizon"]).any():
        raise AssertionError("Forecast contains duplicate ItemCode/horizon rows.")

    f_cols = [f"F{i}" for i in range(1, config.SUBMISSION_HORIZON + 1)]
    wide = forecast.pivot(index="ItemCode", columns="horizon", values="y_pred").fillna(0.0)
    validation = wide.loc[:, list(range(1, 29))].copy()
    evaluation = wide.loc[:, list(range(29, 57))].copy()
    validation.columns = f_cols
    evaluation.columns = f_cols
    validation["id"] = validation.index.astype(str) + "_validation"
    evaluation["id"] = evaluation.index.astype(str) + "_evaluation"
    pred_submission = pd.concat(
        [validation.reset_index(drop=True), evaluation.reset_index(drop=True)],
        ignore_index=True,
    )[["id"] + f_cols]

    final = sample_submission[["id"]].merge(pred_submission, on="id", how="left")
    final[f_cols] = final[f_cols].fillna(0.0).astype(float)
    validate_submission(final, sample_submission)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(output_path, index=False)
        write_metadata(output_path, run_mode, {"rows": len(final), "columns": len(final.columns)})
    return final


def validate_submission(submission: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    """Assert Kaggle submission shape, ids, order, and value constraints."""

    f_cols = [f"F{i}" for i in range(1, config.SUBMISSION_HORIZON + 1)]
    if submission.shape != sample_submission.shape:
        raise AssertionError(
            f"Submission shape {submission.shape} != sample shape {sample_submission.shape}."
        )
    if list(submission.columns) != list(sample_submission.columns):
        raise AssertionError("Submission columns do not match sample_submission exactly.")
    if not submission["id"].is_unique:
        raise AssertionError("Submission ids are not unique.")
    if set(submission["id"]) != set(sample_submission["id"]):
        raise AssertionError("Submission id set does not match sample_submission.")
    if not submission["id"].equals(sample_submission["id"]):
        raise AssertionError("Submission row order does not match sample_submission.")
    finite_nonnegative_frame(submission, f_cols, "submission")
    val = submission.loc[submission["id"].str.endswith("_validation"), f_cols]
    eva = submission.loc[submission["id"].str.endswith("_evaluation"), f_cols]
    if len(val) != len(eva):
        raise AssertionError("Validation/evaluation row counts differ.")


def prediction_summary(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
    history_panel: Optional[pd.DataFrame] = None,
    target_col: str = config.DEFAULT_TARGET,
    weight_table: Optional[pd.DataFrame] = None,
    top_n: int = 20,
) -> Dict[str, object]:
    """Create submission diagnostic summary."""

    validate_submission(submission, sample_submission)
    f_cols = [f"F{i}" for i in range(1, 29)]
    values = submission[f_cols].to_numpy(dtype=float)
    summary: Dict[str, object] = {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "total_predicted_validation": float(
            submission.loc[submission["id"].str.endswith("_validation"), f_cols].sum().sum()
        ),
        "total_predicted_evaluation": float(
            submission.loc[submission["id"].str.endswith("_evaluation"), f_cols].sum().sum()
        ),
    }
    if history_panel is not None:
        hist = history_panel.copy()
        hist["Date"] = pd.to_datetime(hist["Date"])
        max_date = hist["Date"].max()
        summary["last_28_actual_total"] = float(
            hist.loc[hist["Date"] > max_date - pd.Timedelta(days=28), target_col].sum()
        )
        summary["last_56_actual_total"] = float(
            hist.loc[hist["Date"] > max_date - pd.Timedelta(days=56), target_col].sum()
        )
    sku_totals = _submission_sku_totals(submission)
    summary["top_predicted_skus"] = (
        sku_totals.sort_values("predicted_total", ascending=False)
        .head(top_n)
        .to_dict(orient="records")
    )
    if weight_table is not None:
        top_profit = weight_table.sort_values("profit_weight", ascending=False).head(top_n)
        merged = top_profit.merge(sku_totals, on="ItemCode", how="left").fillna(0.0)
        summary["top_profit_sku_predictions"] = merged[
            ["ItemCode", "profit_weight", "profit_rank", "predicted_total"]
        ].to_dict(orient="records")
    return summary


def _submission_sku_totals(submission: pd.DataFrame) -> pd.DataFrame:
    f_cols = [f"F{i}" for i in range(1, 29)]
    tmp = submission.copy()
    tmp["ItemCode"] = tmp["id"].str.replace(r"_(validation|evaluation)$", "", regex=True)
    tmp["predicted_total"] = tmp[f_cols].sum(axis=1)
    return tmp.groupby("ItemCode", observed=True)["predicted_total"].sum().reset_index()


def write_submission_checks(
    path: Path,
    summary: Dict[str, object],
    title: str = "Submission Checks",
) -> None:
    """Write a human-readable submission check report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    scalar_keys = [
        "min",
        "max",
        "mean",
        "median",
        "total_predicted_validation",
        "total_predicted_evaluation",
        "last_28_actual_total",
        "last_56_actual_total",
    ]
    for key in scalar_keys:
        if key in summary:
            lines.append(f"- {key}: {summary[key]}")
    lines.append("")
    lines.append("## Top Predicted SKUs")
    for row in summary.get("top_predicted_skus", [])[:20]:
        lines.append(f"- {row['ItemCode']}: {row['predicted_total']}")
    lines.append("")
    lines.append("## Top Profit SKU Predictions")
    for row in summary.get("top_profit_sku_predictions", [])[:20]:
        lines.append(
            f"- {row['ItemCode']}: weight={row['profit_weight']}, prediction={row['predicted_total']}"
        )
    path.write_text("\n".join(lines) + "\n")

