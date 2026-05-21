"""Post-processing and ensemble utilities."""

from __future__ import annotations

from itertools import product
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import config
from .metrics import wrmsse_score


def apply_postprocess_rules(
    forecast: pd.DataFrame,
    sku_profile: Optional[pd.DataFrame] = None,
    history_panel: Optional[pd.DataFrame] = None,
    target_col: str = config.DEFAULT_TARGET,
    dead_sum_last_365_zero: bool = True,
    cold_positive_days_threshold: Optional[int] = None,
    cold_multiplier: float = 1.0,
    cap_quantile: Optional[float] = None,
) -> pd.DataFrame:
    """Apply conservative post-processing rules tuned by CV."""

    out = forecast.copy()
    out["y_pred"] = out["y_pred"].clip(lower=0)

    if sku_profile is not None and dead_sum_last_365_zero and "sum_last_365" in sku_profile.columns:
        dead = set(sku_profile.loc[sku_profile["sum_last_365"] <= 0, "ItemCode"])
        out.loc[out["ItemCode"].isin(dead), "y_pred"] = 0.0

    if (
        sku_profile is not None
        and cold_positive_days_threshold is not None
        and "positive_days_last_180" in sku_profile.columns
    ):
        cold = set(
            sku_profile.loc[
                sku_profile["positive_days_last_180"] <= cold_positive_days_threshold,
                "ItemCode",
            ]
        )
        out.loc[out["ItemCode"].isin(cold), "y_pred"] *= cold_multiplier

    if history_panel is not None and cap_quantile is not None:
        caps = (
            history_panel.groupby("ItemCode", observed=True)[target_col]
            .quantile(cap_quantile)
            .rename("cap")
            .reset_index()
        )
        out = out.merge(caps, on="ItemCode", how="left")
        out["cap"] = out["cap"].fillna(np.inf)
        out["y_pred"] = np.minimum(out["y_pred"], out["cap"])
        out = out.drop(columns=["cap"])

    out["y_pred"] = out["y_pred"].clip(lower=0)
    return out


def ensemble_forecasts(
    forecasts: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    pred_col: str = "y_pred",
) -> pd.DataFrame:
    """Weighted average of long-format forecasts."""

    total_weight = float(sum(max(v, 0.0) for v in weights.values()))
    if total_weight <= 0:
        raise ValueError("Ensemble weights must contain positive mass.")
    normalized = {k: max(v, 0.0) / total_weight for k, v in weights.items()}
    base_keys = ["ItemCode", "Date", "horizon"]
    merged: Optional[pd.DataFrame] = None
    for name, frame in forecasts.items():
        if name not in normalized or normalized[name] == 0:
            continue
        part = frame[base_keys + [pred_col]].rename(columns={pred_col: f"pred_{name}"})
        merged = part if merged is None else merged.merge(part, on=base_keys, how="outer")
    if merged is None:
        raise ValueError("No forecasts selected for ensemble.")
    pred_cols = [c for c in merged.columns if c.startswith("pred_")]
    merged[pred_cols] = merged[pred_cols].fillna(0.0)
    merged["y_pred"] = 0.0
    for name, weight in normalized.items():
        col = f"pred_{name}"
        if col in merged.columns:
            merged["y_pred"] += weight * merged[col]
    return merged[base_keys + ["y_pred"]]


def tune_postprocess_grid(
    base_forecast: pd.DataFrame,
    actual: pd.DataFrame,
    history: pd.DataFrame,
    weight_table: pd.DataFrame,
    sku_profile: pd.DataFrame,
    target_col: str = config.DEFAULT_TARGET,
) -> pd.DataFrame:
    """Small controlled grid for post-processing choices."""

    rows = []
    for threshold, multiplier, cap_q in product([None, 0, 1, 3, 5], [1.0, 0.75, 0.5], [None, 0.99, 0.995]):
        processed = apply_postprocess_rules(
            base_forecast,
            sku_profile=sku_profile,
            history_panel=history,
            target_col=target_col,
            dead_sum_last_365_zero=True,
            cold_positive_days_threshold=threshold,
            cold_multiplier=multiplier,
            cap_quantile=cap_q,
        )
        joined = actual.merge(processed, on=["ItemCode", "Date"], how="left")
        joined["y_pred"] = joined["y_pred"].fillna(0.0).clip(lower=0)
        score, diag, _ = wrmsse_score(joined, history, weight_table, target_col)
        rows.append(
            {
                "positive_days_threshold": threshold,
                "cold_multiplier": multiplier,
                "cap_quantile": cap_q,
                "wrmsse": score,
                "zero_scale_skus": diag["zero_scale_skus"],
            }
        )
    return pd.DataFrame(rows).sort_values("wrmsse")


def tune_ensemble_weights_random(
    forecasts: Dict[str, pd.DataFrame],
    actual: pd.DataFrame,
    history: pd.DataFrame,
    weight_table: pd.DataFrame,
    target_col: str = config.DEFAULT_TARGET,
    n_trials: int = 500,
    seed: int = config.RANDOM_SEED,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Random simplex search for non-negative ensemble weights."""

    rng = np.random.default_rng(seed)
    names = list(forecasts.keys())
    rows = []
    best_score = np.inf
    best_weights: Dict[str, float] = {}
    for trial in range(n_trials):
        draw = rng.dirichlet(np.ones(len(names)))
        weights = dict(zip(names, draw))
        ens = ensemble_forecasts(forecasts, weights)
        joined = actual.merge(ens, on=["ItemCode", "Date"], how="left")
        joined["y_pred"] = joined["y_pred"].fillna(0.0).clip(lower=0)
        score, diag, _ = wrmsse_score(joined, history, weight_table, target_col)
        row = {"trial": trial, "wrmsse": score, "zero_scale_skus": diag["zero_scale_skus"]}
        row.update({f"weight_{name}": weights[name] for name in names})
        rows.append(row)
        if score < best_score:
            best_score = score
            best_weights = weights
    return best_weights, pd.DataFrame(rows).sort_values("wrmsse")

