"""Data loading, cleaning, aggregation, and panel construction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger, require_columns, write_metadata

LOGGER = get_logger(__name__)


RAW_REQUIRED_COLUMNS = [
    "Date",
    "Stt",
    "ItemCode",
    "Quantity",
    "UnitPrice",
    "SalesAmount",
    "Unit_Cost",
    "Cost_Amount",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names while preserving values."""

    out = df.copy()
    out.columns = out.columns.astype(str).str.strip().str.replace(" ", "_", regex=False)
    return out


def parse_decimal_comma(series: pd.Series) -> pd.Series:
    """Parse numeric strings that may use comma as the decimal separator."""

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def load_sample_submission(path: Path) -> pd.DataFrame:
    """Load and validate sample submission format."""

    sample = pd.read_csv(path, dtype={"id": str})
    f_cols = [f"F{i}" for i in range(1, config.SUBMISSION_HORIZON + 1)]
    require_columns(sample.columns, ["id"] + f_cols, "sample_submission")
    if not sample["id"].is_unique:
        raise ValueError("sample_submission contains duplicate id values.")
    suffix_ok = sample["id"].str.endswith(("_validation", "_evaluation"))
    if not suffix_ok.all():
        bad = sample.loc[~suffix_ok, "id"].head().tolist()
        raise ValueError(f"sample_submission has unexpected ids: {bad}")
    return sample


def extract_submission_skus(sample: pd.DataFrame) -> pd.Index:
    """Extract SKU ids from validation/evaluation submission rows."""

    skus = sample["id"].str.replace(
        r"_(validation|evaluation)$", "", regex=True
    )
    return pd.Index(skus.drop_duplicates(), name="ItemCode")


def load_transactions(path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load raw train transactions with robust parsing and return flags."""

    raw = pd.read_csv(path, dtype=str)
    raw = normalize_columns(raw)
    require_columns(raw.columns, RAW_REQUIRED_COLUMNS, "train.csv")

    for col in ["Quantity", "SalesAmount", "Cost_Amount"]:
        raw[f"{col}_raw"] = raw[col]

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw["Stt"] = raw["Stt"].astype(str).str.strip()
    raw["ItemCode"] = raw["ItemCode"].astype(str).str.strip()

    for col in ["Quantity", "UnitPrice", "SalesAmount", "Unit_Cost", "Cost_Amount"]:
        raw[col] = parse_decimal_comma(raw[col])

    initial_rows = int(len(raw))
    critical_bad = (
        raw["Date"].isna()
        | raw["ItemCode"].isna()
        | raw["ItemCode"].eq("")
        | raw["Quantity"].isna()
        | raw["SalesAmount"].isna()
        | raw["Cost_Amount"].isna()
    )
    clean = raw.loc[~critical_bad].copy()

    clean["is_negative_qty"] = (clean["Quantity"] < 0).astype(np.int8)
    clean["is_negative_sales"] = (clean["SalesAmount"] < 0).astype(np.int8)
    clean["is_negative_cost"] = (clean["Cost_Amount"] < 0).astype(np.int8)
    clean["is_return"] = (
        (clean["Quantity"] < 0)
        & (clean["SalesAmount"] < 0)
        & (clean["Cost_Amount"] < 0)
    ).astype(np.int8)
    clean["is_zero_quantity"] = (clean["Quantity"] == 0).astype(np.int8)
    clean["profit_line"] = clean["SalesAmount"] - clean["Cost_Amount"]
    clean["positive_quantity"] = clean["Quantity"].clip(lower=0)
    clean["return_quantity_abs"] = (-clean["Quantity"].clip(upper=0)).astype(float)
    clean["valid_price"] = np.where(
        (clean["Quantity"] > 0) & (clean["UnitPrice"] > 0) & (clean["Unit_Cost"] > 0),
        clean["UnitPrice"],
        np.nan,
    )
    clean["valid_unit_cost"] = np.where(
        (clean["Quantity"] > 0) & (clean["UnitPrice"] > 0) & (clean["Unit_Cost"] > 0),
        clean["Unit_Cost"],
        np.nan,
    )
    clean["sign_mismatch"] = (
        ((clean["Quantity"] < 0) & ((clean["SalesAmount"] > 0) | (clean["Cost_Amount"] > 0)))
        | ((clean["Quantity"] > 0) & ((clean["SalesAmount"] < 0) | (clean["Cost_Amount"] < 0)))
    ).astype(np.int8)

    report = {
        "initial_rows": initial_rows,
        "rows_after_critical_cleaning": int(len(clean)),
        "dropped_critical_rows": int(critical_bad.sum()),
        "unique_skus": int(clean["ItemCode"].nunique()),
        "date_min": clean["Date"].min(),
        "date_max": clean["Date"].max(),
        "negative_quantity_rows": int((clean["Quantity"] < 0).sum()),
        "all_negative_return_rows": int(clean["is_return"].sum()),
        "zero_quantity_rows": int(clean["is_zero_quantity"].sum()),
        "sign_mismatch_rows": int(clean["sign_mismatch"].sum()),
    }
    return clean, report


def compute_profit_weights(
    transactions: pd.DataFrame,
    sample_skus: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Compute official profit weights and ABC groups."""

    profit = transactions.groupby("ItemCode", observed=True)["profit_line"].sum()
    if sample_skus is not None:
        profit = profit.reindex(pd.Index(sample_skus, name="ItemCode"), fill_value=0.0)
    weight = profit.clip(lower=0).rename("profit_clipped").reset_index()
    total_profit = float(weight["profit_clipped"].sum())
    if total_profit <= 0:
        raise ValueError("Total positive profit is zero; official weights cannot be computed.")
    weight["profit_raw"] = weight["ItemCode"].map(profit).astype(float)
    weight["profit_weight"] = weight["profit_clipped"] / total_profit
    weight = weight.sort_values("profit_clipped", ascending=False).reset_index(drop=True)
    weight["profit_rank"] = np.arange(1, len(weight) + 1)
    weight["cum_profit_share"] = weight["profit_clipped"].cumsum() / total_profit
    weight["abc_group"] = np.select(
        [
            weight["cum_profit_share"] <= 0.80,
            weight["cum_profit_share"] <= 0.95,
        ],
        ["ABC_A", "ABC_B"],
        default="ABC_C",
    )
    return weight.sort_values("ItemCode").reset_index(drop=True)


def profit_concentration(weight_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize profit concentration for audit and reporting."""

    ordered = weight_table.sort_values("profit_clipped", ascending=False)
    rows = []
    total = float(ordered["profit_clipped"].sum())
    for top_n in [10, 50, 100, 500, 1000]:
        rows.append(
            {
                "top_n": top_n,
                "profit_share": float(ordered.head(top_n)["profit_clipped"].sum() / total),
            }
        )
    return pd.DataFrame(rows)


def select_smoke_skus(
    sample_skus: Sequence[str],
    weight_table: pd.DataFrame,
    n_skus: int = config.SMOKE_N_SKUS,
    top_profit_skus: int = config.SMOKE_TOP_PROFIT_SKUS,
) -> List[str]:
    """Pick a deterministic smoke subset preserving high-profit SKUs."""

    top = (
        weight_table.sort_values("profit_weight", ascending=False)
        .head(top_profit_skus)["ItemCode"]
        .tolist()
    )
    ordered_sample = list(sample_skus)
    selected = []
    seen = set()
    for sku in top + ordered_sample:
        if sku in seen:
            continue
        selected.append(sku)
        seen.add(sku)
        if len(selected) >= n_skus:
            break
    return selected


def aggregate_daily_sparse(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction rows to sparse daily SKU rows."""

    group_cols = ["Date", "ItemCode"]
    daily = (
        transactions.groupby(group_cols, observed=True)
        .agg(
            net_daily_qty=("Quantity", "sum"),
            gross_positive_qty=("positive_quantity", "sum"),
            return_qty=("return_quantity_abs", "sum"),
            sales_amount_net=("SalesAmount", "sum"),
            cost_amount_net=("Cost_Amount", "sum"),
            profit_net=("profit_line", "sum"),
            transaction_count=("Stt", "count"),
            sale_line_count=("positive_quantity", lambda s: int((s > 0).sum())),
            return_line_count=("is_return", "sum"),
            sign_mismatch_count=("sign_mismatch", "sum"),
            daily_valid_price=("valid_price", "median"),
            daily_valid_unit_cost=("valid_unit_cost", "median"),
        )
        .reset_index()
    )
    daily["net_daily_qty_clip0"] = daily["net_daily_qty"].clip(lower=0)
    daily["has_return"] = (daily["return_line_count"] > 0).astype(np.int8)
    daily["return_ratio_vs_gross"] = daily["return_qty"] / (
        daily["gross_positive_qty"] + config.EPSILON
    )
    return daily.sort_values(["ItemCode", "Date"]).reset_index(drop=True)


def build_sku_profile(
    daily_sparse: pd.DataFrame,
    sample_skus: Iterable[str],
    train_end: str = config.TRAIN_END,
) -> pd.DataFrame:
    """Build SKU-level profile from sparse daily demand."""

    sample_index = pd.Index(sample_skus, name="ItemCode")
    train_end_ts = pd.Timestamp(train_end)
    positive = daily_sparse.loc[daily_sparse["gross_positive_qty"] > 0]
    grouped = daily_sparse.groupby("ItemCode", observed=True)

    profile = pd.DataFrame(index=sample_index)
    profile["first_transaction_date"] = grouped["Date"].min()
    profile["last_transaction_date"] = grouped["Date"].max()
    profile["first_sale_date"] = positive.groupby("ItemCode", observed=True)["Date"].min()
    profile["last_sale_date"] = positive.groupby("ItemCode", observed=True)["Date"].max()
    profile["total_net_qty"] = grouped["net_daily_qty"].sum()
    profile["total_gross_positive_qty"] = grouped["gross_positive_qty"].sum()
    profile["positive_transaction_days"] = positive.groupby("ItemCode", observed=True)["Date"].nunique()
    profile["transaction_days"] = grouped["Date"].nunique()
    profile["sku_age_days"] = (
        train_end_ts - pd.to_datetime(profile["first_transaction_date"])
    ).dt.days.add(1)
    profile["days_since_last_sale"] = (
        train_end_ts - pd.to_datetime(profile["last_sale_date"])
    ).dt.days
    profile["avg_qty_when_positive"] = (
        positive.groupby("ItemCode", observed=True)["gross_positive_qty"].mean()
    )
    profile["median_valid_price"] = (
        daily_sparse.groupby("ItemCode", observed=True)["daily_valid_price"].median()
    )
    profile = profile.reset_index()
    fill_zero = [
        "total_net_qty",
        "total_gross_positive_qty",
        "positive_transaction_days",
        "transaction_days",
    ]
    profile[fill_zero] = profile[fill_zero].fillna(0)
    profile["sku_age_days"] = profile["sku_age_days"].fillna(0)
    profile["days_since_last_sale"] = profile["days_since_last_sale"].fillna(9999)
    return profile


def _date_grid_for_sku(
    sku: str,
    first_date: pd.Timestamp,
    end_date: pd.Timestamp,
    full_start: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    start = full_start if full_start is not None else first_date
    if pd.isna(start):
        start = end_date
    dates = pd.date_range(start=start, end=end_date, freq="D")
    return pd.DataFrame({"Date": dates, "ItemCode": sku})


def build_dense_panel(
    daily_sparse: pd.DataFrame,
    sample_skus: Iterable[str],
    mode: str = "active_span",
    start_date: str = config.TRAIN_START,
    end_date: str = config.TRAIN_END,
) -> pd.DataFrame:
    """Create dense daily panel using active-span or full-span zeros."""

    if mode not in {"active_span", "full_span"}:
        raise ValueError("mode must be 'active_span' or 'full_span'.")
    end_ts = pd.Timestamp(end_date)
    full_start = pd.Timestamp(start_date) if mode == "full_span" else None
    first_dates = daily_sparse.groupby("ItemCode", observed=True)["Date"].min()
    frames = [
        _date_grid_for_sku(sku, first_dates.get(sku, pd.NaT), end_ts, full_start)
        for sku in sample_skus
    ]
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.merge(daily_sparse, on=["Date", "ItemCode"], how="left")

    zero_cols = [
        "net_daily_qty",
        "gross_positive_qty",
        "return_qty",
        "sales_amount_net",
        "cost_amount_net",
        "profit_net",
        "transaction_count",
        "sale_line_count",
        "return_line_count",
        "sign_mismatch_count",
        "net_daily_qty_clip0",
        "has_return",
        "return_ratio_vs_gross",
    ]
    for col in zero_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0)
    return panel.sort_values(["ItemCode", "Date"]).reset_index(drop=True)


def save_preprocessing_artifacts(
    transactions: pd.DataFrame,
    sample: pd.DataFrame,
    run_mode: str,
    sku_subset: Optional[Sequence[str]] = None,
    paths: Optional[config.PipelinePaths] = None,
) -> Dict[str, Path]:
    """Build and save sparse, dense, weight, and profile artifacts."""

    paths = paths or config.PipelinePaths()
    sample_skus = extract_submission_skus(sample)
    if sku_subset is not None:
        keep = set(sku_subset)
        transactions = transactions.loc[transactions["ItemCode"].isin(keep)].copy()
        sample_skus = pd.Index([sku for sku in sample_skus if sku in keep], name="ItemCode")

    weights = compute_profit_weights(transactions, sample_skus)
    daily_sparse = aggregate_daily_sparse(transactions)
    sku_profile = build_sku_profile(daily_sparse, sample_skus)
    active_panel = build_dense_panel(daily_sparse, sample_skus, mode="active_span")
    full_panel = build_dense_panel(daily_sparse, sample_skus, mode="full_span")

    outputs = {
        "daily_sparse": paths.processed_dir / "daily_sparse.parquet",
        "daily_panel_active_span": paths.processed_dir / "daily_panel_active_span.parquet",
        "daily_panel_full_span": paths.processed_dir / "daily_panel_full_span.parquet",
        "sku_profile": paths.processed_dir / "sku_profile.parquet",
        "profit_weights": paths.processed_dir / "profit_weights.parquet",
        "profit_concentration": paths.report_dir / "profit_concentration.csv",
    }
    daily_sparse.to_parquet(outputs["daily_sparse"], index=False)
    active_panel.to_parquet(outputs["daily_panel_active_span"], index=False)
    full_panel.to_parquet(outputs["daily_panel_full_span"], index=False)
    sku_profile.to_parquet(outputs["sku_profile"], index=False)
    weights.to_parquet(outputs["profit_weights"], index=False)
    profit_concentration(weights).to_csv(outputs["profit_concentration"], index=False)

    for key, path in outputs.items():
        write_metadata(path, run_mode, {"artifact_key": key, "rows": _artifact_rows(path)})
    return outputs


def _artifact_rows(path: Path) -> Optional[int]:
    if path.suffix == ".csv":
        return None
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        try:
            return int(pd.read_parquet(path).shape[0])
        except Exception:
            return None
