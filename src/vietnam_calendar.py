"""Leakage-safe Vietnam calendar features for daily demand forecasting.

The holiday table is intentionally hardcoded for the train/CV years used in
this competition. Tet dates are Lunar New Year day 1. Observed holiday windows
include common public-sector closure windows where known; they are calendar
facts, not future sales information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HolidayWindow:
    """Named inclusive holiday window."""

    name: str
    start: str
    end: str
    is_tet: bool = False
    is_long: bool = False


TET_DAY_1 = {
    2020: "2020-01-25",
    2021: "2021-02-12",
    2022: "2022-02-01",
    2023: "2023-01-22",
    2024: "2024-02-10",
    2025: "2025-01-29",
    2026: "2026-02-17",
}


HOLIDAY_WINDOWS = [
    HolidayWindow("new_year_2021", "2021-01-01", "2021-01-01"),
    HolidayWindow("tet_2021", "2021-02-10", "2021-02-16", is_tet=True, is_long=True),
    HolidayWindow("hung_kings_2021", "2021-04-21", "2021-04-21"),
    HolidayWindow("reunification_labor_2021", "2021-04-30", "2021-05-03", is_long=True),
    HolidayWindow("national_day_2021", "2021-09-02", "2021-09-03", is_long=True),
    HolidayWindow("new_year_2022", "2022-01-01", "2022-01-03", is_long=True),
    HolidayWindow("tet_2022", "2022-01-29", "2022-02-06", is_tet=True, is_long=True),
    HolidayWindow("hung_kings_2022", "2022-04-10", "2022-04-11", is_long=True),
    HolidayWindow("reunification_labor_2022", "2022-04-30", "2022-05-03", is_long=True),
    HolidayWindow("national_day_2022", "2022-09-01", "2022-09-04", is_long=True),
    HolidayWindow("new_year_2023", "2023-01-01", "2023-01-02", is_long=True),
    HolidayWindow("tet_2023", "2023-01-20", "2023-01-26", is_tet=True, is_long=True),
    HolidayWindow("hung_reunification_labor_2023", "2023-04-29", "2023-05-03", is_long=True),
    HolidayWindow("national_day_2023", "2023-09-01", "2023-09-04", is_long=True),
    HolidayWindow("new_year_2024", "2024-01-01", "2024-01-01"),
    HolidayWindow("tet_2024", "2024-02-08", "2024-02-14", is_tet=True, is_long=True),
    HolidayWindow("hung_kings_2024", "2024-04-18", "2024-04-18"),
    HolidayWindow("reunification_labor_2024", "2024-04-30", "2024-05-01", is_long=True),
    HolidayWindow("national_day_2024", "2024-08-31", "2024-09-03", is_long=True),
    HolidayWindow("new_year_2025", "2025-01-01", "2025-01-01"),
    HolidayWindow("tet_2025", "2025-01-25", "2025-02-02", is_tet=True, is_long=True),
    HolidayWindow("hung_kings_2025", "2025-04-07", "2025-04-07"),
    HolidayWindow("reunification_labor_2025", "2025-04-30", "2025-05-01", is_long=True),
    HolidayWindow("national_day_2025", "2025-08-30", "2025-09-02", is_long=True),
]


def _date_set_from_windows(windows: Iterable[HolidayWindow], attr: str | None = None) -> set[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    for window in windows:
        if attr is not None and not bool(getattr(window, attr)):
            continue
        dates.update(pd.date_range(window.start, window.end, freq="D"))
    return dates


PUBLIC_HOLIDAY_DATES = _date_set_from_windows(HOLIDAY_WINDOWS)
TET_PERIOD_DATES = _date_set_from_windows(HOLIDAY_WINDOWS, "is_tet")
LONG_HOLIDAY_DATES = _date_set_from_windows(HOLIDAY_WINDOWS, "is_long")


def tet_reference_frame() -> pd.DataFrame:
    """Return Tet day-1 dates used by the feature generator."""

    return pd.DataFrame(
        {
            "year": list(TET_DAY_1),
            "tet_day_1": [pd.Timestamp(value) for value in TET_DAY_1.values()],
        }
    ).sort_values("year")


def holiday_window_frame() -> pd.DataFrame:
    """Return the hardcoded observed holiday windows."""

    return pd.DataFrame(
        [
            {
                "name": window.name,
                "start": pd.Timestamp(window.start),
                "end": pd.Timestamp(window.end),
                "is_tet": window.is_tet,
                "is_long": window.is_long,
            }
            for window in HOLIDAY_WINDOWS
        ]
    )


def _days_to_next_tet(date: pd.Timestamp) -> float:
    future = [pd.Timestamp(value) for value in TET_DAY_1.values() if pd.Timestamp(value) >= date]
    if not future:
        return np.nan
    return float((min(future) - date).days)


def _days_after_previous_tet(date: pd.Timestamp) -> float:
    previous = [pd.Timestamp(value) for value in TET_DAY_1.values() if pd.Timestamp(value) <= date]
    if not previous:
        return np.nan
    return float((date - max(previous)).days)


def add_vietnam_calendar_features(
    frame: pd.DataFrame,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Add Vietnam calendar features based only on the row date."""

    out = frame.copy()
    dates = pd.to_datetime(out[date_col])
    normalized = dates.dt.normalize()
    public_holidays = pd.Index(PUBLIC_HOLIDAY_DATES)
    tet_period = pd.Index(TET_PERIOD_DATES)
    long_holidays = pd.Index(LONG_HOLIDAY_DATES)

    out["is_sunday"] = dates.dt.dayofweek.eq(6).astype(np.int8)
    out["is_public_holiday_vn"] = normalized.isin(public_holidays).astype(np.int8)
    out["is_tet_period"] = normalized.isin(tet_period).astype(np.int8)
    out["is_long_holiday_window"] = normalized.isin(long_holidays).astype(np.int8)
    out["days_to_tet"] = normalized.map(_days_to_next_tet).astype(float)
    out["days_after_tet"] = normalized.map(_days_after_previous_tet).astype(float)
    out["is_pre_tet_7"] = out["days_to_tet"].between(1, 7).astype(np.int8)
    out["is_pre_tet_14"] = out["days_to_tet"].between(1, 14).astype(np.int8)
    out["is_pre_tet_28"] = out["days_to_tet"].between(1, 28).astype(np.int8)
    out["is_post_tet_7"] = out["days_after_tet"].between(1, 7).astype(np.int8)
    out["is_post_tet_14"] = out["days_after_tet"].between(1, 14).astype(np.int8)
    out["is_post_tet_28"] = out["days_after_tet"].between(1, 28).astype(np.int8)
    out["is_month_start"] = dates.dt.is_month_start.astype(np.int8)
    out["is_month_end"] = dates.dt.is_month_end.astype(np.int8)
    out["day_of_week"] = dates.dt.dayofweek.astype(np.int16)
    out["week_of_year"] = dates.dt.isocalendar().week.astype(np.int16)
    out["month"] = dates.dt.month.astype(np.int16)
    return out


def vietnam_calendar_for_dates(dates: Iterable[pd.Timestamp | str]) -> pd.DataFrame:
    """Build a calendar feature frame for arbitrary dates."""

    frame = pd.DataFrame({"Date": pd.to_datetime(list(dates))})
    return add_vietnam_calendar_features(frame)
