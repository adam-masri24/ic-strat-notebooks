"""Attach VIX term-structure features to the IC condor panel.

Market data source: ``strategies/ic_strat/term_structure_daily.csv`` — a
session-level VIX1D/VIX9D/VIX30 file extracted from an external export (see
this strategy's README for provenance). It is **not** the repo's canonical
``data/market/vix/`` pipeline (``src/vix_term_structure.py``), which has no
local parquet in this checkout; kept scoped to this strategy until verified
against that pipeline.

Everything here is at **session** (daily) resolution — open, close, and
prior-close snapshots, plus three intraday snapshots (10:38 / 12:38 / 14:38).
There is no per-minute "live" curve, so a tranche's term-structure feature is
its day's value, not a value interpolated to its own entry minute.

Leakage discipline (matches the bwb_bearish precedent in this repo): only
``*_open``, ``*_close_prev`` and intraday-snapshot-before-entry columns are
attached by default. ``*_close`` is same-day, end-of-session information and
is kept out of the merged panel unless requested explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import VIX_TERM_STRUCTURE_DAILY_FILE

from .ic_analysis import require_file

TERM_STRUCTURE_CSV = VIX_TERM_STRUCTURE_DAILY_FILE

# The ratio names used throughout this strategy's analyses, in the vocabulary
# the trade files already use: ts_<near>_<far>.
TS_1D_9D = "ts_1d_9d_open"
TS_9D_30 = "ts_9d_30_open"
TS_1D_30 = "ts_1d_30_open"

LEAKY_SUFFIXES = ("_close",)  # same-day close; _close_prev is fine, kept separately


def load_term_structure_daily(path: str | Path = TERM_STRUCTURE_CSV) -> pd.DataFrame:
    """Load the session-level VIX1D/VIX9D/VIX30 term-structure file."""
    df = pd.read_csv(require_file(Path(path), "VIX term-structure file"), parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def attach_term_structure(
    ic: pd.DataFrame,
    term_structure: pd.DataFrame | None = None,
    include_same_day_close: bool = False,
) -> pd.DataFrame:
    """Left-join session-level term-structure features onto the condor panel.

    Args:
        ic: Condor panel with a ``date`` column.
        term_structure: Result of :func:`load_term_structure_daily`; loaded
            from the default path when omitted.
        include_same_day_close: Keep ``*_close`` (same-session) columns. Off
            by default — those aren't known at any tranche's entry.

    Returns:
        ``ic`` with term-structure columns attached, one row per condor.
    """
    if term_structure is None:
        term_structure = load_term_structure_daily()

    cols = list(term_structure.columns)
    if not include_same_day_close:
        cols = [
            c for c in cols
            if c == "date" or not (c.endswith("_close") and "_close_prev" not in c)
        ]
    merged = ic.merge(term_structure[cols], on="date", how="left")
    return merged


def add_moving_averages(
    term_structure: pd.DataFrame,
    column: str = TS_9D_30,
    windows: tuple[int, ...] = (5, 9),
) -> pd.DataFrame:
    """Add rolling SMA/EMA of a term-structure (or VIX) column, at session level.

    Adds ``sma{w}_<column>`` and ``ema{w}_<column>`` for each window, plus the
    two-window crossover flag ``<column>_sma{w0}_gt_sma{w1}`` for the first two
    windows given (fast > slow). Computed on the daily series *before* any join
    to the condor panel, so the average is over trading sessions, not tranches.
    """
    out = term_structure.sort_values("date").copy()
    for window in windows:
        out[f"sma{window}_{column}"] = out[column].rolling(window, min_periods=window).mean()
        out[f"ema{window}_{column}"] = out[column].ewm(span=window, min_periods=window).mean()

    if len(windows) >= 2:
        fast, slow = windows[0], windows[1]
        out[f"{column}_sma{fast}_gt_sma{slow}"] = (
            out[f"sma{fast}_{column}"] > out[f"sma{slow}_{column}"]
        )
        out[f"{column}_ema{fast}_gt_ema{slow}"] = (
            out[f"ema{fast}_{column}"] > out[f"ema{slow}_{column}"]
        )
    return out


def day_frequency_by_threshold(
    term_structure: pd.DataFrame,
    column: str = TS_9D_30,
    thresholds: np.ndarray | None = None,
    direction: str = "above",
    daily_pl: pd.Series | None = None,
) -> pd.DataFrame:
    """Count how many sessions clear a threshold on ``column``, threshold by threshold.

    This answers "how does day frequency change with the {column} threshold":
    as the bar is raised, fewer sessions qualify. Reported both as a session
    count and as a share of all sessions with a non-null value, so a thin tail
    doesn't read as a strong filter just because the count looks non-trivial.

    Args:
        term_structure: Daily term-structure frame (post :func:`add_moving_averages`
            if you want to sweep a smoothed series instead of the raw ratio).
        column: Column to threshold.
        thresholds: Threshold grid; defaults to 25 points spanning the column's
            5th-95th percentile.
        direction: ``"above"`` counts sessions with ``column >= threshold``;
            ``"below"`` counts ``column <= threshold``.
        daily_pl: Optional total condor P/L per session (index or a ``date``
            column matching ``term_structure["date"]``). When given, each row
            also reports the **average daily P/L** across qualifying
            sessions — not per-condor P/L, but per trading day, since that is
            the unit a threshold rule actually gates.

    Returns:
        One row per threshold: the threshold, qualifying session count, share
        of valid sessions, the column's total valid-session count, and (when
        ``daily_pl`` is given) ``avg_daily_pl`` and ``total_pl`` for the
        qualifying sessions.
    """
    working = term_structure[["date", column]].dropna(subset=[column]).copy()
    if daily_pl is not None:
        pl_series = daily_pl.set_index("date")["pl"] if isinstance(daily_pl, pd.DataFrame) else daily_pl
        working["pl"] = working["date"].map(pl_series)

    series = working[column]
    if thresholds is None:
        lo, hi = series.quantile([0.05, 0.95])
        thresholds = np.linspace(lo, hi, 25)

    total_valid = len(series)
    rows = []
    for threshold in thresholds:
        if direction == "above":
            mask = series >= threshold
        elif direction == "below":
            mask = series <= threshold
        else:
            raise ValueError(f"Unknown direction {direction!r}; expected 'above' or 'below'.")
        qualifying = int(mask.sum())
        row = {
            "threshold": threshold,
            "column": column,
            "direction": direction,
            "qualifying_sessions": qualifying,
            "share_of_valid_sessions": qualifying / total_valid if total_valid else np.nan,
            "total_valid_sessions": total_valid,
        }
        if daily_pl is not None:
            qualifying_pl = working.loc[mask, "pl"]
            row["avg_daily_pl"] = qualifying_pl.mean() if qualifying else np.nan
            row["total_pl"] = qualifying_pl.sum() if qualifying else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
