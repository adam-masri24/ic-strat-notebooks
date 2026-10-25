from __future__ import annotations

import numpy as np
import pandas as pd

from .vix_deciles import load_vix_decile_scheme


# VIX regimes are the deciles of VIX open over a reference window, loaded from
# config/vix_open_deciles.json. See src/vix_deciles.py for the two adjustments
# made to the file's edges: open-ended tails, and right-closed buckets.
_VIX_DECILES = load_vix_decile_scheme()
VIX_BINS = _VIX_DECILES.bins
VIX_LABELS = _VIX_DECILES.labels
# Right-closed, matching the decile file: a VIX open of exactly 13.467 is D1.
VIX_BINS_RIGHT_CLOSED = True
WEEKDAY_LABELS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

SLR_BINS = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, np.inf]
SLR_LABELS = [
    "0.00-0.30",
    "0.30-0.40",
    "0.40-0.50",
    "0.50-0.60",
    "0.60-0.70",
    "0.70-0.80",
    "0.80-0.90",
    "0.90-1.00",
    "1.00+",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while treating zero denominators as missing.

    Args:
        numerator: Values to divide.
        denominator: Values to divide by.

    Returns:
        The element-wise division result.
    """
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator


def cut_vix_regime(values: pd.Series) -> pd.Series:
    """Assign each VIX level to its decile regime.

    Args:
        values: VIX levels to bucket.

    Returns:
        An ordered categorical of regime labels. The outer bins are unbounded,
        so a level outside the decile file's reference range lands in D1 or the
        top decile instead of becoming null.
    """
    return pd.cut(
        values,
        bins=VIX_BINS,
        labels=VIX_LABELS,
        right=VIX_BINS_RIGHT_CLOSED,
        include_lowest=True,
    )


def enrich_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Add calculated fields and research buckets to the combined trade log."""
    enriched = df.copy()

    # P/L outcomes and diagnostics.
    enriched["win"] = (enriched["p_l"] > 0).astype(int)
    enriched["loss"] = (enriched["p_l"] < 0).astype(int)
    enriched["flat"] = (enriched["p_l"] == 0).astype(int)
    enriched["outcome"] = np.select(
        [enriched["p_l"] > 0, enriched["p_l"] < 0],
        ["win", "loss"],
        default="flat",
    )
    enriched["return_on_margin"] = _safe_divide(enriched["p_l"], enriched["margin_req"])
    enriched["opening_slr"] = enriched["opening_short_long_ratio"]
    enriched["closing_slr"] = enriched["closing_short_long_ratio"]
    enriched["vix_change"] = enriched["closing_vix"] - enriched["opening_vix"]
    enriched["slr_change"] = (
        enriched["closing_short_long_ratio"] - enriched["opening_short_long_ratio"]
    )
    enriched["slr_change_pct"] = _safe_divide(
        enriched["slr_change"], enriched["opening_short_long_ratio"]
    )
    enriched["slr_close_open_change"] = (
        _safe_divide(
            enriched["closing_short_long_ratio"],
            enriched["opening_short_long_ratio"],
        )
        - 1
    )

    # Entry-side research buckets.
    enriched["vix_regime"] = cut_vix_regime(enriched["opening_vix"])
    # Same edges on the exit side, so a trade's open and close regimes are
    # directly comparable rather than being cut on different scales. The edges
    # come from VIX *open*, so a closing regime reads as "where this close sits
    # on the opening-VIX scale" rather than as its own decile.
    enriched["closing_vix_regime"] = cut_vix_regime(enriched["closing_vix"])

    
    enriched["opening_slr_bucket"] = pd.cut(
        enriched["opening_short_long_ratio"],
        bins=SLR_BINS,
        labels=SLR_LABELS,
        right=False,
        include_lowest=True,
    )
    enriched["slr_bucket"] = enriched["opening_slr_bucket"]

    if "time_opened_label" not in enriched.columns:
        enriched["time_opened_label"] = enriched["time_opened"]

    enriched["entry_hour"] = enriched["time_opened_label"].str.slice(0, 2)
    enriched["weekday"] = pd.Categorical(
        pd.to_datetime(enriched["date_opened"], errors="coerce").dt.day_name(),
        categories=WEEKDAY_LABELS,
        ordered=True,
    )

    return enriched
