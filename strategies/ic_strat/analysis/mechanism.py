"""What actually drives the SMA5/SMA9 crossover split, and how stable it is.

A clean-looking mean split between two states can still be an artefact of a
handful of days, a single calendar year, or a proxy for something duller (raw
volatility) than the crossover itself. Every function here exists to rule one
of those out before calling the split "real."
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_state_panel(panel: pd.DataFrame, state_col: str, value_col: str | None = None) -> pd.DataFrame:
    """Collapse the condor panel to one row per session: total P/L and state.

    Args:
        panel: Condor-level panel with ``date``, ``pl`` and ``state_col``.
        value_col: Optional extra column to carry through at its first value
            per day (e.g. the raw ratio, for level-vs-state comparisons).

    Returns:
        One row per session with ``pl`` (summed) and ``state`` (first value).
    """
    agg = {"pl": ("pl", "sum"), "state": (state_col, "first")}
    if value_col:
        agg["value"] = (value_col, "first")
    daily = panel.groupby("date").agg(**agg).dropna(subset=["state"])
    return daily


def yearly_breakdown(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-year, per-state session count/mean/total P/L.

    A split that only holds in some years is a regime effect, not a constant;
    this is the first thing to check before trusting the pooled mean.
    """
    working = daily.copy()
    working["year"] = pd.to_datetime(working.index).year
    return (
        working.groupby(["year", "state"])["pl"]
        .agg(["count", "mean", "sum"])
        .rename(columns={"count": "n_sessions", "mean": "avg_daily_pl", "sum": "total_pl"})
    )


def monthly_edge(daily: pd.DataFrame) -> pd.DataFrame:
    """Month-by-month total P/L for each state and their difference (the "edge").

    Shows whether the yearly pattern is a steady accumulation or the product
    of a few outsized months.
    """
    working = daily.copy()
    working["month"] = pd.to_datetime(working.index).to_period("M")
    monthly = working.groupby(["month", "state"])["pl"].sum().unstack()
    monthly.columns = [f"state_{c}_sum" for c in monthly.columns]
    if monthly.shape[1] == 2:
        monthly["edge"] = monthly.iloc[:, 0] - monthly.iloc[:, 1]
    return monthly


def state_run_lengths(daily: pd.DataFrame) -> pd.DataFrame:
    """How many consecutive sessions each state run lasts, summarized by state.

    Similar run lengths for both states means neither is a rare, brief spike
    driving the pooled average — the states are comparably persistent regimes.
    """
    runs = (daily["state"] != daily["state"].shift()).cumsum()
    run_table = daily.groupby(runs)["state"].agg(["first", "count"])
    return run_table.groupby("first")["count"].describe()


def volatility_correlate(daily_with_value: pd.DataFrame, realized_vol_col: str = "day_range") -> pd.DataFrame:
    """Compare realized intraday range between states — is the crossover a vol proxy?

    Args:
        daily_with_value: Output of :func:`daily_state_panel` with an extra
            realized-volatility-proxy column already attached (e.g. the
            session's cumulative SPX range from the condor panel).
    """
    return daily_with_value.groupby("state")[realized_vol_col].describe()


def outlier_robustness(daily: pd.DataFrame, trim_quantile: float = 0.01) -> pd.DataFrame:
    """Mean vs. median vs. trimmed mean per state — is the split tail-driven?

    Drops the worst ``trim_quantile`` of days *within each state* and reports
    the mean with and without them. If the sign of the split flips once the
    single worst days are removed, the effect is not broad-based.
    """
    rows = []
    for state, group in daily.groupby("state"):
        threshold = group["pl"].quantile(trim_quantile)
        trimmed = group.loc[group["pl"] > threshold, "pl"]
        rows.append({
            "state": state,
            "n": len(group),
            "mean": group["pl"].mean(),
            "median": group["pl"].median(),
            f"trimmed_mean_drop_worst_{trim_quantile:.0%}": trimmed.mean(),
            "pl_from_dropped_days": group["pl"].sum() - trimmed.sum(),
        })
    return pd.DataFrame(rows).set_index("state")


def threshold_scan_in_sample_vs_oos(
    panel: pd.DataFrame,
    column: str,
    thresholds: np.ndarray,
    direction: str = "below",
    date_col: str = "date",
    train_frac: float = 0.70,
    valid_frac: float = 0.15,
) -> pd.DataFrame:
    """Scan a threshold on ``column``, scored separately in-sample and out-of-sample.

    Picking a "best threshold" by scanning a metric against every candidate
    cut and reading off the max is a classic overfitting trap — the max is
    partly signal and partly noise, and the noise part won't repeat. This
    scores every threshold on the **train+valid** sessions (the sessions a
    threshold could legitimately be chosen from) and, separately, on a
    **test** block those sessions never touched (via
    :func:`cascading.time_based_split`), so a threshold whose avg-P/L curve
    looks similar in both columns is a real, checkable pattern rather than a
    single sample's noise.

    Args:
        panel: Condor panel with ``column``, ``pl``, ``win`` and ``date_col``.
        thresholds: Candidate cut values.
        direction: ``"below"`` keeps ``column <= threshold``; ``"above"``
            keeps ``column >= threshold``.
        date_col: Session date column for the time-based split.
        train_frac: Passed through to :func:`cascading.time_based_split`.
        valid_frac: Passed through to :func:`cascading.time_based_split`.

    Returns:
        One row per (threshold, sample) pair: n, avg P/L, win rate, total P/L.
    """
    from .cascading import time_based_split

    splits = time_based_split(panel, date_col=date_col, train_frac=train_frac, valid_frac=valid_frac)
    in_sample = pd.concat([splits.train, splits.valid])
    samples = {"in_sample (train+valid)": in_sample, "out_of_sample (test)": splits.test}

    rows = []
    for sample_name, frame in samples.items():
        for threshold in thresholds:
            selected = (
                frame[frame[column] <= threshold] if direction == "below"
                else frame[frame[column] >= threshold]
            )
            rows.append({
                "sample": sample_name, "threshold": threshold, "n": len(selected),
                "avg_pl": selected["pl"].mean() if len(selected) else np.nan,
                "win_rate": selected["win"].mean() if len(selected) else np.nan,
                "total_pl": selected["pl"].sum() if len(selected) else np.nan,
            })
    return pd.DataFrame(rows)


def weekday_performance(panel: pd.DataFrame) -> pd.DataFrame:
    """Average P/L, win rate, and total P/L by day of week.

    Uses ``dow`` (day name) so the table reads without a lookup; sessions, not
    condors, are what vary across weekdays (holidays, early closes), so
    ``n_sessions`` is reported alongside condor count.
    """
    WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    stats = panel.groupby("dow").agg(
        n_condors=("pl", "size"), n_sessions=("date", "nunique"),
        avg_pl=("pl", "mean"), win_rate=("win", "mean"), total_pl=("pl", "sum"),
    )
    return stats.reindex([d for d in WEEKDAY_ORDER if d in stats.index])


def threshold_scan_full_sample(
    panel: pd.DataFrame,
    column: str,
    thresholds: np.ndarray,
    direction: str = "below",
) -> pd.DataFrame:
    """Scan a threshold on the full sample — no in-sample/held-out split.

    Use :func:`threshold_scan_in_sample_vs_oos` when the point is to check a
    threshold survives unseen data; use this when the point is a direct,
    descriptive read of where a threshold sits given all the data on hand.

    Args:
        panel: Condor panel with ``column``, ``pl`` and ``win``.
        thresholds: Candidate cut values.
        direction: ``"below"`` keeps ``column <= threshold``; ``"above"``
            keeps ``column >= threshold``.

    Returns:
        One row per threshold: n, avg P/L, win rate, total P/L.
    """
    rows = []
    for threshold in thresholds:
        selected = (
            panel[panel[column] <= threshold] if direction == "below"
            else panel[panel[column] >= threshold]
        )
        rows.append({
            "threshold": threshold, "n": len(selected),
            "avg_pl": selected["pl"].mean() if len(selected) else np.nan,
            "win_rate": selected["win"].mean() if len(selected) else np.nan,
            "total_pl": selected["pl"].sum() if len(selected) else np.nan,
        })
    return pd.DataFrame(rows)


def vix_regime_performance(panel: pd.DataFrame, vix_col: str = "vix30_open") -> pd.DataFrame:
    """P/L and win rate by this repo's shared VIX-open decile regime.

    Uses ``src.enrich_trades.cut_vix_regime`` — the same D1-D10 bucketing
    every other strategy in this repo uses (``config/vix_open_deciles.json``),
    so "VIX regime" means the same thing here as it does in dc/bwb_bearish/
    bwb_bullish rather than a one-off bucketing invented for this strategy.
    """
    from src.enrich_trades import cut_vix_regime

    working = panel.copy()
    working["vix_regime"] = cut_vix_regime(working[vix_col])
    stats = working.groupby("vix_regime", observed=True).agg(
        n=("pl", "size"), avg_pl=("pl", "mean"), win_rate=("win", "mean"), total_pl=("pl", "sum"),
    )
    return stats.sort_values("avg_pl", ascending=False)


def tranche_performance_masked(panel: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Per-entry-time performance restricted to rows where ``mask`` is True.

    Shared core behind :func:`tranche_performance_conditional`; call this
    directly when the condition isn't a simple range on one column (e.g. two
    conditions combined).
    """
    band = panel[mask]
    stats = band.groupby("entry_time").agg(
        n=("pl", "size"), avg_pl=("pl", "mean"), win_rate=("win", "mean"),
        total_pl=("pl", "sum"), pl_std=("pl", "std"),
    )
    stats["sharpe_like"] = stats["avg_pl"] / stats["pl_std"]
    stats.attrs["n_sessions"] = int(band["date"].nunique())
    stats.attrs["n_condors"] = int(len(band))
    return stats.sort_values("avg_pl", ascending=False)


def tranche_performance_conditional(
    panel: pd.DataFrame,
    condition_col: str,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    """Per-entry-time performance restricted to sessions in a value band.

    Answers "given the market sits in this band on `condition_col`, which
    tranches are actually worth trading" — a materially different question
    than "which tranches are best overall" (see the full-sample comparison
    this is usually read alongside).

    Args:
        panel: Condor panel with ``entry_time``, ``pl``, ``win`` and
            ``condition_col``.
        lower: Inclusive lower bound on ``condition_col``.
        upper: Inclusive upper bound on ``condition_col``.

    Returns:
        One row per entry-time tranche, sorted by average P/L, descending.
    """
    mask = (panel[condition_col] >= lower) & (panel[condition_col] <= upper)
    stats = tranche_performance_masked(panel, mask)
    stats.attrs["band"] = (lower, upper)
    return stats.sort_values("avg_pl", ascending=False)
