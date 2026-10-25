"""Orchestrates one IC 20-tranche 10-spread study run end to end.

Load raw spreads -> pair into condors -> attach intraday path + term-structure
features -> write the correlation / partial-correlation / day-frequency /
SMA-crossover / VIX-regime tables and figures.

The curated feature set (``CURATED_FEATS``) is deliberately short: this
strategy's term-structure ratios are collinear by construction (they share
VIX1D/VIX9D/VIX30), and every table and chart here is built to make *that*
relationship legible, not to bury it inside a wall of unrelated columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .ic_analysis import add_intraday_path_features, build_iron_condors, daily_close_series, load_ic_trades
from .mechanism import (
    daily_state_panel,
    monthly_edge,
    outlier_robustness,
    state_run_lengths,
    vix_regime_performance,
    volatility_correlate,
    yearly_breakdown,
)
from .partial_correlation import correlation_vs_partial, partial_corr_matrix
from .term_structure_features import (
    TS_9D_30,
    add_moving_averages,
    attach_term_structure,
    day_frequency_by_threshold,
    load_term_structure_daily,
)

STUDY_NAME = "SPX-IC-20-Tranche-10-Study"

# The traded band on ts_9d_30_open; regime_study.TS_LOW/TS_HIGH must match.
TS_9D_30_THRESHOLD = 0.87
TS_9D_30_UPPER = 1.02

CURATED_FEATS = [
    "mins_from_open", "dow_code",
    "spx_move_pct", "range_so_far", "dd_from_hi",
    "credit", "skew_credit", "otm_asym", "slratio_call", "slratio_put",
    "vix30_open", "vix9d_open", "vix1d_open",
    "ts_9d_30_open", "ts_1d_9d_open", "ts_1d_9d_rank252_prev",
]
# Primary SMA5/SMA9 signal: SPX PRICE momentum, lagged one session so a
# morning tranche only ever sees a fully-formed prior close (its own day's
# close is not knowable at 09:32). This is the one that matches an
# Options-Omega-style price moving-average filter and the direction a live
# backtest there would show.
PRICE_CROSSOVER_COL = "close_sma5_gt_sma9_lagged"
# Secondary signal: SMA5/SMA9 on the ts_9d_30_open VIX term-structure ratio
# itself. A real, checked effect (see the mechanism deep dive), but a
# different indicator from PRICE_CROSSOVER_COL — don't conflate the two.
TS_CROSSOVER_COL = f"{TS_9D_30}_sma5_gt_sma9"


@dataclass(frozen=True)
class StudyRunResult:
    """Summary of one completed IC study run."""

    study_name: str
    output_dir: Path
    condor_count: int
    table_count: int
    figure_count: int


def build_panel(
    trades_csv: str | Path,
    term_structure_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Run the full load -> pair -> feature-attach chain, once.

    Returns:
        One row per condor, with path features, term-structure features
        (including SMA5/SMA9 crossover flags on ``ts_9d_30_open``), and the
        SPX price SMA5/SMA9 crossover — both a same-day version (uses that
        day's own close; kept only to show why it overstates the effect) and
        the properly lagged, actually-tradeable version.
    """
    spreads = load_ic_trades(trades_csv)
    ic = build_iron_condors(spreads)
    ic = add_intraday_path_features(ic)

    ts_kwargs = {} if term_structure_csv is None else {"path": term_structure_csv}
    term_structure = load_term_structure_daily(**ts_kwargs)
    term_structure = add_moving_averages(term_structure, column=TS_9D_30, windows=(5, 9))

    panel = attach_term_structure(ic, term_structure)
    panel["dow_code"] = panel["date"].dt.dayofweek

    daily_close = daily_close_series(ic)
    price_ma = add_moving_averages(daily_close, column="close", windows=(5, 9))
    # .shift() upcasts a bool column to object (NaN for the first row), which
    # silently breaks `~` (bitwise, not logical, not on plain Python bools) —
    # the nullable "boolean" dtype keeps `~`/masking correct downstream.
    price_ma[PRICE_CROSSOVER_COL] = price_ma["close_sma5_gt_sma9"].shift(1).astype("boolean")
    panel = panel.merge(
        price_ma[["date", "close", "sma5_close", "sma9_close",
                   "close_sma5_gt_sma9", PRICE_CROSSOVER_COL]],
        on="date", how="left",
    )
    return panel


def _save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _correlation_tables(panel: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> None:
    cols = CURATED_FEATS + ["pl"]
    corr = panel[cols].dropna().corr(method="spearman")
    corr.to_csv(tables_dir / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 6}, square=True, ax=ax)
    ax.set_title("Spearman correlation — curated features vs P/L")
    _save_fig(fig, figures_dir / "correlation_heatmap.png")

    partial = partial_corr_matrix(panel, cols)
    partial.to_csv(tables_dir / "partial_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(partial, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 6}, square=True, ax=ax)
    ax.set_title("Partial correlation — each pair controlling for the rest")
    _save_fig(fig, figures_dir / "partial_correlation_heatmap.png")

    comparison = correlation_vs_partial(panel, cols, target="pl")
    comparison.to_csv(tables_dir / "correlation_vs_partial_correlation.csv")


def _day_frequency_table(panel: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    daily_ts = panel.drop_duplicates("date")[["date", TS_9D_30]]
    daily_pl = panel.groupby("date")["pl"].sum()
    freq = day_frequency_by_threshold(daily_ts, column=TS_9D_30, direction="above", daily_pl=daily_pl)
    freq.to_csv(tables_dir / "day_frequency_by_ts_9d_30_threshold.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(freq["threshold"], freq["qualifying_sessions"], marker="o", ms=3, lw=1.4)
    ax.set_xlabel("ts_9d_30_open threshold (qualify if ratio >= threshold)")
    ax.set_ylabel("qualifying sessions")
    ax.set_title("Trading-day frequency vs. ts_9d_30 threshold")
    ax2 = ax.twinx()
    ax2.plot(freq["threshold"], freq["share_of_valid_sessions"] * 100, color="grey", lw=0.8, ls="--")
    ax2.set_ylabel("share of sessions (%)", color="grey")

    ax = axes[1]
    colors = np.where(freq["avg_daily_pl"] >= 0, "#2a78d6", "#e34948")
    ax.bar(freq["threshold"], freq["avg_daily_pl"], width=(freq["threshold"].diff().median() or 0.01) * 0.8,
           color=colors)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(TS_9D_30_THRESHOLD, TS_9D_30_UPPER, color="#2a78d6", alpha=0.08,
               label=f"traded band {TS_9D_30_THRESHOLD}-{TS_9D_30_UPPER}")
    ax.set_xlabel("ts_9d_30_open threshold (qualify if ratio >= threshold)")
    ax.set_ylabel("avg daily P/L, qualifying sessions ($)")
    ax.set_title("Average P/L per day vs. ts_9d_30 threshold")
    ax.legend(fontsize=8)
    _save_fig(fig, figures_dir / "day_frequency_by_ts_9d_30_threshold.png")
    return freq


def _significance_by_state(daily_pl: pd.DataFrame) -> pd.DataFrame:
    """Session-level Welch/Mann-Whitney test between a daily True/False state.

    Condors, not sessions, are the row unit in the panel, but 20 same-day
    condors share one tape — a per-condor test overstates the sample size by
    roughly 20x. This mirrors the ICC discipline bwb_bearish applies to the
    same kind of clustered panel.
    """
    from scipy import stats

    in_state = daily_pl.loc[daily_pl.state, "pl"]
    out_state = daily_pl.loc[~daily_pl.state, "pl"]
    t_stat, t_p = stats.ttest_ind(in_state, out_state, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(in_state, out_state, alternative="two-sided")
    return pd.DataFrame([{
        "n_days_true": len(in_state), "n_days_false": len(out_state),
        "mean_daily_pl_true": in_state.mean(), "mean_daily_pl_false": out_state.mean(),
        "welch_t_stat": t_stat, "welch_p_value": t_p,
        "mannwhitney_u_stat": u_stat, "mannwhitney_p_value": u_p,
    }])


def _price_sma_crossover_analysis(panel: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    """SPX price SMA5/SMA9 crossover — the primary momentum signal.

    Uses the **lagged** state (``PRICE_CROSSOVER_COL``): today's trades only
    ever see a fully-formed prior session's SMA, matching a live filter (and
    an Options-Omega-style backtest). A same-day comparison is included only
    to show how much a look-ahead version overstates the effect.
    """
    daily = panel.drop_duplicates("date").sort_values("date")

    by_state = (
        panel.dropna(subset=[PRICE_CROSSOVER_COL])
        .groupby(PRICE_CROSSOVER_COL, observed=True)
        .agg(n_condors=("pl", "size"), avg_pl=("pl", "mean"), win_rate=("win", "mean"),
             total_pl=("pl", "sum"))
        .rename(index={True: "sma5 > sma9 (lagged)", False: "sma5 <= sma9 (lagged)"})
    )
    by_state.to_csv(tables_dir / "price_sma_crossover_summary.csv")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                              gridspec_kw={"height_ratios": [1.5, 1]})
    ax = axes[0]
    ax.plot(daily["date"], daily["close"], color="#999", lw=0.6, alpha=0.7, label="SPX close (daily)")
    ax.plot(daily["date"], daily["sma5_close"], color="#2a78d6", lw=1.1, label="SMA5")
    ax.plot(daily["date"], daily["sma9_close"], color="#e34948", lw=1.1, label="SMA9")
    ax.fill_between(daily["date"], daily["sma5_close"], daily["sma9_close"],
                     where=(daily["sma5_close"] > daily["sma9_close"]), color="#2a78d6", alpha=0.12,
                     label="SMA5 > SMA9")
    ax.set_ylabel("SPX close")
    ax.set_title("SPX price: SMA5 vs SMA9 crossover")
    ax.legend(fontsize=8, ncol=4)

    ax = axes[1]
    daily_pl = panel.dropna(subset=[PRICE_CROSSOVER_COL]).groupby("date").agg(
        pl=("pl", "sum"), state=(PRICE_CROSSOVER_COL, "first")
    )
    cum_in = daily_pl.loc[daily_pl.state, "pl"].reindex(daily_pl.index).fillna(0).cumsum()
    cum_out = daily_pl.loc[~daily_pl.state, "pl"].reindex(daily_pl.index).fillna(0).cumsum()
    ax.plot(daily_pl.index, cum_in.values, color="#2a78d6", lw=1.2, label="equity: SMA5 > SMA9 days")
    ax.plot(daily_pl.index, cum_out.values, color="#e34948", lw=1.2, label="equity: SMA5 <= SMA9 days")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("cumulative $ (attributed by day state)")
    ax.legend(fontsize=8)
    _save_fig(fig, figures_dir / "price_sma_crossover.png")

    significance = _significance_by_state(daily_pl)
    significance.to_csv(tables_dir / "price_sma_crossover_day_level_significance.csv", index=False)

    # Same-day (look-ahead) comparison, for the record only.
    same_day = (
        panel.dropna(subset=["close_sma5_gt_sma9"])
        .groupby("close_sma5_gt_sma9", observed=True)
        .agg(n_condors=("pl", "size"), avg_pl=("pl", "mean"), win_rate=("win", "mean"), total_pl=("pl", "sum"))
    )
    same_day.to_csv(tables_dir / "price_sma_crossover_same_day_lookahead_comparison.csv")
    return by_state


def _ts_ratio_sma_crossover_analysis(panel: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    """SMA5/SMA9 on the ts_9d_30_open VIX term-structure ratio (secondary signal).

    A separate, real effect from :func:`_price_sma_crossover_analysis` — do
    not conflate the two "SMA5/SMA9" results.
    """
    daily = panel.drop_duplicates("date").sort_values("date")
    fast_col, slow_col = f"sma5_{TS_9D_30}", f"sma9_{TS_9D_30}"

    by_state = (
        panel.dropna(subset=[TS_CROSSOVER_COL])
        .groupby(TS_CROSSOVER_COL, observed=True)
        .agg(n_condors=("pl", "size"), avg_pl=("pl", "mean"), win_rate=("win", "mean"),
             total_pl=("pl", "sum"))
        .rename(index={True: "sma5 > sma9", False: "sma5 <= sma9"})
    )
    by_state.to_csv(tables_dir / "ts_ratio_sma_crossover_summary.csv")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily["date"], daily[TS_9D_30], color="#999", lw=0.6, alpha=0.7, label="ts_9d_30 (daily)")
    ax.plot(daily["date"], daily[fast_col], color="#2a78d6", lw=1.1, label="SMA5")
    ax.plot(daily["date"], daily[slow_col], color="#e34948", lw=1.1, label="SMA9")
    ax.fill_between(daily["date"], daily[fast_col], daily[slow_col],
                     where=(daily[fast_col] > daily[slow_col]), color="#2a78d6", alpha=0.12,
                     label="SMA5 > SMA9")
    ax.set_ylabel("VIX9D / VIX30")
    ax.set_title("VIX9D/VIX30 term-structure ratio: SMA5 vs SMA9 crossover (secondary signal)")
    ax.legend(fontsize=8, ncol=4)
    _save_fig(fig, figures_dir / "ts_ratio_sma_crossover.png")

    daily_pl = panel.dropna(subset=[TS_CROSSOVER_COL]).groupby("date").agg(
        pl=("pl", "sum"), state=(TS_CROSSOVER_COL, "first")
    )
    significance = _significance_by_state(daily_pl)
    significance.to_csv(tables_dir / "ts_ratio_sma_crossover_day_level_significance.csv", index=False)
    return by_state


def _mechanism_deep_dive(panel: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> None:
    """Is the price SMA5/SMA9 split real, stable, and what does it proxy for.

    Checks the **primary** (price) crossover year by year, for outlier-
    dependence, and against a realized-volatility proxy. Tranches are covered
    inside the traded regime by ``regime_study``.
    """
    daily = daily_state_panel(panel, PRICE_CROSSOVER_COL)
    day_range = panel.groupby("date")["range_so_far"].max()
    daily["day_range"] = day_range.reindex(daily.index)

    yearly = yearly_breakdown(daily)
    yearly.to_csv(tables_dir / "price_sma_crossover_by_year.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = yearly["total_pl"].unstack("state").rename(columns={True: "SMA5 > SMA9", False: "SMA5 <= SMA9"})
    pivot.plot.bar(ax=ax, color=["#e34948", "#2a78d6"])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("total $ P/L")
    ax.set_title("Price SMA5/SMA9 crossover split, by calendar year")
    _save_fig(fig, figures_dir / "price_sma_crossover_by_year.png")

    monthly = monthly_edge(daily)
    monthly.to_csv(tables_dir / "price_sma_crossover_monthly_edge.csv")

    runs = state_run_lengths(daily)
    runs.to_csv(tables_dir / "price_sma_crossover_run_lengths.csv")

    vol = volatility_correlate(daily, realized_vol_col="day_range")
    vol.to_csv(tables_dir / "price_sma_crossover_vs_realized_range.csv")

    robustness = outlier_robustness(daily)
    robustness.to_csv(tables_dir / "price_sma_crossover_outlier_robustness.csv")


def _vix_regime_table(panel: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    """P/L by this repo's shared VIX-open decile regime (config/vix_open_deciles.json)."""
    regime = vix_regime_performance(panel, vix_col="vix30_open")
    regime.to_csv(tables_dir / "vix_regime_performance.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = np.where(regime["avg_pl"] >= 0, "#2a78d6", "#e34948")
    ax.bar(regime.index.astype(str), regime["avg_pl"], color=colors)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(regime.index)))
    ax.set_xticklabels(regime.index.astype(str), rotation=45, ha="right")
    ax.set_ylabel("avg P/L per condor ($)")
    ax.set_title("P/L by VIX-open regime (vix30_open, this repo's shared deciles)")
    _save_fig(fig, figures_dir / "vix_regime_performance.png")
    return regime


def run_study_pipeline(
    trades_csv: str | Path,
    reports_root: Path,
    term_structure_csv: str | Path | None = None,
) -> StudyRunResult:
    """Load, analyze, and write every table/figure for one IC study run."""
    output_dir = reports_root / STUDY_NAME
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    panel = build_panel(trades_csv, term_structure_csv)
    panel.to_csv(tables_dir / "condor_panel.csv", index=False)

    _correlation_tables(panel, tables_dir, figures_dir)
    _day_frequency_table(panel, tables_dir, figures_dir)
    _price_sma_crossover_analysis(panel, tables_dir, figures_dir)
    _ts_ratio_sma_crossover_analysis(panel, tables_dir, figures_dir)
    _mechanism_deep_dive(panel, tables_dir, figures_dir)
    _vix_regime_table(panel, tables_dir, figures_dir)

    return StudyRunResult(
        study_name=STUDY_NAME,
        output_dir=output_dir,
        condor_count=len(panel),
        table_count=len(list(tables_dir.glob("*.csv"))),
        figure_count=len(list(figures_dir.glob("*.png"))),
    )
