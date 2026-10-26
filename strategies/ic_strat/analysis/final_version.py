"""The strategy as traded, and why each of its filters is there.

The strategy (``STRATEGY``), measured the way the backtester measures it (``backtester_rules``):

* ``0.87 <= VIX9D / VIX <= 1.02`` at each trade's entry minute,
* SPX intraday movement since the 09:30 open at most 0%,
* SPX at entry above its 10-day SMA (prior 9 closes and the entry price; the 20-day before 2026-10-18),
* the short/long ratio of the whole condor at most 4,
* no Thursdays; seven entry times (11:33, 11:53, 12:13, 13:13, 13:53, 14:13, 14:53),
* every condor held to expiry: no stop (a 400% call stop until 2026-10-18; ``stop_scan`` shows why it went).

The baseline is the raw export: every condor on every session, held to expiry
(column ``pl``), which is also the strategy's P/L. ``pl_stop`` is the same condor with a 400% call stop.

The checks, each on its own table (and at most one figure):

0. **profiles** — how P/L, each side's P/L and the move after entry vary across the
   term structure, the move since the open and SPX's distance from its SMA10, with
   that filter off and the rest on: why each filter works;
1. **build-up** — the filters added one at a time, and each one removed (leave-one-out);
2. **sweeps** — each threshold moved with the rest fixed;
3. **trend rule** — the SMA10 rule against other SPX trend rules (no EMAs), and SMA10 vs SMA20;
4. **calendar** — P/L per weekday and per tranche inside the other filters, tranche
   correlation, and forward selection of the tranches with a walk-forward check;
5. **exit** — the call-stop level at $0 and $20 slippage;
6. **short/long ratio** — its correlation with P/L, P/L by quintile, and a maximum ratio;
7. **remaining features and ranking** — every other candidate as a further filter,
   against 200 noise features put through the same search, then every feature
   ranked on one Sharpe scale.

Every version is scored on **all 998 sessions**, skipped days as zero, and in
2022-2024 and 2025-2026 separately (a stability check, not a train/test split).
Sharpe is the backtester's: daily P/L over $200k less a 4% risk-free rate.
Every choice is made and scored on the same sessions, so every gain is in-sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import feature_study as fs
from . import regime_study as rs
from . import backtester_rules as br
from .term_structure_features import load_term_structure_daily

# VIX9D / VIX at each trade's entry minute (the backtester's rule); the session-open value stays in TS_OPEN.
TS_9D_30 = "ts_9d_30_entry"
TS_OPEN = "ts_9d_30_open"

STUDY_NAME = "SPX-IC-Final-Version-Study"
LAST_TRANCHE = "15:53:00"
ALL_TRANCHES = ["09:32", "09:52", "10:13", "10:33", "10:53", "11:13", "11:33", "11:53", "12:13", "12:33",
                "12:53", "13:13", "13:33", "13:53", "14:13", "14:33", "14:53", "15:13", "15:33", "15:53"]
# The tranches the strategy trades (live from 2026-10-18): the first seven forward selection adds at the
# tranche step of the cascade (``cascade.tranche_selection``). Before: 10:53-15:53 without 12:33 (15).
SELECTED_TRANCHES = ["11:33", "11:53", "12:13", "13:13", "13:53", "14:13", "14:53"]
PREVIOUS_TRANCHES = ["10:53", "11:13", "11:33", "11:53", "12:13", "12:53", "13:13", "13:33", "13:53", "14:13",
                     "14:33", "14:53", "15:13", "15:33", "15:53"]
AB_DROPPED = [t for t in ALL_TRANCHES if t not in SELECTED_TRANCHES]   # the tranches the strategy doesn't trade
THURSDAY = 3
SPLIT_YEAR = 2024  # first half: sessions up to and including this year
PL = "pl"              # the strategy's P/L: held to expiry (no stop since 2026-10-18)
PL_EXPIRY = "pl"       # held to expiry: the raw export
PL_STOP = "pl_stop"    # with a CALL_STOP call stop, studied in ``stop_scan`` but not traded
EXIT_SLIPPAGE = fs.SLIPPAGE  # $0.20 per contract = $20 per stopped spread
CALL_STOP = 400              # % of the call credit: the stop studied (traded 2026-10-07 to 2026-10-18; 300% before)
MAX_SLRATIO = 4.0            # whole-condor short/long cap (5 before 2026-10-07)

MORNING = tuple(f"{t}:00" for t in AB_DROPPED)   # the entry times the strategy drops
# The trend rule: SPX at entry above its 10-day SMA (``sma10_min``; the 20-day, ``sma20_min``, before 2026-10-18).
STRATEGY = {"ts_low": 0.87, "ts_high": 1.02, "max_move": 0.0, "sma10_min": 0.0,
            "drop": MORNING, "skip_thursday": True, "max_slratio": MAX_SLRATIO}

SWEEPS = {
    "ts_high": [0.98, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.08, 1.10, None],
    "ts_low": [0.80, 0.83, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95, None],
    "max_move": [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, None],
    "sma10_min": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, None],
}
SWEEP_LABELS = {
    "ts_high": "Cap: VIX9D/VIX at entry <= x",
    "ts_low": "Floor: VIX9D/VIX at entry >= x",
    "max_move": "Max SPX move up since the open (%)",
    "sma10_min": "SPX at entry vs its SMA10 (%, > x)",
}
N_NOISE = 200
N_BOOT = 2000
SEED = 0

# Remaining-features candidates: feature_study's list without the EMAs and the columns
# the strategy already sets. ``noise`` (a random number per session) goes through the
# same search as a control: whatever it scores is luck.
EMA_FEATURES = ["ema5_gt_ema9", "ema_spread_pct", "ema20m_gt_ema40m", "ema20m_40m_spread_pct"]
ALREADY_SET = ["spx_vs_sma10_pct", "ts_9d_30_open", "spx_move_pct", "dow_code", "slratio_call", "slratio_put",
               "slratio_condor"]
# Left out of the ranking to keep it readable: weak, redundant, or a variant of a kept feature.
DROPPED_FEATURES = ["ts_1d_9d_open", "ivr30", "ivr252", "vvix_prev", "vvix_chg_pct", "vvix_vix_ratio",
                    "spx_vs_sma50_pct", "sma_spread_pct", "ret_prev_pct", "vix_overnight_pct", "mins_from_open",
                    "rsi14_lagged"]
EXTRA_FEATURES = {
    "spx_vs_sma20_pct": ("trend", "SPX at entry vs its 20-day SMA (backtester rule), %"),
    "ret5_prev_pct": ("trend", "SPX 5-day return, prior close, %"),
    "rv10_prev": ("realized vol", "SPX 10-day realized vol, prior close, % (annualised)"),
    "prior_day_range_pct": ("realized vol", "SPX prior day's high-low range, %"),
    "range_so_far": ("intraday", "SPX high-low range from 09:32 to entry, %"),
    "dd_from_hi": ("intraday", "SPX below its intraday high at entry, %"),
    "ts_overnight_chg": ("term structure", "VIX9D/VIX30 change, prior close to open"),
    "spx_off_20d_high_pct": ("trend", "SPX prior close below its 20-day high, %"),
    "up_days_5": ("trend", "SPX up days in the last 5 sessions"),
    "vix_vs_sma20_pct": ("vol trend", "VIX at the open vs its 20-day SMA, %"),
    "vix_5d_chg_pct": ("vol trend", "VIX at the open vs 5 sessions ago, %"),
    "ts_5d_chg": ("term structure", "VIX9D/VIX30 change over 5 sessions"),
    "ts_1d_30_open": ("term structure", "VIX1D / VIX30 at the open"),
    "skew_credit": ("fill", "put credit minus call credit, $"),
    "ret_last_tranche": ("intraday", "SPX change over the 20 minutes before entry, %"),
    "opex_week": ("calendar", "monthly options-expiration week (1) or not (0)"),
    # From the vol-regime feature library (the ones it found held up, plus a few cheap ones). The repo has no
    # SPX OHLC bars, so a session's high, low and open are the extremes and first of its tranche spots.
    "vix9d_gap_pct": ("vol gap", "VIX9D overnight gap, open vs prior close, %"),
    "vix9d_gap_band": ("vol gap", "VIX9D gap in [-3.3%, -0.8%) (1) or not (0)"),
    "month_end": ("calendar", "last session of the month (1) or not (0)"),
    "spx_entry_over_sma5": ("trend", "SPX at entry vs SMA5 of prior closes, %"),
    "atr20_pct": ("realized vol", "SPX 20-day ATR, % of prior close"),
    "ivrv_vix9d_pk1": ("VRP", "VIX9D prior close / prior day's Parkinson vol"),
    "vrp20": ("VRP", "VIX prior close minus SPX 20-day realized vol"),
    "day_efficiency_prev": ("realized vol", "prior day |close - open| / range"),
    "implied_vs_range5": ("VRP", "VIX-implied daily move / 5-day average range"),
}
ADDON_FEATURES = {
    **{k: v for k, v in fs.FEATURES.items() if k not in EMA_FEATURES + ALREADY_SET + DROPPED_FEATURES},
    "sma5_gt_sma9": ("trend", "SPX SMA5 > SMA9, prior close"),
    **EXTRA_FEATURES,
}
ADDON_FLAGS = ["sma5_gt_sma9", "opex_week", "vix9d_gap_band", "month_end"]

SL_SIDES = {"slratio_condor": "whole condor"}
SL_CAPS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, None]
WEEKDAYS = rs.WEEKDAYS


# --------------------------------------------------------------------------- #
# panel and versions
# --------------------------------------------------------------------------- #
def build_final_panel(trades_csv: str | Path,
                      term_structure_csv: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(panel, spreads)``: every condor on every session with its features, and the per-side rows.

    ``panel`` is ``feature_study.build_feature_panel`` with the strategy's filters
    measured the way the backtester measures them (see ``backtester_rules``):
    VIX9D / VIX at the entry minute, the short/long ratio of the whole condor,
    SPX at entry against its N-day SMA (prior N - 1 16:00 closes plus the entry
    price), and a call stop replayed from the option chain (``pl_stop``, not traded).
    ``close`` is the 16:00 close. ``spreads`` keeps the per-side rows.
    """
    panel, spreads = fs.build_feature_panel(trades_csv, term_structure_csv)
    closes = br.session_closes(spreads)
    panel["close"] = panel["date"].map(closes)
    key = pd.MultiIndex.from_frame(panel[["date", "entry_time"]])
    panel[TS_9D_30] = br.entry_term_structure(panel)
    panel["slratio_condor"] = br.condor_slratio(spreads).reindex(key).to_numpy()
    for w in (10, 20):
        panel[f"spx_vs_sma{w}_pct"] = br.spot_vs_sma_pct(panel["spot"], panel["date"], closes, w)
    sma20 = closes.rolling(20, min_periods=20).mean()
    panel["spx_close_vs_sma20_pct"] = panel["date"].map(((closes / sma20 - 1) * 100).shift(1))
    sma50 = closes.rolling(50, min_periods=50).mean()
    panel["spx_vs_sma50_pct"] = panel["date"].map(((closes / sma50 - 1) * 100).shift(1))
    # Prior-session features: computed on the session series, then shifted one session.
    rets = closes.pct_change()
    panel["ret5_prev_pct"] = panel["date"].map((closes.pct_change(5) * 100).shift(1))
    panel["rv10_prev"] = panel["date"].map((rets.rolling(10, min_periods=10).std() * np.sqrt(252) * 100).shift(1))
    day_range = panel.groupby("date")["range_so_far"].max().sort_index()
    panel["prior_day_range_pct"] = panel["date"].map(day_range.shift(1))
    panel["ts_overnight_chg"] = panel[TS_OPEN] - panel["ts_9d_30_close_prev"]
    high20 = closes.rolling(20, min_periods=20).max()
    panel["spx_off_20d_high_pct"] = panel["date"].map(((closes / high20 - 1) * 100).shift(1))
    panel["up_days_5"] = panel["date"].map((rets > 0).astype(float).rolling(5, min_periods=5).sum().shift(1))
    # VIX and term-structure history from the daily file (closes before the session, opens of past sessions).
    tsd = load_term_structure_daily(**({} if term_structure_csv is None else {"path": term_structure_csv}))
    tsd = tsd.set_index("date").sort_index()
    vix_sma20_prev = tsd["vix30_close"].rolling(20, min_periods=20).mean().shift(1)
    panel["vix_vs_sma20_pct"] = (panel["vix30_open"] / panel["date"].map(vix_sma20_prev) - 1) * 100
    panel["vix_5d_chg_pct"] = (panel["vix30_open"] / panel["date"].map(tsd["vix30_close"].shift(5)) - 1) * 100
    panel["ts_5d_chg"] = panel[TS_OPEN] - panel["date"].map(tsd[TS_OPEN].shift(5))
    # Monthly OPEX week: the Monday-Friday week holding the month's third Friday.
    dates = pd.Series(pd.to_datetime(panel["date"].unique()))
    first = dates.dt.to_period("M").dt.to_timestamp()
    third_friday = first + pd.to_timedelta((4 - first.dt.weekday) % 7 + 14, unit="D")
    opex = dict(zip(dates, (dates.dt.isocalendar().week.to_numpy() == third_friday.dt.isocalendar().week.to_numpy())
                    & (dates.dt.year.to_numpy() == third_friday.dt.year.to_numpy())))
    panel["opex_week"] = panel["date"].map(opex).astype(float)
    # Vol-regime library features. Lags run on the market calendar (the daily VIX file) or the session series.
    vix9d_gap = (tsd["vix9d_open"] / tsd["vix9d_close"].shift(1) - 1) * 100
    panel["vix9d_gap_pct"] = panel["date"].map(vix9d_gap)
    panel["vix9d_gap_band"] = ((panel["vix9d_gap_pct"] >= -3.3) & (panel["vix9d_gap_pct"] < -0.8)).astype(float)
    cal = pd.Series(tsd.index, index=tsd.index)
    month_end = cal.dt.to_period("M") != cal.shift(-1).dt.to_period("M")
    panel["month_end"] = panel["date"].map(month_end).astype(float)
    sma5_prev = closes.rolling(5, min_periods=5).mean().shift(1)
    panel["spx_entry_over_sma5"] = (panel["spot"] / panel["date"].map(sma5_prev) - 1) * 100
    by_day = panel.groupby("date")["spot"]
    o, h, l = by_day.first().sort_index(), by_day.max().sort_index(), by_day.min().sort_index()
    cp = closes.shift(1)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    panel["atr20_pct"] = panel["date"].map((tr.rolling(20, min_periods=20).mean() / closes * 100).shift(1))
    parkinson = 100 * np.log(h / l) / np.sqrt(4 * np.log(2)) * np.sqrt(252)
    vix9d_close_prev = tsd["vix9d_close"].shift(1)
    panel["ivrv_vix9d_pk1"] = panel["date"].map(vix9d_close_prev) / panel["date"].map(parkinson.shift(1))
    rv20 = rets.rolling(20, min_periods=20).std() * np.sqrt(252) * 100
    panel["vrp20"] = panel["vix30_close_prev"] - panel["date"].map(rv20.shift(1))
    panel["day_efficiency_prev"] = panel["date"].map(((closes - o).abs() / (h - l)).shift(1))
    range5 = ((h - l) / o * 100).rolling(5, min_periods=5).mean().shift(1)
    panel["implied_vs_range5"] = (panel["vix30_open"] / np.sqrt(252)) / panel["date"].map(range5)
    # feature_study applies its own 175% stop; replace it with the CALL_STOP stop replayed from the chain.
    panel[PL_STOP] = condor_pl(panel, spreads, CALL_STOP, EXIT_SLIPPAGE)[0]
    return add_outcomes(panel, spreads), spreads


def condor_pl(panel: pd.DataFrame, spreads: pd.DataFrame, stop_pct: float | None,
              slippage: float = EXIT_SLIPPAGE) -> tuple[pd.Series, pd.Series]:
    """Condor P/L with the call stop replayed from the chain (puts to expiry), and whether the stop fired."""
    key = pd.MultiIndex.from_frame(panel[["date", "entry_time"]])
    call, hit = br.call_side_pl(spreads, stop_pct, slippage)
    put = spreads[spreads["side"] == "put"].set_index(["date", "entry_time"])["p_l"]
    return (pd.Series(call.reindex(key).to_numpy() + put.reindex(key).to_numpy(), index=panel.index),
            pd.Series(hit.reindex(key).fillna(False).to_numpy(bool), index=panel.index))


def add_outcomes(panel: pd.DataFrame, spreads: pd.DataFrame) -> pd.DataFrame:
    """What happened **after** each entry — to explain why a filter works, never to filter on.

    * ``pl_call`` / ``pl_put`` — each side's P/L, held to expiry;
    * ``pl_call_stop`` / ``call_stop_hit`` — the call side with a CALL_STOP stop, and whether it fired;
    * ``move_after_entry_pct`` — SPX from the entry to the 16:00 close, %;
    * ``call_breached`` / ``put_breached`` — SPX closed beyond the short call / short put;
    * ``day_range_pct`` — the session's full high-low range, % of the open.
    """
    out = panel.copy()
    key = pd.MultiIndex.from_frame(out[["date", "entry_time"]])
    call, hit = br.call_side_pl(spreads, CALL_STOP, EXIT_SLIPPAGE)
    out["pl_call_stop"] = call.reindex(key).to_numpy()
    side = spreads.set_index(["date", "entry_time"])
    out["pl_call"] = side.loc[side["side"] == "call", "p_l"].reindex(key).to_numpy()
    out["pl_put"] = side.loc[side["side"] == "put", "p_l"].reindex(key).to_numpy()
    out["call_stop_hit"] = hit.reindex(key).fillna(False).to_numpy(bool)
    out["move_after_entry_pct"] = (out["close"] / out["spot"] - 1) * 100
    out["call_breached"] = (out["close"] - out["spot"]) > out["otm_points_call"]
    out["put_breached"] = (out["spot"] - out["close"]) > out["otm_points_put"]
    out["day_range_pct"] = out.groupby("date")["range_so_far"].transform("max")
    return out


# --------------------------------------------------------------------------- #
# 0. how P/L varies across each filter's feature, and why
# --------------------------------------------------------------------------- #
# feature -> (plain name, bins, the strategy's kept range, the filter keys switched off to see both sides).
PROFILES = {
    TS_9D_30: ("VIX9D / VIX at entry", [0, 0.8, 0.85, 0.87, 0.9, 0.95, 1.0, 1.02, 1.04, 1.08, 2],
                      (0.87, 1.02), {"ts_low": None, "ts_high": None}),
    "spx_move_pct": ("SPX move since the 09:30 open at entry, %", [-10, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 10],
                     (-10, 0.0), {"max_move": None}),
    "spx_vs_sma20_pct": ("SPX at entry vs its 20-day SMA (backtester rule), %", [-20, -2, -1, 0, 0.5, 1, 1.5, 2, 3, 20],
                         None, {"sma10_min": None}),
    "spx_vs_sma10_pct": ("SPX at entry vs its 10-day SMA (backtester rule), %", [-20, -2, -1, 0, 0.5, 1, 1.5, 2, 3, 20],
                         (0.0, 20), {"sma10_min": None}),
}


# Profiles whose name isn't the column they bin on.
PROFILE_COLUMN: dict[str, str] = {}

# The outcome column that explains each feature, shown as the third panel of its profile.
MECHANISM = {
    TS_9D_30: ("put_breached_pct", "SPX closes through the short put", "% of condors"),
    "spx_move_pct": ("call_breached_pct", "SPX closes through the short call", "% of condors"),
    "spx_vs_sma20_pct": ("abs_move_after_entry_pct", "How far SPX moves after entry", "avg |entry -> close|, %"),
    "spx_vs_sma10_pct": ("abs_move_after_entry_pct", "How far SPX moves after entry", "avg |entry -> close|, %"),
}


def _bin_label(iv: pd.Interval, lo_open: float, hi_open: float) -> str:
    if iv.left <= lo_open:
        return f"<= {iv.right:g}"
    if iv.right >= hi_open:
        return f"> {iv.left:g}"
    return f"{iv.left:g} to {iv.right:g}"


def feature_profile(panel: pd.DataFrame, feature: str) -> pd.DataFrame:
    """P/L and what happened after entry, per bin of ``feature``, with that filter off and the rest on.

    One row per bin: condors and sessions, avg P/L per condor with a session-bootstrap
    CI, win rate, total P/L, each side's avg P/L, how often the call stop fired and
    either short strike was breached at the close, how far SPX moved from entry to
    the close, and the session's range. ``in_strategy`` marks the bins the strategy keeps.
    """
    _, bins, kept, off = PROFILES[feature]
    col = PROFILE_COLUMN.get(feature, feature)
    book = panel[version_mask(panel, **{**STRATEGY, **off})]
    rows = []
    for iv, grp in book.groupby(pd.cut(book[col], bins), observed=True):
        lo, hi = rs.session_bootstrap_mean(grp[PL], grp["date"])
        first = grp["date"].dt.year <= SPLIT_YEAR
        rows.append({
            "bin": _bin_label(iv, bins[0], bins[-1]), "condors": len(grp), "trades": 2 * len(grp),
            "sessions": grp["date"].nunique(),
            "avg_pl": grp[PL].mean(), "ci_low": lo, "ci_high": hi, "win_rate": (grp[PL] > 0).mean() * 100,
            f"avg_pl_to_{SPLIT_YEAR}": grp.loc[first, PL].mean(),
            f"avg_pl_from_{SPLIT_YEAR + 1}": grp.loc[~first, PL].mean(),
            "total_pl": grp[PL].sum(), "avg_pl_call": grp["pl_call"].mean(), "avg_pl_put": grp["pl_put"].mean(),
            "call_stop_hit_pct": grp["call_stop_hit"].mean() * 100,
            "call_breached_pct": grp["call_breached"].mean() * 100,
            "put_breached_pct": grp["put_breached"].mean() * 100,
            "abs_move_after_entry_pct": grp["move_after_entry_pct"].abs().mean(),
            "day_range_pct": grp["day_range_pct"].mean(),
            "in_strategy": kept is not None and bool(iv.left >= kept[0] - 1e-9 and iv.right <= kept[1] + 1e-9),
        })
    return pd.DataFrame(rows).set_index("bin")


def sma_regime(panel: pd.DataFrame) -> pd.DataFrame:
    """Every session (no filter): the market above vs below its 10-day SMA — range, VIX, and the book's P/L."""
    d = panel.groupby("date").agg(above=("spx_vs_sma10_pct", "first"), range_=("day_range_pct", "first"),
                                  vix=("vix30_open", "first"), ts=(TS_9D_30, "first"), pl=(PL, "sum")).dropna()
    d["state"] = np.where(d["above"] > 0, "SPX above SMA10", "SPX below SMA10")
    out = d.groupby("state").agg(sessions=("pl", "size"), avg_day_range_pct=("range_", "mean"),
                                 day_range_p90_pct=("range_", lambda x: x.quantile(0.9)),
                                 avg_vix=("vix", "mean"), avg_ts_9d_30=("ts", "mean"),
                                 avg_session_pl_all_tranches=("pl", "mean"))
    return out


def version_mask(panel: pd.DataFrame, ts_low: float | None = None, ts_high: float | None = None,
                 max_move: float | None = None, sma20_min: float | None = None, sma10_min: float | None = None,
                 drop: tuple[str, ...] = (), skip_thursday: bool = False, max_slratio: float | None = None,
                 trend: pd.Series | None = None) -> pd.Series:
    """Keep-mask for one version. ``None`` switches a condition off; ``trend`` replaces the SMA rules."""
    keep = pd.Series(True, index=panel.index)
    if ts_low is not None:
        keep &= panel[TS_9D_30] >= ts_low
    if ts_high is not None:
        keep &= panel[TS_9D_30] <= ts_high
    if max_move is not None:
        keep &= panel["spx_move_pct"] <= max_move
    if trend is not None:
        keep &= trend.fillna(False).astype(bool)
    else:
        if sma20_min is not None:
            keep &= panel["spx_vs_sma20_pct"] > sma20_min
        if sma10_min is not None:
            keep &= panel["spx_vs_sma10_pct"] > sma10_min
    if drop:
        keep &= ~panel["entry_time"].isin(drop)
    if skip_thursday:
        keep &= panel["dow_code"] != THURSDAY
    if max_slratio is not None:
        keep &= panel["slratio_condor"] <= max_slratio
    return keep


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
@dataclass
class Calendars:
    """The session calendars every version is scored on: all sessions, and each half."""

    all: np.ndarray
    first: np.ndarray
    second: np.ndarray
    first_mask: pd.Series = field(repr=False)

    @classmethod
    def of(cls, panel: pd.DataFrame) -> "Calendars":
        first = panel["date"].dt.year <= SPLIT_YEAR
        return cls(np.sort(panel["date"].unique()), np.sort(panel.loc[first, "date"].unique()),
                   np.sort(panel.loc[~first, "date"].unique()), first)


H1, H2 = f"sharpe_to_{SPLIT_YEAR}", f"sharpe_from_{SPLIT_YEAR + 1}"
SCORE_COLS = ["sessions", "trades", "win_rate", "avg_pl", "total_pl", "sharpe", "max_drawdown", H1, H2]


def score_pl(pl: pd.Series, dates: pd.Series, cal: Calendars) -> dict:
    """Book stats of ``pl`` on all sessions, Sharpe in each half, and the number of sessions traded.

    Sharpe is the backtester's (``backtester_rules.sharpe``): daily P/L over $200k less a 4% risk-free rate.
    """
    full = rs.book_stats(pl, dates, cal.all)
    first = cal.first_mask.loc[pl.index]
    return {
        "sessions": int(dates.nunique()), "trades": 2 * int(len(pl)),  # one trade = one vertical spread
        "win_rate": full["win_rate"], "avg_pl": full["avg_pl"],
        "total_pl": full["total_pl"], "sharpe": br.sharpe(rs.daily_series(pl, dates, cal.all)),
        "max_drawdown": full["max_drawdown"],
        H1: br.sharpe(rs.daily_series(pl[first], dates[first], cal.first)),
        H2: br.sharpe(rs.daily_series(pl[~first], dates[~first], cal.second)),
    }


def score(panel: pd.DataFrame, keep: pd.Series, cal: Calendars, col: str = PL) -> dict:
    return score_pl(panel.loc[keep, col], panel.loc[keep, "date"], cal)


def _fmt(value) -> str:
    return "off" if value is None else f"{value:g}"


# --------------------------------------------------------------------------- #
# 1. build-up and leave-one-out
# --------------------------------------------------------------------------- #
BUILD_STEPS = {
    "raw book (no filter)": {},
    "+ 0.87-1.02 band": {"ts_low": 0.87, "ts_high": 1.02},
    "+ move <= 0%": {"max_move": 0.0},
    "+ above 10-day SMA": {"sma10_min": 0.0},
    f"+ short/long ratio <= {MAX_SLRATIO:g} (condor)": {"max_slratio": MAX_SLRATIO},
    "+ no Thursdays": {"skip_thursday": True},
    "+ forward-selected tranches": {"drop": MORNING},
}


def build_up(panel: pd.DataFrame, cal: Calendars) -> pd.DataFrame:
    """From the raw book, each entry filter added in turn; every row held to expiry (``pl``), the last is the strategy.

    ``win_rate`` is per condor; ``spread_win_rate`` counts each vertical spread as a trade, like the backtester.
    """
    cfg, rows = {}, {}
    for name, add in BUILD_STEPS.items():
        cfg = {**cfg, **add}
        keep = version_mask(panel, **cfg)
        rows[name] = score(panel, keep, cal, PL)
        sides = pd.concat([panel.loc[keep, "pl_call"], panel.loc[keep, "pl_put"]]).dropna()
        rows[name]["spread_win_rate"] = (sides > 0).mean() * 100
    out = pd.DataFrame(rows).T[SCORE_COLS[:2] + ["spread_win_rate"] + SCORE_COLS[2:]]
    out["sharpe_step"] = out["sharpe"].diff()
    return out


WITH_STOP = f"with a {CALL_STOP}% call stop (not traded)"
LEAVE_ONE_OUT = {
    "strategy": {},
    WITH_STOP: {},
    "without the band": {"ts_low": None, "ts_high": None},
    "without move <= 0%": {"max_move": None},
    "without SPX > SMA10": {"sma10_min": None},
    "trading Thursdays": {"skip_thursday": False},
    "keeping the morning tranches": {"drop": ()},  # i.e. every tranche
    f"without short/long ratio <= {MAX_SLRATIO:g}": {"max_slratio": None},
}


def leave_one_out(panel: pd.DataFrame, cal: Calendars) -> pd.DataFrame:
    """The strategy with one filter removed at a time, the rest kept."""
    rows = {name: score(panel, version_mask(panel, **{**STRATEGY, **change}), cal,
                        PL_STOP if name == WITH_STOP else PL)
            for name, change in LEAVE_ONE_OUT.items()}
    out = pd.DataFrame(rows).T[SCORE_COLS]
    out["sharpe_vs_strategy"] = out["sharpe"] - out.loc["strategy", "sharpe"]
    out["total_pl_vs_strategy"] = out["total_pl"] - out.loc["strategy", "total_pl"]
    return out


# --------------------------------------------------------------------------- #
# 2. sweeps
# --------------------------------------------------------------------------- #
def sweeps(panel: pd.DataFrame, cal: Calendars) -> dict[str, pd.DataFrame]:
    """For each threshold in ``SWEEPS``: every value, the rest of the strategy fixed."""
    out = {}
    for name, values in SWEEPS.items():
        rows = {_fmt(v): score(panel, version_mask(panel, **{**STRATEGY, name: v}), cal) for v in values}
        t = pd.DataFrame(rows).T[["sessions", "trades", "total_pl", "sharpe", "max_drawdown", H1, H2]]
        t.index.name = name
        t["is_strategy"] = [v == _fmt(STRATEGY[name]) for v in t.index]
        out[name] = t
    return out


def sweep_verdict(sw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per threshold: the strategy's value and Sharpe, and the best value overall and in each half."""
    rows = []
    for name, t in sw.items():
        mine = _fmt(STRATEGY[name])
        best = t["sharpe"].astype(float).idxmax()
        rows.append({"threshold": name, "strategy_value": mine, "strategy_sharpe": t.loc[mine, "sharpe"],
                     "best_value": best, "best_sharpe": t.loc[best, "sharpe"],
                     f"best_{H1}": t[H1].astype(float).idxmax(), f"best_{H2}": t[H2].astype(float).idxmax()})
    return pd.DataFrame(rows).set_index("threshold")


# --------------------------------------------------------------------------- #
# 3. the trend rule
# --------------------------------------------------------------------------- #
STRATEGY_TREND = "SPX > SMA10 (strategy)"


def trend_rules(panel: pd.DataFrame) -> dict[str, pd.Series | None]:
    return {
        "no trend rule": None,
        STRATEGY_TREND: panel["spx_vs_sma10_pct"] > 0,
        "SPX >= 0.5% above SMA10": panel["spx_vs_sma10_pct"] >= 0.5,
        "SPX >= 1% above SMA10": panel["spx_vs_sma10_pct"] >= 1,
        "SPX > SMA20 (previous rule)": panel["spx_vs_sma20_pct"] > 0,
        "SPX > SMA10 and SMA20": (panel["spx_vs_sma10_pct"] > 0) & (panel["spx_vs_sma20_pct"] > 0),
        "SPX > SMA50": panel["spx_vs_sma50_pct"] > 0,
        "SMA5 > SMA9": panel["sma5_gt_sma9"] == 1,
        "SPX > SMA10 and SMA5 > SMA9": (panel["spx_vs_sma10_pct"] > 0) & (panel["sma5_gt_sma9"] == 1),
    }


def trend_swap(panel: pd.DataFrame, cal: Calendars) -> pd.DataFrame:
    """The SMA10 rule swapped for each alternative, the rest of the strategy fixed."""
    base = {k: v for k, v in STRATEGY.items() if k not in ("sma10_min", "sma20_min")}
    rows = {name: score(panel, version_mask(panel, **base) if rule is None
                        else version_mask(panel, **base, trend=rule), cal)
            for name, rule in trend_rules(panel).items()}
    out = pd.DataFrame(rows).T[["sessions", "trades", "total_pl", "sharpe", "max_drawdown", H1, H2]]
    out["sharpe_vs_strategy"] = out["sharpe"] - out.loc[STRATEGY_TREND, "sharpe"]
    return out


# --------------------------------------------------------------------------- #
# 4. calendar: weekdays and tranches
# --------------------------------------------------------------------------- #
def _session_mean_ci(daily: pd.Series, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = daily.to_numpy()
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def weekday_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Session-level P/L per weekday inside the other filters (Thursdays included, to show why they're out)."""
    book = panel[version_mask(panel, **{**STRATEGY, "skip_thursday": False})]
    daily = book.groupby(["date", "dow"])[PL].sum().reset_index()
    rows = []
    for day in WEEKDAYS:
        d = daily.loc[daily["dow"] == day, PL]
        lo, hi = _session_mean_ci(d)
        rows.append({"weekday": day, "sessions": len(d), "trades": 2 * int((book["dow"] == day).sum()),
                     "avg_session_pl": d.mean(), "ci_low": lo, "ci_high": hi,
                     "winning_sessions_pct": (d > 0).mean() * 100, "total_pl": d.sum()})
    return pd.DataFrame(rows).set_index("weekday")


def tranche_table(panel: pd.DataFrame, cal: Calendars) -> pd.DataFrame:
    """Per tranche inside the other filters (15:53 included): avg P/L with a session-bootstrap CI."""
    book_keep = version_mask(panel, **{**STRATEGY, "drop": ()})
    ref = score(panel, book_keep, cal)
    rows = []
    for t in sorted(panel.loc[book_keep, "entry_time"].unique()):
        this = book_keep & (panel["entry_time"] == t)
        lo, hi = rs.session_bootstrap_mean(panel.loc[this, PL], panel.loc[this, "date"])
        without = score(panel, book_keep & (panel["entry_time"] != t), cal)
        rows.append({"entry_time": t[:5], "condors": int(this.sum()), "trades": 2 * int(this.sum()),
                     "avg_pl": panel.loc[this, PL].mean(),
                     "ci_low": lo, "ci_high": hi, "total_pl": panel.loc[this, PL].sum(),
                     "sharpe_if_dropped": without["sharpe"] - ref["sharpe"], "status": tranche_status(t[:5])})
    return pd.DataFrame(rows).set_index("entry_time")


def tranche_status(t: str) -> str:
    """Whether the live strategy trades a tranche (``HH:MM``)."""
    return "dropped" if t in AB_DROPPED else "traded"


def highlight_dropped(df: pd.DataFrame, digits: int = 2):
    """A tranche table (index ``HH:MM``) with the tranches the live strategy no longer trades shaded."""
    shade = {"dropped": "background-color: #fbe3d9; font-weight: bold"}

    def row_style(row):
        return [shade.get(tranche_status(row.name), "")] * len(row)

    return df.style.apply(row_style, axis=1).format(precision=digits)


def spread_panel(panel: pd.DataFrame, spreads: pd.DataFrame) -> pd.DataFrame:
    """One row per vertical spread with the entry features the backtester filters on."""
    cols = [TS_9D_30, "spx_vs_sma20_pct", "spx_vs_sma10_pct", "slratio_condor"]
    per = panel.set_index(["date", "entry_time"])[cols]
    sp = spreads.copy()
    sp = sp.join(per, on=["date", "entry_time"])
    sp["move_pct"] = sp["movement"] / (sp["opening_price"] - sp["movement"]) * 100
    sp["dow_code"] = sp["date"].dt.dayofweek
    sp["hhmm"] = sp["entry_time"].str[:5]
    return sp


AB_BOOT = 2000


def _paired_boot(a: np.ndarray, b: np.ndarray, n_boot: int = AB_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Sharpe(b) - Sharpe(a) resampling whole sessions on the same draws: (2.5th pct, 97.5th pct, share b better)."""
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n_boot, len(a)))

    def sharpe(x):
        return br.sharpe_rows(x)

    d = sharpe(b[idx]) - sharpe(a[idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


# --------------------------------------------------------------------------- #
# forward selection of the tranches, and a walk-forward check of it
# --------------------------------------------------------------------------- #
# Inside the strategy (every filter, the call stop, condor P/L), with all 20 tranches available.
LIVE_TRANCHES = [t for t in ALL_TRANCHES if t not in AB_DROPPED]
WF_TEST_YEARS = [2023, 2024, 2025, 2026]


def _tranche_book(panel: pd.DataFrame) -> pd.DataFrame:
    """The strategy's condors with every tranche allowed (morning and 15:53 included), tranche as ``HH:MM``."""
    book = panel[version_mask(panel, **{**STRATEGY, "drop": ()})].copy()
    book["tranche"] = book["entry_time"].str[:5]
    return book


def _sharpe_cols(m: np.ndarray) -> float:
    return br.sharpe(m)


def _set_stats(book: pd.DataFrame, daily: pd.DataFrame, tranches: list[str], first: np.ndarray) -> dict:
    """Book stats for a tranche set: Sharpe on all sessions and per half, P/L, drawdown, win rates."""
    d = daily[tranches].sum(axis=1).to_numpy()
    eq = np.cumsum(d)
    rows = book[book["tranche"].isin(tranches)]
    sides = pd.concat([rows["pl_call"], rows["pl_put"]]).dropna()
    return {"tranches": len(tranches), "condors": len(rows), "total_pl": float(d.sum()), "sharpe": _sharpe_cols(d),
            H1: _sharpe_cols(d[first]), H2: _sharpe_cols(d[~first]),
            "max_drawdown": float((eq - np.maximum.accumulate(eq)).min()),
            "win_rate_spread": float((sides > 0).mean() * 100), "win_rate_condor": float((rows[PL] > 0).mean() * 100),
            "avg_pl_condor": float(rows[PL].mean())}


def _forward_path(daily: pd.DataFrame, trials: list | None = None) -> list[tuple[str, float]]:
    """Forward selection on a date x tranche P/L matrix: start with the tranche whose own Sharpe is highest,
    then at each step try adding every remaining tranche and keep the one that gives the book the highest
    Sharpe, until every tranche is in. Returns (added, Sharpe) per step; ``trials`` (if given) receives each
    step's Sharpe for every tranche tried."""
    left, chosen, path = list(daily.columns), [], []
    total = np.zeros(len(daily))
    while left:
        scores = {t: _sharpe_cols(total + daily[t].to_numpy()) for t in left}
        best = max(scores, key=lambda t: (np.nan_to_num(scores[t], nan=-np.inf), t))
        if trials is not None:
            trials.append(scores)
        chosen.append(best)
        left.remove(best)
        total = total + daily[best].to_numpy()
        path.append((best, scores[best]))
    return path


def tranche_forward_selection(panel: pd.DataFrame, cal: Calendars) -> pd.DataFrame:
    """Forward selection over all 20 tranches on all sessions; one row per step (the book after adding it).

    In-sample: the order and the best set are chosen and scored on the same sessions.
    ``attrs["best"]`` is the set at the highest Sharpe; :func:`tranche_walk_forward` checks it out of sample.
    """
    book = _tranche_book(panel)
    daily = book.pivot_table(index="date", columns="tranche", values=PL, aggfunc="sum").reindex(
        cal.all, fill_value=0).reindex(columns=ALL_TRANCHES, fill_value=0).fillna(0)
    first = pd.Series(cal.all).dt.year.to_numpy() <= SPLIT_YEAR
    rows, chosen, trials = [], [], []
    for step, (t, _) in enumerate(_forward_path(daily, trials), start=1):
        chosen.append(t)
        rows.append({"step": step, "added": t, **_set_stats(book, daily, list(chosen), first)})
    out = pd.DataFrame(rows).set_index("step")
    best = int(out["sharpe"].idxmax())
    tried = pd.DataFrame(trials, index=pd.Index(range(1, len(trials) + 1), name="step"))[ALL_TRANCHES]
    out.attrs.update(best=list(out["added"].iloc[:best]), best_step=best, trials=tried,
                     live=_set_stats(book, daily, PREVIOUS_TRANCHES, first), live_set=PREVIOUS_TRANCHES,
                     all20=_set_stats(book, daily, ALL_TRANCHES, first))
    return out


def forward_trials_view(fwd: pd.DataFrame):
    """Every step of the forward selection: the book's Sharpe if each remaining tranche were added (blank once
    a tranche is in), the one picked (the highest) in bold on a blue cell."""
    tried = fwd.attrs["trials"].copy()
    tried.insert(0, "picked", fwd["added"])

    def mark(row):
        return ["background-color: #d6e6f9; font-weight: bold" if c == row["picked"] else "" for c in row.index]

    return tried.style.apply(mark, axis=1).format(precision=2, na_rep="")


def tranche_walk_forward(panel: pd.DataFrame, cal: Calendars) -> pd.DataFrame:
    """Walk-forward: for each test year, run the forward selection on every earlier session, take the
    best-Sharpe set, and trade it in the test year. Compared with the live set and all 20 in the same years.

    The ``out of sample`` row joins the test years: the selected sets' P/L was never used to choose them.
    """
    book = _tranche_book(panel)
    daily = book.pivot_table(index="date", columns="tranche", values=PL, aggfunc="sum").reindex(
        cal.all, fill_value=0).reindex(columns=ALL_TRANCHES, fill_value=0).fillna(0)
    years = pd.Series(daily.index).dt.year.to_numpy()
    rows, oos = [], {"selected": [], "live set": [], "all 20": []}
    for y in WF_TEST_YEARS:
        train, test = daily[years < y], daily[years == y]
        path = _forward_path(train)
        best = int(np.nanargmax([sh for _, sh in path])) + 1
        chosen = [t for t, _ in path[:best]]
        for name, ts in (("selected", chosen), ("live set", LIVE_TRANCHES), ("all 20", ALL_TRANCHES)):
            oos[name].append(test[ts].sum(axis=1))
        rows.append({"test_year": str(y), "train": f"{daily.index[years < y].min():%Y-%m} to {y - 1}",
                     "tranches_selected": len(chosen),
                     "dropped_by_selection": ", ".join(t for t in ALL_TRANCHES if t not in chosen) or "-",
                     "selected_sharpe": _sharpe_cols(test[chosen].sum(axis=1).to_numpy()),
                     "live_sharpe": _sharpe_cols(test[LIVE_TRANCHES].sum(axis=1).to_numpy()),
                     "all20_sharpe": _sharpe_cols(test[ALL_TRANCHES].sum(axis=1).to_numpy()),
                     "selected_pl": float(test[chosen].sum().sum()),
                     "live_pl": float(test[LIVE_TRANCHES].sum().sum()),
                     "all20_pl": float(test[ALL_TRANCHES].sum().sum())})
    joined = {k: pd.concat(v).to_numpy() for k, v in oos.items()}
    rows.append({"test_year": "out of sample", "train": "expanding", "tranches_selected": np.nan,
                 "dropped_by_selection": "-",
                 "selected_sharpe": _sharpe_cols(joined["selected"]), "live_sharpe": _sharpe_cols(joined["live set"]),
                 "all20_sharpe": _sharpe_cols(joined["all 20"]), "selected_pl": float(joined["selected"].sum()),
                 "live_pl": float(joined["live set"].sum()), "all20_pl": float(joined["all 20"].sum())})
    out = pd.DataFrame(rows).set_index("test_year")
    lo, hi, p = _paired_boot(joined["live set"], joined["selected"])
    out.attrs.update(selected_vs_live_ci=(lo, hi), p_selected_better=p)
    return out


def fig_tranche_forward(fwd: pd.DataFrame, wf: pd.DataFrame, note: str | None = None):
    """Forward-selection path (Sharpe as tranches are added) and the walk-forward test years."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
        ax = axes[0]
        x = fwd.index.to_numpy()
        ax.plot(x, fwd["sharpe"], color=BLUE, marker="o", ms=5.5, label="all sessions")
        for h, style in HALF_STYLE.items():
            ax.plot(x, fwd[h], **style)
        b = fwd.attrs["best_step"]
        ax.scatter([b], [fwd.loc[b, "sharpe"]], s=130, facecolor="none", edgecolor=INK, lw=1.4, zorder=4)
        ax.annotate(f"best: {b} tranches, Sharpe {fwd.loc[b, 'sharpe']:.2f}", (b, fwd.loc[b, "sharpe"]),
                    xytext=(8, 10), textcoords="offset points", fontsize=8, color=INK)
        ax.axhline(fwd.attrs["live"]["sharpe"], color=INK, ls=":", lw=1.4,
                   label=f"previous live set ({len(fwd.attrs['live_set'])}): {fwd.attrs['live']['sharpe']:.2f}")
        ax.set_xticks(x, [f"+{t}" for t in fwd["added"]], rotation=90, fontsize=8)
        ax.set_ylabel("Sharpe of the book")
        ax.set_title("Forward selection: best tranche first, then the one that adds most Sharpe")
        ax.legend(loc="lower right", fontsize=8)
        ax = axes[1]
        w = wf.drop(index="out of sample")
        xx = np.arange(len(w))
        for k, (col, color, label) in enumerate((("selected_sharpe", BLUE, "set selected on earlier years"),
                                                  ("live_sharpe", ORANGE, "live set"),
                                                  ("all20_sharpe", GREY, "all 20"))):
            ax.bar(xx + (k - 1) * 0.26, w[col], width=0.24, color=color, label=label)
        ax.axhline(0, color=INK_2, lw=0.7)
        ax.set_xticks(xx, w.index)
        ax.set_ylabel("Sharpe in the test year")
        ax.set_title("Walk-forward: choose on the past, trade the next year")
        if note:
            fig.suptitle(note, x=0.01, ha="left")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# A/B: SPX above its 20-day vs its 10-day SMA
# --------------------------------------------------------------------------- #
def sma_ab(panel: pd.DataFrame, spreads: pd.DataFrame, cal: Calendars) -> pd.DataFrame:
    """The live strategy above the 10-day SMA (A) against above the 20-day SMA (B), and above both.

    Built the way the backtester counts and filters: **each vertical spread is a trade**, and "above the
    N-day SMA" is **SPX at entry against the average of the prior N - 1 16:00 closes and the entry price**
    (``backtester_rules``). P/L has the strategy's call stop, replayed from the chain. Paired session bootstrap
    of the Sharpe difference vs A. ``attrs`` holds how many trades each rule keeps that the other doesn't,
    and the trades the SMA20 warm-up (first 20 sessions of the export) can't score.
    """
    sp = spread_panel(panel, spreads)
    key = pd.MultiIndex.from_frame(sp[["date", "entry_time"]])
    call_pl = spreads[spreads["side"] == "call"].set_index(["date", "entry_time"])["p_l"].reindex(key).to_numpy()
    put_pl = spreads[spreads["side"] == "put"].set_index(["date", "entry_time"])["p_l"].reindex(key).to_numpy()
    sp["pl_trade"] = np.where(sp["side"] == "call", call_pl, put_pl)
    closes = br.session_closes(spreads)
    above = {w: sp[f"spx_vs_sma{w}_pct"] > 0 for w in (10, 20)}
    base = (sp["hhmm"].isin(LIVE_TRANCHES) & (sp["dow_code"] != THURSDAY)
            & sp[TS_9D_30].between(STRATEGY["ts_low"], STRATEGY["ts_high"])
            & (sp["move_pct"] <= STRATEGY["max_move"]) & (sp["slratio_condor"] <= STRATEGY["max_slratio"]))
    versions = {"A: above 10-day SMA (live)": base & above[10], "B: above 20-day SMA": base & above[20],
                "above both": base & above[20] & above[10]}
    first = pd.Series(cal.all).dt.year.to_numpy() <= SPLIT_YEAR
    daily = {k: rs.daily_series(sp.loc[m, "pl_trade"], sp.loc[m, "date"], cal.all).to_numpy()
             for k, m in versions.items()}
    base_key = next(iter(versions))
    rows = {}
    for k, m in versions.items():
        pl, d = sp.loc[m, "pl_trade"], daily[k]
        eq = np.cumsum(d)
        lo, hi, pb = (np.nan, np.nan, np.nan) if k == base_key else _paired_boot(daily[base_key], d)
        rows[k] = {"trades": int(m.sum()), "sessions": int(sp.loc[m, "date"].nunique()), "total_pl": float(pl.sum()),
                   "sharpe": _sharpe_cols(d), H1: _sharpe_cols(d[first]), H2: _sharpe_cols(d[~first]),
                   "max_drawdown": float((eq - np.maximum.accumulate(eq)).min()),
                   "win_rate_spread": float((pl > 0).mean() * 100), "avg_pl_trade": float(pl.mean()),
                   "win_days_pct": float((d[d != 0] > 0).mean() * 100),
                   "sharpe_vs_A": _sharpe_cols(d) - _sharpe_cols(daily[base_key]),
                   "ci_low": lo, "ci_high": hi, "p_better_than_A": pb}
    out = pd.DataFrame(rows).T
    warm = sp["date"].map(closes.rolling(19, min_periods=19).sum().shift(1)).isna()
    out.attrs.update(only_sma10=int((base & above[10] & ~above[20]).sum()),
                     only_sma20=int((base & above[20] & ~above[10]).sum()),
                     warmup_trades=int((base & warm).sum()))
    return out


def fig_sma_ab(ab: pd.DataFrame, note: str | None = None):
    """The strategy's SMA vs the other vs both: Sharpe, total P/L, win rate per spread and max drawdown."""
    colors = [BLUE, ORANGE, AQUA]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
        for ax, (col, title, fmt) in zip(axes, (("sharpe", "Sharpe", "{:.2f}"), ("total_pl", "Total P/L", None),
                                               ("win_rate_spread", "Win rate per trade (spread)", "{:.1f}%"),
                                               ("max_drawdown", "Max drawdown", None))):
            v = ab[col].astype(float)
            x = np.arange(len(v))
            ax.bar(x, v, width=0.6, color=colors[:len(v)])
            for xi, val in zip(x, v):
                ax.text(xi, val, fmt.format(val) if fmt else _k(val), ha="center",
                        va="bottom" if val >= 0 else "top", fontsize=8.5, color=INK_2)
            ax.set_xticks(x, [("SMA10 (strategy)" if "10-day" in k else "SMA20") if "both" not in k else "both"
                              for k in ab.index])
            ax.set_title(title)
            ax.axhline(0, color=INK_2, lw=0.7)
            if col == "win_rate_spread":
                ax.set_ylim(v.min() - 3, v.max() + 1.5)
            if col in ("total_pl", "max_drawdown"):
                from matplotlib.ticker import MultipleLocator
                if col == "max_drawdown":
                    ax.yaxis.set_major_locator(MultipleLocator(5000))
                ax.yaxis.set_major_formatter(lambda val, _: _k(val))
        fig.suptitle(note or "A/B: SPX at entry above its 10-day vs its 20-day SMA, backtester rules (each spread a trade)",
                     x=0.01, ha="left")
        fig.tight_layout()
    return fig


TRANCHE_MIN_SESSIONS = 30   # a tranche pair needs this many shared sessions to get a correlation
TRANCHE_BLOCK_RHO = 0.6     # tranches in one block correlate at least this much on average


def _hhmm_minutes(label: str) -> int:
    return int(label[:2]) * 60 + int(label[3:5])


def tranche_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of daily P/L between every pair of tranches, inside the strategy.

    One value per tranche per session (the condor's P/L with the strategy's stop);
    a pair is only compared on the sessions both traded, and needs at least
    ``TRANCHE_MIN_SESSIONS`` of them. Pearson rather than rank, because the
    question is how the dollars move together, which is what drives the book's risk.
    """
    book = panel[version_mask(panel, **STRATEGY)]
    pv = book.pivot_table(index="date", columns="entry_time", values=PL, aggfunc="sum")
    corr = pv.corr(min_periods=TRANCHE_MIN_SESSIONS)
    corr.index = corr.columns = [t[:5] for t in corr.columns]
    corr.attrs["sessions"] = pv.notna().sum().rename(lambda t: t[:5])
    return corr


def tranche_corr_by_gap(corr: pd.DataFrame) -> pd.DataFrame:
    """Average correlation between two tranches by how far apart they enter (minutes, rounded to 20)."""
    names = list(corr.columns)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = corr.loc[a, b]
            if not np.isnan(rho):
                gap = int(round((_hhmm_minutes(b) - _hhmm_minutes(a)) / 20) * 20)
                rows.append({"gap_minutes": gap, "rho": rho})
    d = pd.DataFrame(rows)
    return d.groupby("gap_minutes")["rho"].agg(mean="mean", median="median", min="min", max="max", pairs="size")


def tranche_blocks(corr: pd.DataFrame, rho: float = TRANCHE_BLOCK_RHO) -> pd.DataFrame:
    """Group tranches into blocks that move together (average-linkage clustering on 1 - correlation)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    dist = (1 - corr.fillna(0)).clip(lower=0).to_numpy()
    np.fill_diagonal(dist, 0)
    labels = fcluster(linkage(squareform(dist, checks=False), method="average"), t=1 - rho, criterion="distance")
    order = {lab: k for k, lab in enumerate(dict.fromkeys(labels), start=1)}  # number blocks by time of day
    rows = []
    for lab, k in order.items():
        members = [t for t, l in zip(corr.columns, labels) if l == lab]
        sub = corr.loc[members, members].to_numpy()
        inside = sub[np.triu_indices(len(members), 1)]
        rows.append({"block": k, "from": members[0], "to": members[-1], "tranches": len(members),
                     "members": ", ".join(members),
                     "avg_rho_inside": float(np.nanmean(inside)) if len(inside) else np.nan})
    return pd.DataFrame(rows).set_index("block")


def tranche_summary(corr: pd.DataFrame) -> dict:
    """Average pairwise correlation and the effective number of independent tranches it implies."""
    n = len(corr)
    off = corr.to_numpy()[~np.eye(n, dtype=bool)]
    avg = float(np.nanmean(off))
    return {"tranches": n, "avg_pairwise_rho": avg, "effective_independent_tranches": n / (1 + (n - 1) * avg),
            "pairs_above_0.7": int((off > 0.7).sum() // 2)}


def tranche_high_pairs(corr: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    names = list(corr.columns)
    rows = [{"tranche_a": a, "tranche_b": b, "rho": corr.loc[a, b],
             "minutes_apart": _hhmm_minutes(b) - _hhmm_minutes(a)}
            for i, a in enumerate(names) for b in names[i + 1:] if corr.loc[a, b] >= threshold]
    out = pd.DataFrame(rows, columns=["tranche_a", "tranche_b", "rho", "minutes_apart"])
    return out.sort_values("rho", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. the exit: call stop
# --------------------------------------------------------------------------- #
SLIPPAGES = (0.0, EXIT_SLIPPAGE)


def put_condor_pl(panel: pd.DataFrame, spreads: pd.DataFrame, stop_pct: float | None,
                  slippage: float = EXIT_SLIPPAGE) -> tuple[pd.Series, pd.Series]:
    """Condor P/L with a put stop replayed from the chain (calls to expiry), and whether the stop fired."""
    key = pd.MultiIndex.from_frame(panel[["date", "entry_time"]])
    put, hit = br.put_side_pl(spreads, stop_pct, slippage)
    call = spreads[spreads["side"] == "call"].set_index(["date", "entry_time"])["p_l"]
    return (pd.Series(put.reindex(key).to_numpy() + call.reindex(key).to_numpy(), index=panel.index),
            pd.Series(hit.reindex(key).fillna(False).to_numpy(bool), index=panel.index))


def stop_scan_put(panel: pd.DataFrame, spreads: pd.DataFrame, cal: Calendars) -> pd.DataFrame:
    """The strategy's condors with a put-side stop at every level, at $0 and $20 slippage (the call side held to
    expiry): the same replay as ``stop_scan``, on the put spreads."""
    book = panel[version_mask(panel, **STRATEGY)]
    rows = []
    for slip in SLIPPAGES:
        for stop in [None, *br.STOP_LEVELS]:
            pl, hit = put_condor_pl(book, spreads, stop, slip)
            rows.append({"slippage": slip, "stop_pct": _fmt(stop), "put_stop_hit_rate": float(hit.mean() * 100),
                         **score_pl(pl, book["date"], cal)})
    return pd.DataFrame(rows).set_index(["slippage", "stop_pct"])


def stop_scan(panel: pd.DataFrame, spreads: pd.DataFrame, cal: Calendars) -> pd.DataFrame:
    """The strategy's condors with the call stop at every level in ``br.STOP_LEVELS``, at $0 and $20 slippage.

    Replayed the backtester's way (``backtester_rules.call_side_pl``): the stop fires when the spread's
    worst intraday loss reached it (never on the settlement print) and fills where the option chain first
    crossed it, plus the slippage and the fees; the put side is held to expiry.
    """
    book = panel[version_mask(panel, **STRATEGY)]
    rows = []
    for slip in SLIPPAGES:
        for stop in [None, *br.STOP_LEVELS]:
            pl, hit = condor_pl(book, spreads, stop, slip)
            rows.append({"slippage": slip, "stop_pct": _fmt(stop), "call_stop_hit_rate": float(hit.mean() * 100),
                         **score_pl(pl, book["date"], cal)})
    return pd.DataFrame(rows).set_index(["slippage", "stop_pct"])


# --------------------------------------------------------------------------- #
# 6. the short/long ratio
# --------------------------------------------------------------------------- #
NO_RATIO_CAP = {**STRATEGY, "max_slratio": None}


def slratio_quintiles(panel: pd.DataFrame) -> pd.DataFrame:
    """Inside the strategy without its ratio cap: avg P/L by quintile of each side's short/long ratio, and per half."""
    book = panel[version_mask(panel, **NO_RATIO_CAP)]
    first = book["date"].dt.year <= SPLIT_YEAR
    rows = []
    for c, side in SL_SIDES.items():
        for i, (level, grp) in enumerate(book.groupby(pd.qcut(book[c], 5), observed=True)):
            f = first.loc[grp.index]
            rows.append({"side": side, "quintile": i + 1, "range": f"{level.left:.2f} to {level.right:.2f}",
                         "condors": len(grp), "trades": 2 * len(grp), "sessions": grp["date"].nunique(),
                         "win_rate": (grp[PL] > 0).mean() * 100, "avg_pl": grp[PL].mean(),
                         "total_pl": grp[PL].sum(), f"avg_pl_to_{SPLIT_YEAR}": grp.loc[f, PL].mean(),
                         f"avg_pl_from_{SPLIT_YEAR + 1}": grp.loc[~f, PL].mean()})
    return pd.DataFrame(rows)


def slratio_caps(panel: pd.DataFrame, cal: Calendars) -> pd.DataFrame:
    """The strategy without its ratio cap, then a maximum short/long ratio of the whole condor, at every level."""
    base = version_mask(panel, **NO_RATIO_CAP)
    rows = []
    for c, side in SL_SIDES.items():
        for cap in SL_CAPS:
            keep = base if cap is None else base & (panel[c] <= cap)
            rows.append({"side": side, "max_ratio": _fmt(cap), "kept_pct": keep.sum() / base.sum() * 100,
                         **score(panel, keep, cal)})
    cols = ["kept_pct", "trades", "total_pl", "sharpe", "max_drawdown", H1, H2]
    return pd.DataFrame(rows).set_index(["side", "max_ratio"])[cols]


SL_CONTROLS = {
    "none": [],
    "credit": ["credit"],
    "strike distance (both sides)": ["otm_points_call", "otm_points_put"],
    "entry time": ["mins_from_open"],
    "VIX9D": ["vix9d_open"],
    "credit, VIX9D, entry time": ["credit", "vix9d_open", "mins_from_open"],
}
SL_RELATED = ["credit", "otm_points_call", "otm_points_put", "mins_from_open", "vix9d_open", "vvix_prev"]


def _partial_spearman(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> float:
    """Rank correlation of ``x`` and ``y`` after regressing both ranks on the controls' ranks."""
    d = df[[x, y] + controls].dropna().rank()
    X = np.c_[np.ones(len(d)), d[controls].to_numpy()]
    rx = d[x] - X @ np.linalg.lstsq(X, d[x], rcond=None)[0]
    ry = d[y] - X @ np.linalg.lstsq(X, d[y], rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def slratio_correlation(panel: pd.DataFrame, n_boot: int = 500, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """How each side's entry short/long ratio correlates with P/L inside the strategy.

    Returns three tables:

    * ``summary`` — condor-level Spearman with P/L (with a 95% interval from
      resampling whole sessions), session-level Spearman of the day's mean ratio
      with the day's P/L, Pearson, and Spearman below / above the ratio's 60th
      percentile (the hump), with the whole book for comparison;
    * ``partial`` — Spearman with P/L after controlling for credit, strike
      distance, entry time and VIX9D: what is left of the ratio once what it
      moves with is held fixed;
    * ``related`` — the ratio's Spearman with the columns it moves with.
    """
    book = panel[version_mask(panel, **NO_RATIO_CAP)]
    rng = np.random.default_rng(seed)
    groups = book.groupby("date").indices
    days = np.array(list(groups))
    rows = []
    for name, frame in (("strategy", book), ("whole book", panel)):
        for c, side in SL_SIDES.items():
            row = {"book": name, "side": side,
                   "spearman_condor": frame[c].corr(frame[PL], method="spearman"),
                   "pearson_condor": frame[c].corr(frame[PL])}
            daily = frame.groupby("date").agg(r=(c, "mean"), pl=(PL, "sum"))
            row["spearman_session"] = daily["r"].corr(daily["pl"], method="spearman")
            cut = frame[c].quantile(0.6)
            lo, hi = frame[frame[c] <= cut], frame[frame[c] > cut]
            row.update({"p60_ratio": cut, "spearman_below_p60": lo[c].corr(lo[PL], method="spearman"),
                        "spearman_above_p60": hi[c].corr(hi[PL], method="spearman")})
            if name == "strategy":
                boots = []
                for _ in range(n_boot):
                    idx = np.concatenate([groups[d] for d in rng.choice(days, len(days))])
                    s = book.iloc[idx]
                    boots.append(s[c].corr(s[PL], method="spearman"))
                row.update({"ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5)})
            rows.append(row)
    summary = pd.DataFrame(rows).set_index(["book", "side"])
    partial = pd.DataFrame({side: {ctrl: _partial_spearman(book, c, PL, cols) for ctrl, cols in SL_CONTROLS.items()}
                            for c, side in SL_SIDES.items()})
    partial.index.name = "controlling for"
    related = book[list(SL_SIDES) + SL_RELATED].corr(method="spearman").loc[SL_RELATED, list(SL_SIDES)]
    related.columns = list(SL_SIDES.values())
    by_year = book.groupby(book["date"].dt.year).apply(
        lambda g: pd.Series({side: g[c].corr(g[PL], method="spearman") for c, side in SL_SIDES.items()}))
    return {"summary": summary, "partial": partial, "related": related, "by_year": by_year}


# --------------------------------------------------------------------------- #
# 7. remaining features
# --------------------------------------------------------------------------- #
def addon_scan(panel: pd.DataFrame, cal: Calendars, min_keep: float = 0.30) -> pd.DataFrame:
    """Each remaining candidate's best single keep-rule as a further filter on the strategy.

    Same search as ``feature_study.filter_scan`` (a min or max at each decile,
    or keep 1 / keep 0 for a flag, keeping at least ``min_keep`` of condors),
    scored on all sessions and in each half. ``attrs["noise_p95"]`` is the strict
    bar: the 95th percentile of ``N_NOISE`` random session-level features put
    through the same search.
    """
    keep_book = version_mask(panel, **STRATEGY)
    book = panel[keep_book]
    base = score(panel, keep_book, cal)
    rows = []
    for c in ADDON_FEATURES:
        x = book[c]
        if c in ADDON_FLAGS:
            candidates = [(f"{c} = 1", x == 1), (f"{c} = 0", x == 0)]
        else:
            candidates = []
            for q in np.arange(0.1, 1.0, 0.1):
                cut = x.quantile(q)
                candidates += [(f"{c} >= {cut:.4g}", x >= cut), (f"{c} <= {cut:.4g}", x <= cut)]
        best = None
        for label, k in candidates:
            if k.mean() < min_keep or k.all():
                continue
            st = score_pl(book.loc[k, PL], book.loc[k, "date"], cal)
            if best is None or st["sharpe"] > best[0]["sharpe"]:
                best = (st, label, k)
        if best is None:
            continue
        st, label, k = best
        rows.append({"feature": c, "family": ADDON_FEATURES[c][0], "description": ADDON_FEATURES[c][1],
                     "rule": label, "kept_pct": k.mean() * 100, "sharpe_gain": st["sharpe"] - base["sharpe"],
                     f"gain_to_{SPLIT_YEAR}": st[H1] - base[H1], f"gain_from_{SPLIT_YEAR + 1}": st[H2] - base[H2],
                     "total_pl_change": st["total_pl"] - base["total_pl"],
                     "kept_avg_pl": book.loc[k, PL].mean(), "dropped_avg_pl": book.loc[~k, PL].mean()})
    out = pd.DataFrame(rows).set_index("feature").sort_values("sharpe_gain", ascending=False)

    rng = np.random.default_rng(SEED + 1)
    days = book["date"].unique()
    pl, d = book[PL], book["date"]
    gains = []
    for _ in range(N_NOISE):
        z = d.map(dict(zip(days, rng.standard_normal(len(days))))).to_numpy()
        top = -np.inf
        for q in np.arange(0.1, 1.0, 0.1):
            cut = np.quantile(z, q)
            for k in (z >= cut, z <= cut):
                if k.mean() >= min_keep:
                    top = max(top, br.sharpe(rs.daily_series(pl[k], d[k], cal.all)))
        gains.append(top - base["sharpe"])
    out.attrs.update(base=base, noise_p95=float(np.percentile(gains, 95)),
                     noise_p50=float(np.percentile(gains, 50)))
    out["beats_noise_p95"] = out["sharpe_gain"] > out.attrs["noise_p95"]
    out["better_in_both_halves"] = (out[f"gain_to_{SPLIT_YEAR}"] > 0) & (out[f"gain_from_{SPLIT_YEAR + 1}"] > 0)
    return out


def addon_candidates(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """Round-number versions of the scan's leaders, with neighbouring thresholds, to test on top of the strategy."""
    return {
        "strategy": pd.Series(True, index=panel.index),
        **{f"+ VVIX 252-day rank <= {t}": panel["vvix_rank252"] <= t for t in (25, 30, 35, 40, 50)},
        "+ condor credit <= $450": panel["credit"] <= 450,
        **{f"+ SPX gap >= {t:g}%": panel["gap_pct"] >= t for t in (-0.5, -0.25, 0.0)},
        "+ SMA5 > SMA9": panel["sma5_gt_sma9"] == 1,
    }


def addon_test(panel: pd.DataFrame, cal: Calendars) -> pd.DataFrame:
    """Each round-number add-on on top of the strategy, and its change from it."""
    base_keep = version_mask(panel, **STRATEGY)
    rows = {name: score(panel, base_keep & rule.fillna(False), cal) for name, rule in addon_candidates(panel).items()}
    out = pd.DataFrame(rows).T[["sessions", "total_pl", "sharpe", "max_drawdown", H1, H2]]
    for c in ("sharpe", H1, H2, "total_pl"):
        out[f"{c}_change"] = out[c] - out.loc["strategy", c]
    out["better_in_both_halves"] = (out[f"{H1}_change"] > 0) & (out[f"{H2}_change"] > 0)
    return out


# --------------------------------------------------------------------------- #
# figures — the dataviz reference palette (validated: blue/orange/aqua all-pairs)
# --------------------------------------------------------------------------- #
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
LIGHT_BLUE, GREY, INK, INK_2 = "#9ec5f4", "#b9b8b2", "#0b0b0b", "#52514e"
HALF_STYLE = {H1: dict(color=ORANGE, lw=1.4, ls="-", marker="s", ms=4, label=f"2022-{SPLIT_YEAR}"),
              H2: dict(color=AQUA, lw=1.4, ls="--", marker="^", ms=4, label=f"{SPLIT_YEAR + 1}-2026")}
STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#ebeae6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "font.size": 9, "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2,
    "figure.titlesize": 12, "figure.titleweight": "semibold",
}


def _mark(ax, x, label: str, y_frac: float = 0.97) -> None:
    """Shade the strategy's value on a categorical x-axis and label it in ink, not series colour."""
    ax.axvspan(x - 0.45, x + 0.45, color=BLUE, alpha=0.08, lw=0)
    ax.text(x, y_frac, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5,
            color=INK_2, fontweight="semibold")


def _k(v: float) -> str:
    return f"{'-' if v < 0 else ''}${abs(v) / 1000:,.0f}k"


def fig_profile(prof: pd.DataFrame, feature: str, note: str | None = None):
    """P/L across one filter's feature (that filter off, the rest on), and why: the sides and the move after entry."""
    name = PROFILES[feature][0]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
        x = np.arange(len(prof))
        kept = prof["in_strategy"].to_numpy()

        def shade(ax):
            for xi in np.flatnonzero(kept):
                ax.axvspan(xi - 0.5, xi + 0.5, color=BLUE, alpha=0.06, lw=0)
            ax.axhline(0, color=INK_2, lw=0.7)
            ax.set_xticks(x, prof.index, rotation=40, ha="right", fontsize=8)

        ax = axes[0]
        ax.bar(x, prof["avg_pl"], width=0.66, color=[BLUE if v >= 0 else RED for v in prof["avg_pl"]],
               alpha=1.0)
        ax.errorbar(x, prof["avg_pl"], yerr=[prof["avg_pl"] - prof["ci_low"], prof["ci_high"] - prof["avg_pl"]],
                    fmt="none", ecolor=INK_2, lw=0.8, capsize=2)
        ax.set_title("Avg P/L per condor (95% CI)")
        ax.set_ylabel("$ per condor")
        ax = axes[1]
        ax.bar(x - 0.19, prof["avg_pl_call"], width=0.36, color=BLUE, label="call side")
        ax.bar(x + 0.19, prof["avg_pl_put"], width=0.36, color=ORANGE, label="put side")
        ax.set_title("Which side earns or loses")
        ax.set_ylabel("$ per condor")
        ax.legend(loc="best")
        ax = axes[2]
        col, title, ylabel = MECHANISM[feature]
        ax.bar(x, prof[col], width=0.66, color=LIGHT_BLUE)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax = axes[3]
        ax.bar(x, prof["total_pl"], width=0.66, color=[BLUE if v >= 0 else RED for v in prof["total_pl"]])
        ax.yaxis.set_major_formatter(lambda v, _: _k(v))
        ax.set_title("Total P/L")
        for a in axes:
            shade(a)
        kept_note = "shaded = kept by the strategy" if kept.any() else "not a filter"
        fig.suptitle(note or f"{name}: P/L across its range, the rest of the strategy on ({kept_note})",
                     x=0.01, ha="left")
        fig.tight_layout()
    return fig


def fig_build_up(bu: pd.DataFrame):
    """Sharpe, total P/L and max drawdown as the filters are added one at a time."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        x = np.arange(len(bu))
        for ax, col, title, fmt in ((axes[0], "sharpe", "Sharpe", "{:.2f}"),
                                    (axes[1], "total_pl", "Total P/L", None),
                                    (axes[2], "max_drawdown", "Max drawdown", None)):
            vals = bu[col].astype(float)
            last = len(vals) - 1
            ax.bar(x, vals, width=0.62,
                   color=[BLUE if i == last else "#9cc2ee" if i == last - 1
                          else (RED if v < 0 and col != "max_drawdown" else GREY) for i, v in enumerate(vals)])
            for xi, v in zip(x, vals):
                ax.text(xi, v, fmt.format(v) if fmt else _k(v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8, color=INK_2)
            ax.axhline(0, color=INK_2, lw=0.7)
            ax.set_xticks(x, bu.index, rotation=30, ha="right")
            ax.set_title(title)
            if col != "sharpe":
                ax.yaxis.set_major_formatter(lambda v, _: _k(v))
        fig.suptitle("Where the edge comes from: the raw book held to expiry, each filter added in turn, then the "
                     "call stop (red = losing, light blue = every entry filter held to expiry, blue = the strategy)", x=0.01, ha="left")
        fig.tight_layout()
    return fig


def fig_sweeps(sw: dict[str, pd.DataFrame], title: str | None = None):
    """Each threshold moved with the rest fixed: Sharpe (all sessions and each half), then total P/L.

    Pass a subset of ``sweeps`` (e.g. just ``max_move``) to draw one threshold on its own.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, len(sw), figsize=(max(10.0, 4.5 * len(sw) + 0.5), 7), squeeze=False,
                                 gridspec_kw={"height_ratios": [1.3, 1]})
        for j, (name, t) in enumerate(sw.items()):
            t = t[t.index != "off"]  # the charts show thresholds only; "off" stays in the tables
            x = np.arange(len(t))
            i = int(np.flatnonzero(t["is_strategy"])[0])
            ax = axes[0, j]
            ax.plot(x, t["sharpe"].astype(float), color=BLUE, marker="o", ms=5.5, label="all sessions")
            for h, style in HALF_STYLE.items():
                ax.plot(x, t[h].astype(float), **style)
            _mark(ax, i, "")
            ax.set_title(f"{SWEEP_LABELS[name]}\nstrategy {t.index[i]}: Sharpe {t['sharpe'].iloc[i]:.2f}")
            ax.set_xticks(x, [])
            ax2 = axes[1, j]
            ax2.bar(x, t["total_pl"].astype(float), color=[BLUE if s else LIGHT_BLUE for s in t["is_strategy"]],
                    width=0.66)
            ax2.yaxis.set_major_formatter(lambda v, _: _k(v))
            ax2.set_xticks(x, t.index, rotation=45)
            _mark(ax2, i, "")
            if j == 0:
                ax.set_ylabel("Sharpe")
                ax2.set_ylabel("total P/L")
                ax.legend(loc="lower left")
        fig.suptitle(title or "Each threshold moved one step at a time, the rest of the strategy fixed", x=0.01, ha="left")
        fig.tight_layout()
    return fig


def fig_trend(tr: pd.DataFrame, note: str | None = None):
    """Sharpe of each trend rule, overall and in each half; the strategy's rule marked."""
    with plt.rc_context({**STYLE, "axes.grid.axis": "x"}):
        r = tr.sort_values("sharpe")
        fig, ax = plt.subplots(figsize=(10, 5.2))
        y = np.arange(len(r))
        for yi, (lo, hi) in enumerate(zip(r[[H1, H2]].min(1), r[[H1, H2]].max(1))):
            ax.plot([lo, hi], [yi, yi], color=GREY, lw=1.2, zorder=1)
        ax.scatter(r[H1], y, color=ORANGE, marker="s", s=30, zorder=2, label=HALF_STYLE[H1]["label"])
        ax.scatter(r[H2], y, color=AQUA, marker="^", s=36, zorder=2, label=HALF_STYLE[H2]["label"])
        ax.scatter(r["sharpe"], y, color=BLUE, s=70, zorder=3, edgecolor="white", lw=1.5, label="all sessions")
        for yi, v in zip(y, r["sharpe"]):
            ax.text(v, yi + 0.3, f"{v:.2f}", ha="center", fontsize=7.5, color=INK_2)
        ax.set_yticks(y, r.index)
        for lab in ax.get_yticklabels():
            if lab.get_text() == STRATEGY_TREND:
                lab.set_fontweight("bold")
                lab.set_color(INK)
        ax.axvline(tr.loc[STRATEGY_TREND, "sharpe"], color=BLUE, lw=0.8, ls=":")
        ax.set_xlabel("Sharpe")
        ax.set_title(note or "Trend rule, the rest of the strategy fixed (bold = the strategy's rule)")
        ax.legend(loc="lower right")
        fig.tight_layout()
    return fig


def fig_tranche_corr(corr: pd.DataFrame, by_gap: pd.DataFrame, note: str | None = None):
    """Tranche-by-tranche P/L correlation (lower triangle) and how it fades with time between entries."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", [RED, "#f0efec", BLUE])
    sub = corr.iloc[1:, :-1]
    vals = sub.to_numpy(float)
    mask = np.triu(np.ones_like(vals, bool), k=1)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(1, 2, figsize=(19, 8), gridspec_kw={"width_ratios": [1.35, 1]})
        ax = axes[0]
        im = ax.imshow(np.where(mask, np.nan, vals), cmap=cmap, vmin=-1, vmax=1)
        for (i, j), v in np.ndenumerate(vals):
            if not mask[i, j] and not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 else INK)
        ax.set_xticks(range(sub.shape[1]), sub.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(sub)), sub.index, fontsize=8)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        fig.colorbar(im, ax=ax, shrink=0.7, label="correlation of daily P/L")
        ax.set_title("Daily P/L correlation between tranches")
    with plt.rc_context(STYLE):
        ax = axes[1]
        x = by_gap.index / 60
        ax.fill_between(x, by_gap["min"], by_gap["max"], color=LIGHT_BLUE, alpha=0.5, lw=0, label="range across pairs")
        ax.plot(x, by_gap["mean"], color=BLUE, marker="o", ms=5, label="average")
        ax.axhline(0, color=INK_2, lw=0.7)
        ax.set_xlabel("hours between the two entries")
        ax.set_ylabel("correlation of daily P/L")
        ax.set_title("How the link fades with time between entries")
        ax.legend(loc="upper right")
        fig.suptitle(note or "How correlated the tranches are (inside the strategy, held to expiry)", x=0.01, ha="left")
        fig.tight_layout()
    return fig


def fig_calendar(wd: pd.DataFrame, tt: pd.DataFrame | None = None, note: str | None = None):
    """Session P/L per weekday and condor P/L per tranche, inside the other filters; exclusions marked."""
    with plt.rc_context(STYLE):
        panels = [(wd, "avg_session_pl", [d[:3] for d in wd.index], "Thursday", "Avg P/L per session, by weekday",
                   "$ per session")]
        if tt is not None:
            panels.append((tt, "avg_pl", list(tt.index), None, "Avg P/L per condor, by tranche", "$ per condor"))
        fig, axes = plt.subplots(1, len(panels), figsize=(17, 4.6) if tt is not None else (8, 4.4), squeeze=False,
                                 gridspec_kw={"width_ratios": [1, 2.4][:len(panels)]})
        axes = axes[0]
        for ax, (t, col, labels, excluded, title, ylabel) in zip(axes, panels):
            x = np.arange(len(t))
            out = [i == excluded for i in t.index]
            ab = [t is tt and i in AB_DROPPED for i in t.index]
            ax.bar(x, t[col], width=0.66, color=[RED if o else ORANGE if a else BLUE for o, a in zip(out, ab)],
                   hatch=["///" if o or a else "" for o, a in zip(out, ab)], edgecolor="white", lw=0)
            if t is tt:
                ax.legend([plt.Rectangle((0, 0), 1, 1, color=BLUE),
                           plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE, hatch="///", edgecolor="white")],
                          ["traded", "dropped"],
                          loc="upper left", fontsize=8)
            ax.errorbar(x, t[col], yerr=[t[col] - t["ci_low"], t["ci_high"] - t[col]], fmt="none",
                        ecolor=INK_2, lw=0.8, capsize=2)
            ax.axhline(0, color=INK_2, lw=0.7)
            ax.set_xticks(x, labels, rotation=0 if t is wd else 90)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
        fig.suptitle(note or "Calendar, inside the other filters (whiskers: 95% session-bootstrap CI; "
                     "hatched = not traded by the strategy)", x=0.01, ha="left")
        fig.tight_layout()
    return fig


def fig_stop(sc: pd.DataFrame, note: str | None = None, side: str = "call"):
    """Sharpe and total P/L by call-stop level, at $20 exit slippage and $0 for reference."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
        no_stop = sc.loc[(EXIT_SLIPPAGE, "off")]
        sc = sc[sc.index.get_level_values("stop_pct") != "off"]  # stop levels on the axis; "no stop" as a line
        labels = list(sc.loc[EXIT_SLIPPAGE].index)
        x = np.arange(len(labels))
        for ax, col, title in ((axes[0], "sharpe", "Sharpe"), (axes[1], "total_pl", "Total P/L")):
            ax.plot(x, sc.loc[0.0, col], color=GREY, lw=1.4, ls="--", marker="o", ms=4, label="$0 slippage (reference)")
            ax.plot(x, sc.loc[EXIT_SLIPPAGE, col], color=BLUE, marker="o", ms=5.5,
                    label=f"${EXIT_SLIPPAGE / 100:.2f}/contract exit slippage")
            ax.axhline(no_stop[col], color=INK, ls=":", lw=1.3,
                       label=f"no stop ({no_stop[col]:.2f})" if col == "sharpe" else f"no stop ({_k(no_stop[col])})")
            ax.set_xticks(x, labels)
            ax.set_xlabel(f"{side}-side stop, % of the {side} credit")
            ax.set_title(f"{title} (the strategy has no stop: {no_stop[col]:.2f})" if col == "sharpe"
                         else f"{title} (the strategy has no stop: {_k(no_stop[col])})")
            if col == "total_pl":
                ax.yaxis.set_major_formatter(lambda val, _: _k(val))
        axes[0].legend(loc="lower center")
        axes[1].legend(loc="lower right")
        fig.suptitle(note or f"The exit: call-stop level on the strategy's condors ({int(no_stop['trades']):,} trades at every "
                     "level; the stop changes how they close, not which are opened)", x=0.01, ha="left")
        fig.tight_layout()
    return fig


def fig_slratio(q: pd.DataFrame, caps: pd.DataFrame, note: str | None = None):
    """Avg P/L by quintile of the condor's short/long ratio, and Sharpe as a maximum ratio is tightened."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
        for ax, side in zip(axes[:1], SL_SIDES.values()):
            d = q[q["side"] == side]
            x = np.arange(len(d))
            for off, col, color, label in ((-0.27, "avg_pl", BLUE, "all sessions"),
                                           (0, f"avg_pl_to_{SPLIT_YEAR}", ORANGE, HALF_STYLE[H1]["label"]),
                                           (0.27, f"avg_pl_from_{SPLIT_YEAR + 1}", AQUA, HALF_STYLE[H2]["label"])):
                ax.bar(x + off, d[col], width=0.25, color=color, label=label)
            ax.axhline(0, color=INK_2, lw=0.7)
            ax.set_xticks(x, [f"Q{i}\n{r}" for i, r in zip(d["quintile"], d["range"])], fontsize=7.5)
            ax.set_ylabel("avg P/L per condor ($)")
            ax.set_title(f"Short/long ratio, {side}: P/L by quintile")
            ax.legend(loc="upper left")
        ax = axes[1]
        side = next(iter(SL_SIDES.values()))
        d = caps.loc[side]
        d = d[d.index != "off"]
        ax.plot(d.index, d["sharpe"], color=BLUE, marker="o", ms=5, label=f"max ratio, {side}")
        ax.axhline(caps.loc[(side, "off"), "sharpe"], color=INK_2, ls=":", lw=1, label="no ratio cap")
        ax.axvline(list(d.index).index(_fmt(STRATEGY["max_slratio"])), color=BLUE,
                   lw=0.8, ls="--", label=f"strategy: condor <= {MAX_SLRATIO:g}")
        ax.set_xlabel("maximum entry short/long ratio")
        ax.set_ylabel("Sharpe")
        ax.set_title("Sharpe as a maximum ratio is tightened")
        ax.legend(loc="lower right")
        if note:
            fig.suptitle(note, x=0.01, ha="left")
        fig.tight_layout()
    return fig


# The strategy's own filters, as leave-one-out rows -> the label used in the ranking.
FILTER_LABELS = {
    "without move <= 0%": "ON: move up max 0% (in strategy)",
    "without the band": "ON: VIX9D/VIX 0.87-1.02 band (in strategy)",
    "without SPX > SMA10": "ON: above 10-day SMA (in strategy)",
    "trading Thursdays": "ON: no Thursdays (in strategy)",
    "keeping the morning tranches": "ON: forward-selected tranches (in strategy)",
    f"without short/long ratio <= {MAX_SLRATIO:g}": f"ON: max short/long {MAX_SLRATIO:g}, whole condor (in strategy)",
}


def feature_ranking(loo: pd.DataFrame, scan: pd.DataFrame) -> pd.DataFrame:
    """Every feature on one Sharpe scale, ranked.

    For the strategy's own filters the value is the Sharpe **lost** when the filter
    is removed (leave-one-out); for every other candidate it is the Sharpe
    **gained** by its best single cut on top of the strategy (``addon_scan``). Both
    are measured against the same strategy on all sessions, so they share a scale;
    each is also split by half. A candidate's gain is from a searched threshold, so
    it has to clear the strict noise bar; a filter's contribution is at its set
    value, with no search.
    """
    strat = loo.loc["strategy"]
    rows = []
    for key, label in FILTER_LABELS.items():
        r = loo.loc[key]
        rows.append({"feature": label, "role": "in strategy", "rule": "as set",
                     "sharpe_contribution": strat["sharpe"] - r["sharpe"],
                     f"contribution_to_{SPLIT_YEAR}": strat[H1] - r[H1],
                     f"contribution_from_{SPLIT_YEAR + 1}": strat[H2] - r[H2],
                     "total_pl_effect": strat["total_pl"] - r["total_pl"]})
    for c, r in scan.iterrows():
        rows.append({"feature": ADDON_FEATURES[c][1], "role": "noise control" if c == "noise" else "candidate",
                     "rule": r["rule"],
                     "sharpe_contribution": r["sharpe_gain"],
                     f"contribution_to_{SPLIT_YEAR}": r[f"gain_to_{SPLIT_YEAR}"],
                     f"contribution_from_{SPLIT_YEAR + 1}": r[f"gain_from_{SPLIT_YEAR + 1}"],
                     "total_pl_effect": r["total_pl_change"]})
    out = pd.DataFrame(rows)
    out["holds_in_both_halves"] = (out[f"contribution_to_{SPLIT_YEAR}"] > 0) & (
        out[f"contribution_from_{SPLIT_YEAR + 1}"] > 0)
    bar = scan.attrs["noise_p95"]
    # A filter is set, not searched, so the noise bar doesn't apply to it; a candidate's cut is searched.
    in_strategy = out["role"] == "in strategy"
    helps = out["sharpe_contribution"] > 0
    out["verdict"] = np.select(
        [out["role"] == "noise control",
         in_strategy & helps & out["holds_in_both_halves"],
         in_strategy & helps,
         in_strategy,
         (out["sharpe_contribution"] > bar) & out["holds_in_both_halves"],
         helps & out["holds_in_both_halves"]],
        ["noise: what luck alone scores", "filter: pays in both halves", "filter: pays in one half only", "filter: no gain",
         "candidate: beats the noise bar, both halves", "candidate: under the noise bar"],
        "candidate: not useful")
    out = out.sort_values("sharpe_contribution", ascending=False).reset_index(drop=True)
    out.index = out.index + 1
    out.index.name = "rank"
    out.attrs["noise_p95"] = bar
    return out


def fig_ranking(rank: pd.DataFrame):
    """All features on one Sharpe scale: the strategy's filters (Sharpe lost if removed) and candidates."""
    with plt.rc_context({**STYLE, "axes.grid.axis": "x"}):
        r = rank.iloc[::-1]
        fig, ax = plt.subplots(figsize=(10.5, 0.3 * len(r) + 1.8))
        colors = [BLUE if role == "in strategy" else GREY if role == "noise control"
                  else (ORANGE if both else "#f5b99f") for role, both in zip(r["role"], r["holds_in_both_halves"])]
        ax.barh(r["feature"], r["sharpe_contribution"], color=colors, height=0.66)
        for yi, v in enumerate(r["sharpe_contribution"]):
            ax.text(v, yi, f" {v:+.2f}" if v >= 0 else f"{v:+.2f} ", va="center",
                    ha="left" if v >= 0 else "right", fontsize=7.5, color=INK_2)
        for lab in ax.get_yticklabels():
            if "(in strategy)" in lab.get_text():
                lab.set_fontweight("bold")
                lab.set_color(INK)
        ax.axvline(0, color=INK_2, lw=0.7)
        ax.axvline(rank.attrs["noise_p95"], color=RED, ls="--", lw=1.4)
        ax.text(rank.attrs["noise_p95"], 1.0, f" noise bar +{rank.attrs['noise_p95']:.2f}",
                transform=ax.get_xaxis_transform(), color=INK_2, fontsize=8, va="bottom")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, "#f5b99f", GREY)]
        ax.legend(handles, ["ON in the strategy: Sharpe lost if removed (bold)",
                            "candidate: best-cut gain, holds in both halves",
                            "candidate: best-cut gain, one half only",
                            "noise feature (random number per session)"], loc="lower right")
        ax.set_xlabel("Sharpe contribution (vs the strategy, all 998 sessions)")
        ax.set_title("Feature ranking")
        fig.tight_layout()
    return fig


# The column behind each of the strategy's own filters in the ranking (calendar and stop rows have none).
FILTER_COLUMNS = {
    "ON: move up max 0% (in strategy)": "spx_move_pct",
    "ON: VIX9D/VIX 0.87-1.02 band (in strategy)": TS_9D_30,
    "ON: above 10-day SMA (in strategy)": "spx_vs_sma10_pct",
}
SHORT_NAMES = {
    "spx_move_pct": "move since open", TS_9D_30: "VIX9D/VIX at entry", "spx_vs_sma20_pct": "SPX vs SMA20",
    "slratio_call": "S/L ratio, call", "slratio_put": "S/L ratio, put", "slratio_condor": "S/L ratio, condor",
    "spx_vs_sma10_pct": "SPX vs SMA10",
    "vix9d_open": "VIX9D", "vix30_open": "VIX", "vix1d_open": "VIX1D", "sma5_gt_sma9": "SMA5 > SMA9",
    "ivr90": "IV rank 90", "vvix_rank252": "VVIX rank", "credit": "credit", "gap_pct": "gap",
    "ret5_prev_pct": "5-day return", "rv10_prev": "10-day realized vol", "prior_day_range_pct": "prior-day range",
    "range_so_far": "range so far", "dd_from_hi": "below intraday high", "ts_overnight_chg": "VIX9D/VIX30 overnight",
    "spx_off_20d_high_pct": "below 20-day high", "up_days_5": "up days (5)", "vix_vs_sma20_pct": "VIX vs its SMA20",
    "vix_5d_chg_pct": "VIX 5-day change", "ts_5d_chg": "VIX9D/VIX30 5-day change", "ts_1d_30_open": "VIX1D/VIX30",
    "skew_credit": "put-call credit skew", "ret_last_tranche": "last 20 min move", "opex_week": "OPEX week",
    "vix9d_gap_pct": "VIX9D gap", "vix9d_gap_band": "VIX9D gap band", "month_end": "month end",
    "spx_entry_over_sma5": "SPX at entry vs SMA5", "atr20_pct": "ATR20", "ivrv_vix9d_pk1": "VIX9D / Parkinson vol",
    "vrp20": "VRP (VIX - RV20)", "day_efficiency_prev": "prior-day efficiency", "implied_vs_range5": "implied / 5-day range",
    "noise": "noise", PL: "P/L",
}


def feature_correlation(panel: pd.DataFrame, rank: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between the ranked features (in ranking order) and P/L, inside the strategy.

    The strategy's own filter features are range-restricted inside the strategy
    (every kept condor has move <= 0%, for example), so their correlations are the
    ones that remain within the traded range. The short/long ratio is added on both
    sides for the ratio-cap filter. Condor level; the noise feature is the control.
    """
    by_desc = {v[1]: k for k, v in ADDON_FEATURES.items()}
    cols = []
    for label in rank["feature"]:
        col = FILTER_COLUMNS.get(label) or by_desc.get(label)
        if label.startswith("ON: max short/long"):
            cols += ["slratio_condor"]
        elif col is not None:
            cols.append(col)
    cols = list(dict.fromkeys(cols)) + [PL]
    book = panel[version_mask(panel, **STRATEGY)]
    corr = book[cols].astype(float).corr(method="spearman")
    corr.index = corr.columns = [SHORT_NAMES.get(c, c) for c in cols]
    return corr


def redundant_pairs(corr: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """Feature pairs whose rank correlation is at least ``threshold`` in size: they carry the same information."""
    names = [c for c in corr.columns if c != "P/L"]
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = corr.loc[a, b]
            if abs(rho) >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "spearman": rho,
                             "corr_with_pl_a": corr.loc[a, "P/L"], "corr_with_pl_b": corr.loc[b, "P/L"]})
    return pd.DataFrame(rows).sort_values("spearman", key=abs, ascending=False).reset_index(drop=True)


def fig_feature_corr(corr: pd.DataFrame, top: int | None = 24):
    """Lower-triangle heatmap of the ranked features' rank correlations (diverging, neutral midpoint).

    Only the ``top`` highest-ranked features (plus P/L) are drawn, to keep the cells readable;
    the redundant-pairs table covers them all.
    """
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", [RED, "#f0efec", BLUE])
    if top is not None and len(corr) - 1 > top:
        # The strategy's own filter features always stay; the best-ranked candidates fill the rest.
        own = {SHORT_NAMES[c] for c in (*FILTER_COLUMNS.values(), "slratio_condor")}
        names = [c for c in corr.columns if c != "P/L"]
        room = top - sum(c in own for c in names)
        others = [c for c in names if c not in own][:room]
        keep = [c for c in names if c in own or c in others] + ["P/L"]
        corr = corr.loc[keep, keep]
    # Strictly lower triangle: rows from the 2nd feature down, columns up to the second-last (the diagonal is 1).
    sub = corr.iloc[1:, :-1]
    vals = sub.to_numpy(float)
    mask = np.triu(np.ones_like(vals, bool), k=1)
    shown = np.where(mask, np.nan, vals)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        n = len(sub)
        fig, ax = plt.subplots(figsize=(0.62 * n + 3, 0.55 * n + 2))
        im = ax.imshow(shown, cmap=cmap, vmin=-1, vmax=1)
        for (i, j), v in np.ndenumerate(vals):
            if not mask[i, j]:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else INK)
        ax.set_xticks(range(sub.shape[1]), sub.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(n), sub.index, fontsize=8)
        for lab in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            if lab.get_text() == "P/L":
                lab.set_fontweight("bold")
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman rank correlation")
        ax.set_title(f"How the top {len(corr) - 1} ranked features move together inside the strategy "
                     "(ranking order; last row = P/L)")
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
@dataclass
class FinalStudy:
    panel: pd.DataFrame
    spreads: pd.DataFrame
    build_up: pd.DataFrame
    leave_one_out: pd.DataFrame
    sweeps: dict[str, pd.DataFrame]
    sweep_verdict: pd.DataFrame
    trend: pd.DataFrame
    weekdays: pd.DataFrame
    tranches: pd.DataFrame
    stops: pd.DataFrame
    stops_put: pd.DataFrame
    sl_quintiles: pd.DataFrame
    sl_caps: pd.DataFrame
    sl_corr: dict[str, pd.DataFrame]
    addon_scan: pd.DataFrame
    addon_test: pd.DataFrame
    ranking: pd.DataFrame
    feature_corr: pd.DataFrame
    tranche_corr: pd.DataFrame
    tranche_gap: pd.DataFrame
    tranche_blocks: pd.DataFrame
    tranche_pairs: pd.DataFrame
    tranche_summary: dict
    redundant_pairs: pd.DataFrame
    profiles: dict[str, pd.DataFrame]
    regime: pd.DataFrame
    tranche_forward: pd.DataFrame
    tranche_walk_forward: pd.DataFrame
    sma_ab: pd.DataFrame


def compute_final_study(trades_csv: str | Path, term_structure_csv: str | Path | None = None) -> FinalStudy:
    """Every table in the study, computed once; nothing is written."""
    panel, spreads = build_final_panel(trades_csv, term_structure_csv)
    cal = Calendars.of(panel)
    sw = sweeps(panel, cal)
    loo, scan = leave_one_out(panel, cal), addon_scan(panel, cal)
    return FinalStudy(
        panel=panel, spreads=spreads, build_up=build_up(panel, cal), leave_one_out=loo,
        sweeps=sw, sweep_verdict=sweep_verdict(sw), trend=trend_swap(panel, cal),
        weekdays=weekday_table(panel), tranches=tranche_table(panel, cal), stops=stop_scan(panel, spreads, cal),
        stops_put=stop_scan_put(panel, spreads, cal),
        sl_quintiles=slratio_quintiles(panel), sl_caps=slratio_caps(panel, cal), sl_corr=slratio_correlation(panel),
        addon_scan=scan, addon_test=addon_test(panel, cal), ranking=feature_ranking(loo, scan),
        feature_corr=feature_correlation(panel, feature_ranking(loo, scan)),
        redundant_pairs=redundant_pairs(feature_correlation(panel, feature_ranking(loo, scan))),
        tranche_corr=tranche_correlation(panel), tranche_gap=tranche_corr_by_gap(tranche_correlation(panel)),
        tranche_blocks=tranche_blocks(tranche_correlation(panel)),
        tranche_pairs=tranche_high_pairs(tranche_correlation(panel)),
        tranche_summary=tranche_summary(tranche_correlation(panel)),
        profiles={f: feature_profile(panel, f) for f in PROFILES}, regime=sma_regime(panel),
        tranche_forward=tranche_forward_selection(panel, cal),
        tranche_walk_forward=tranche_walk_forward(panel, cal), sma_ab=sma_ab(panel, spreads, cal),
    )


def figures(s: FinalStudy) -> dict[str, plt.Figure]:
    return {"01_build_up": fig_build_up(s.build_up),
            "02_term_structure": fig_profile(s.profiles[TS_9D_30], TS_9D_30),
            "03_max_move_up": fig_profile(s.profiles["spx_move_pct"], "spx_move_pct"),
            "04_spx_vs_sma10": fig_profile(s.profiles["spx_vs_sma10_pct"], "spx_vs_sma10_pct"),
            "05_trend_rules": fig_trend(s.trend), "06_thresholds": fig_sweeps(s.sweeps),
            "07_calendar": fig_calendar(s.weekdays, s.tranches), "08_call_stop": fig_stop(s.stops),
            "09_short_long_ratio": fig_slratio(s.sl_quintiles, s.sl_caps),
            "10_feature_ranking": fig_ranking(s.ranking),
            "14_feature_correlation": fig_feature_corr(s.feature_corr),
            "15_tranche_correlation": fig_tranche_corr(s.tranche_corr, s.tranche_gap),
            "15c_tranche_forward": fig_tranche_forward(s.tranche_forward, s.tranche_walk_forward),
            "05b_sma10_vs_sma20": fig_sma_ab(s.sma_ab),
}


def run_final_study(trades_csv: str | Path, reports_root: Path = rs.REPORTS_ROOT,
                    term_structure_csv: str | Path | None = None) -> FinalStudy:
    """Compute the study and write every table and figure under ``reports_root/STUDY_NAME``."""
    out = reports_root / STUDY_NAME
    tables, figs = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    s = compute_final_study(trades_csv, term_structure_csv)
    s.build_up.to_csv(tables / "build_up.csv")
    s.leave_one_out.to_csv(tables / "leave_one_out.csv")
    for name, t in s.sweeps.items():
        t.to_csv(tables / f"sweep_{name}.csv")
    s.sweep_verdict.to_csv(tables / "sweep_verdict.csv")
    s.trend.to_csv(tables / "trend_rule.csv")
    s.weekdays.to_csv(tables / "weekdays.csv")
    s.tranches.to_csv(tables / "tranches.csv")
    s.stops.to_csv(tables / "call_stop_scan.csv")
    s.sl_quintiles.to_csv(tables / "slratio_quintiles.csv", index=False)
    s.sl_caps.to_csv(tables / "slratio_caps.csv")
    for name, t in s.sl_corr.items():
        t.to_csv(tables / f"slratio_correlation_{name}.csv")
    s.addon_scan.to_csv(tables / "remaining_features_scan.csv")
    s.addon_test.to_csv(tables / "remaining_features_test.csv")
    s.ranking.to_csv(tables / "feature_ranking.csv")
    s.feature_corr.to_csv(tables / "feature_correlation.csv")
    s.redundant_pairs.to_csv(tables / "feature_redundant_pairs.csv", index=False)
    s.tranche_corr.to_csv(tables / "tranche_correlation.csv")
    s.tranche_gap.to_csv(tables / "tranche_correlation_by_gap.csv")
    s.tranche_blocks.to_csv(tables / "tranche_blocks.csv")
    s.tranche_pairs.to_csv(tables / "tranche_high_pairs.csv", index=False)
    pd.Series(s.tranche_summary).to_csv(tables / "tranche_summary.csv", header=["value"])
    s.tranche_forward.to_csv(tables / "tranche_forward_selection.csv")
    s.tranche_walk_forward.to_csv(tables / "tranche_walk_forward.csv")
    s.sma_ab.to_csv(tables / "sma10_vs_sma20_ab.csv")
    for f, t in s.profiles.items():
        t.to_csv(tables / f"profile_{f}.csv")
    s.regime.to_csv(tables / "sma10_regime_every_session.csv")
    for name, fig in figures(s).items():
        fig.savefig(figs / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    return s
