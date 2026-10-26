"""The backtester's (Option Omega) rules, reproduced from the trade export and the market data.

Each rule below was checked against a backtest log of the live strategy
(3,976 trades on the 12:33 grid): with all of them, the condors traded match
the log one for one outside the first four sessions (which need SPX closes
from before the export starts).

* **VIX9D / VIX at entry** — the open of the 1-minute bar at the entry minute
  (``data/market/vix/ohlc``), not the session open. It matches the log's
  "Opening VIX" exactly on 71% of trades, within 0.01 on the rest.
* **Short/long ratio** — one ratio for the whole condor: the sum of the two
  short prices over the sum of the two long prices. One side can be well above
  the cap.
* **Above the N-day SMA** — SPX at entry against the average of the prior
  N - 1 16:00 closes and the entry price itself.
* **Call stop** — fires when the export's worst intraday loss (``Max Loss``)
  reached the stop, but never on the 16:00 settlement print. It fills at the
  first minute the call spread cost ``1 + stop%`` times its credit to close in
  the SPXW 1-minute chain (``data/market/spx/options_ohlc``), never better than
  the stop; where the chain has no crossing (or no data), at the stop plus the
  average overshoot the chain shows at that level.
* **Sharpe** — daily P/L over the starting capital, less a daily risk-free
  rate, annualised by sqrt(252), over every session. It reproduces the
  backtester's 2.44 on its own log.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import SPX_DATA_ROOT, STRATEGY_DATA_ROOT, VIX_DATA_ROOT

VIX_OHLC = VIX_DATA_ROOT / "ohlc"
SPX_OPTIONS_OHLC = SPX_DATA_ROOT / "options_ohlc"
STOP_FILL_CACHE = STRATEGY_DATA_ROOT / "ic_strat" / "spx" / "derived" / "call_stop_fills.parquet"
PUT_STOP_FILL_CACHE = STRATEGY_DATA_ROOT / "ic_strat" / "spx" / "derived" / "put_stop_fills.parquet"
MARKET_TZ = "America/New_York"

START_CAPITAL = 200_000
RISK_FREE = 0.04
TRADING_DAYS = 252
# Stop levels (% of the credit) the chain replay is cached at.
STOP_LEVELS = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400]


# --------------------------------------------------------------------------- #
# Sharpe
# --------------------------------------------------------------------------- #
def sharpe(daily: np.ndarray | pd.Series) -> float:
    """The backtester's Sharpe of a daily P/L series (no-trade days included as zero)."""
    ex = np.asarray(daily, dtype=float) / START_CAPITAL - RISK_FREE / TRADING_DAYS
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def sharpe_rows(x: np.ndarray) -> np.ndarray:
    """:func:`sharpe` along the last axis (bootstrap draws)."""
    ex = x / START_CAPITAL - RISK_FREE / TRADING_DAYS
    return ex.mean(axis=-1) / ex.std(axis=-1, ddof=1) * np.sqrt(TRADING_DAYS)


# --------------------------------------------------------------------------- #
# entry-time inputs
# --------------------------------------------------------------------------- #
def _entry_stamps(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["date"].dt.strftime("%Y-%m-%d") + " " + frame["entry_time"]).dt.tz_localize(MARKET_TZ)


@lru_cache(maxsize=2)
def _bar_opens(symbol: str) -> pd.Series:
    files = sorted(VIX_OHLC.glob(f"{symbol}_????????_1m.parquet"))
    if not files:
        raise FileNotFoundError(f"No {symbol} 1m files under {VIX_OHLC}; copy the market data into data/market/vix/ohlc.")
    return pd.concat(pd.read_parquet(f, columns=["time", "open"]) for f in files).set_index("time")["open"]


def entry_term_structure(frame: pd.DataFrame) -> pd.Series:
    """VIX9D / VIX at each row's entry minute (the bar's open), aligned to ``frame``."""
    stamps = _entry_stamps(frame)
    v9 = _bar_opens("VIX9D").reindex(stamps).to_numpy()
    v30 = _bar_opens("VIX").reindex(stamps).to_numpy()
    return pd.Series(v9 / v30, index=frame.index)


def _leg_price(legs: str, action: str) -> float:
    return float(re.search(rf"(?:C|P) {action} ([0-9.]+)", legs).group(1))


def condor_slratio(spreads: pd.DataFrame) -> pd.Series:
    """Short/long ratio of the whole condor, indexed by (date, entry_time); NaN for an unpaired leg."""
    s = spreads.assign(sto=spreads["legs"].map(lambda l: _leg_price(l, "STO")),
                       bto=spreads["legs"].map(lambda l: _leg_price(l, "BTO")))
    g = s.groupby(["date", "entry_time"])
    ratio = g["sto"].sum() / g["bto"].sum()
    return ratio.where(g.size() == 2)


def session_closes(spreads: pd.DataFrame) -> pd.Series:
    """The 16:00 SPX close of each session (the export's ``Closing Price``)."""
    return spreads.groupby("date")["closing_price"].first().sort_index()


def spot_vs_sma_pct(spot: pd.Series, dates: pd.Series, closes: pd.Series, window: int) -> pd.Series:
    """SPX at entry against the backtester's N-day SMA (prior N - 1 closes plus the entry price), %."""
    prior = dates.map(closes.rolling(window - 1, min_periods=window - 1).sum().shift(1))
    return (spot / ((prior + spot) / window) - 1) * 100


# --------------------------------------------------------------------------- #
# call and put stops, replayed from the option chain
# --------------------------------------------------------------------------- #
def _strikes(legs: str, right: str = "C") -> tuple[float, float]:
    found = dict((k, float(s)) for s, k in re.findall(rf"(\d+(?:\.\d+)?) {right} (STO|BTO)", legs))
    return found["STO"], found["BTO"]


def _chain_fills(calls: pd.DataFrame, right: str = "C") -> pd.DataFrame:
    """For every spread of one side (``right`` "C" or "P") on a session with chain data: the fill (% loss of credit)
    at each stop level, NaN where the spread never reached that level before 16:00. Index (date, entry_time)."""
    rows = []
    for day, grp in calls.groupby("date"):
        path = SPX_OPTIONS_OHLC / f"SPXW_{day:%Y%m%d}_ohlc.parquet"
        if not path.exists():
            continue
        ch = pd.read_parquet(path, columns=["strike", "right", "quote_datetime", "close"])
        ch = ch[ch["right"] == right].pivot_table(index="quote_datetime", columns="strike", values="close")
        for r in grp.itertuples():
            ks, kl = _strikes(r.legs, right)
            if ks not in ch or kl not in ch:
                continue
            t0 = pd.Timestamp(f"{day:%Y-%m-%d} {r.entry_time}")
            cost = (ch[ks] - ch[kl]).dropna()
            cost = cost[(cost.index > t0) & (cost.index.strftime("%H:%M") < "16:00")].to_numpy()
            loss_pct = (cost * 100 - r.premium) / r.premium * 100
            out = {"date": day, "entry_time": r.entry_time}
            for lv in STOP_LEVELS:
                hit = np.flatnonzero(loss_pct >= lv)
                out[lv] = max(loss_pct[hit[0]], lv) if hit.size else np.nan
            rows.append(out)
    return pd.DataFrame(rows).set_index(["date", "entry_time"])


def call_stop_fills(spreads: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Chain-replayed stop fills for every call spread, cached under ``data/`` (rebuilt when stale)."""
    calls = spreads[spreads["side"] == "call"]
    if STOP_FILL_CACHE.exists() and not refresh:
        cached = pd.read_parquet(STOP_FILL_CACHE)
        cached.columns = [int(c) for c in cached.columns]
        if set(STOP_LEVELS) <= set(cached.columns):
            return cached
    fills = _chain_fills(calls)
    STOP_FILL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    fills.rename(columns=str).to_parquet(STOP_FILL_CACHE)
    return fills


def put_stop_fills(spreads: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Chain-replayed stop fills for every put spread, cached under ``data/`` (rebuilt when stale)."""
    puts = spreads[spreads["side"] == "put"]
    if PUT_STOP_FILL_CACHE.exists() and not refresh:
        cached = pd.read_parquet(PUT_STOP_FILL_CACHE)
        cached.columns = [int(c) for c in cached.columns]
        if set(STOP_LEVELS) <= set(cached.columns):
            return cached
    fills = _chain_fills(puts, "P")
    PUT_STOP_FILL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    fills.rename(columns=str).to_parquet(PUT_STOP_FILL_CACHE)
    return fills


def call_side_pl(spreads: pd.DataFrame, stop_pct: float | None, slippage: float,
                 fills: pd.DataFrame | None = None, side: str = "call") -> tuple[pd.Series, pd.Series]:
    """One side's P/L (``side`` "call" or "put") under a stop at ``stop_pct``% of its credit, and whether it fired;
    index (date, entry_time).

    A stopped spread loses its fill % of the credit plus ``slippage`` (capped at the wing), the opening
    fees and the closing fees ($1.94).
    """
    calls = spreads[spreads["side"] == side].set_index(["date", "entry_time"])
    if stop_pct is None:
        return calls["p_l"], pd.Series(False, index=calls.index)
    if fills is None:
        fills = call_stop_fills(spreads) if side == "call" else put_stop_fills(spreads)
    level = min(STOP_LEVELS, key=lambda lv: abs(lv - stop_pct))
    # Whether it fires: the export's worst intraday loss (the backtester's own quotes) reached the stop,
    # and not only on the settlement print. The chain's trade prints are too noisy to decide this.
    loss = -(calls["p_l"] + calls["opening_commissions_fees"])
    full = loss >= calls["wing_width"] * 100 - calls["premium"] - 0.5
    at_settle = ((-loss / calls["premium"] * 100 - calls["max_loss"]).abs() < 0.05) & ~full
    hit = ((calls["max_loss"] <= -level) & ~at_settle).to_numpy()
    # Where it fills: the chain's first crossing; without one, the average chain overshoot past the stop.
    chain = fills[level].reindex(calls.index).to_numpy()
    overshoot = (fills[level].dropna() / level).mean()
    fill_pct = np.where(np.isfinite(chain), chain, level * overshoot)
    cap = calls["wing_width"] * 100 - calls["premium"]
    stopped = -np.minimum(fill_pct / 100 * calls["premium"] + slippage, cap) - calls["opening_commissions_fees"] - 1.94
    return calls["p_l"].where(~hit, stopped), pd.Series(hit, index=calls.index)


def put_side_pl(spreads: pd.DataFrame, stop_pct: float | None, slippage: float,
                fills: pd.DataFrame | None = None) -> tuple[pd.Series, pd.Series]:
    """Put-side P/L under a stop at ``stop_pct``% of its credit, and whether it fired (see ``call_side_pl``)."""
    return call_side_pl(spreads, stop_pct, slippage, fills, side="put")
