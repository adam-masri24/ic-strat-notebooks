"""Why the IC strategy and the MO1 / MO2 condors don't move together.

Three 10-wide 0DTE SPX iron condor books:

* **IC** — this folder's strategy as traded (``final_version.STRATEGY``), rebuilt from the
  20-tranche export with the backtester's rules (``backtester_rules``);
* **MO1** — a backtest log: 35-delta shorts, entries 12:13-14:13, every weekday, no stop,
  traded when **signal W** holds: SPX at entry at least 0.5% above the average of the
  prior 5 closes;
* **MO2** — MO1 plus Parkinson20 >= 8 (a realized-volatility floor).

Correlation is measured the way the backtester's portfolio view measures it: daily P/L
over every session, days a book doesn't trade counted as zero. Everything is restricted
to the sessions the IC export covers (2022-05-16 to 2026-05-07), so the three books are
compared on one calendar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import SPX_DATA_ROOT, STRATEGY_DATA_ROOT

from . import backtester_rules as br
from . import final_version as fv
from . import regime_study as rs

MO_DIR = STRATEGY_DATA_ROOT / "ic_strat" / "spx" / "mo"
MO_FILES = {"MO1": MO_DIR / "mo1.csv", "MO2": MO_DIR / "mo2.csv"}
BOOKS = ["IC", "MO1", "MO2"]
PAIRS = [("IC", "MO1"), ("IC", "MO2"), ("MO1", "MO2")]
W_MIN = 0.5            # signal W: SPX at entry >= 0.5% above the SMA5 of prior closes
ROLL = 126             # rolling-correlation window, sessions (about six months)
N_BOOT = 2000
SEED = 0
COLORS = {"IC": fv.BLUE, "MO1": fv.ORANGE, "MO2": fv.AQUA}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _strike(legs: str, right: str, action: str) -> float:
    return float(re.search(rf"(\d+(?:\.\d+)?) {right} {action}", legs).group(1))


def _price(legs: str, right: str, action: str) -> float:
    return float(re.search(rf"{right} {action} ([0-9.]+)", legs).group(1))


def load_mo(path: str | Path, closes: pd.Series, calendar: np.ndarray) -> pd.DataFrame:
    """One row per MO condor, with the IC strategy's entry features measured on it."""
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["Date Opened"])
    d = d[d["date"].isin(calendar)].copy()
    d["entry_time"] = d["Time Opened"]
    d["spot"] = d["Opening Price"]
    d["pl"] = d["P/L"]
    d["premium"] = d["Premium"]
    d["otm_call"] = d["Legs"].map(lambda l: _strike(l, "C", "STO")) - d["spot"]
    d["otm_put"] = d["spot"] - d["Legs"].map(lambda l: _strike(l, "P", "STO"))
    sto = d["Legs"].map(lambda l: _price(l, "C", "STO") + _price(l, "P", "STO"))
    bto = d["Legs"].map(lambda l: _price(l, "C", "BTO") + _price(l, "P", "BTO"))
    d["slratio_condor"] = sto / bto
    d["spx_move_pct"] = d["Movement"] / (d["spot"] - d["Movement"]) * 100
    d[fv.TS_9D_30] = br.entry_term_structure(d)
    d["spx_vs_sma20_pct"] = br.spot_vs_sma_pct(d["spot"], d["date"], closes, 20)
    d["spx_vs_sma10_pct"] = br.spot_vs_sma_pct(d["spot"], d["date"], closes, 10)
    d["spx_vs_sma5_prior_pct"] = _vs_sma5_prior(d["spot"], d["date"], closes)
    d["dow_code"] = d["date"].dt.dayofweek
    d["max_loss_pct"] = d["Max Loss"]
    return d.reset_index(drop=True)


def _vs_sma5_prior(spot: pd.Series, dates: pd.Series, closes: pd.Series) -> pd.Series:
    """Signal W's measure: SPX at entry against the average of the prior 5 16:00 closes, %."""
    return (spot / dates.map(closes.rolling(5, min_periods=5).mean().shift(1)) - 1) * 100


@dataclass
class Books:
    ic: pd.DataFrame
    mo: dict[str, pd.DataFrame]
    daily: pd.DataFrame          # session x book daily P/L, zeros on no-trade days
    calendar: np.ndarray
    market: pd.DataFrame         # per session: SPX open, close, high, low, moves, VIX


def load_books(study: fv.FinalStudy) -> Books:
    """The IC strategy's condors (from the study) and the two MO logs, on the IC export's calendar."""
    panel = study.panel
    calendar = np.sort(panel["date"].unique())
    closes = br.session_closes(study.spreads)
    ic = panel[fv.version_mask(panel, **fv.STRATEGY)].copy()
    ic["pl"] = ic[fv.PL]
    ic["premium"] = ic["credit"]
    ic["otm_call"], ic["otm_put"] = ic["otm_points_call"], ic["otm_points_put"]
    ic["spx_vs_sma5_prior_pct"] = _vs_sma5_prior(ic["spot"], ic["date"], closes)
    mo = {k: load_mo(p, closes, calendar) for k, p in MO_FILES.items()}
    frames = {"IC": ic, **mo}
    daily = pd.DataFrame({k: f.groupby("date")["pl"].sum() for k, f in frames.items()}).reindex(calendar).fillna(0.0)
    market = market_days(study, calendar)
    for f in frames.values():
        open_ = f["date"].map(market["open"])
        f["zone_low_pct"] = ((f["spot"] - f["otm_put"]) / open_ - 1) * 100
        f["zone_center_pct"] = (f["spot"] / open_ - 1) * 100
        f["zone_high_pct"] = ((f["spot"] + f["otm_call"]) / open_ - 1) * 100
        f["close_vs_open_pct"] = f["date"].map(market["oc_pct"])
    return Books(ic=ic, mo=mo, daily=daily, calendar=calendar, market=market)


SPX_HL_CACHE = SPX_DATA_ROOT / "derived" / "spx_daily_high_low.parquet"


def spx_daily_high_low(refresh: bool = False) -> pd.DataFrame:
    """SPX's high and low for each session, from the 1-minute bars: one file at a time (low memory), cached under
    ``data/market/spx/derived/`` so later runs read one small file."""
    files = sorted((SPX_DATA_ROOT / "ohlc").glob("SPX_????????_1m.parquet"))
    if SPX_HL_CACHE.exists() and not refresh:
        hl = pd.read_parquet(SPX_HL_CACHE)
        if len(hl) >= len(files):
            return hl
    rows = []
    for f in files:
        bars = pd.read_parquet(f, columns=["high", "low"])
        rows.append((pd.Timestamp(f.stem.split("_")[1]), bars["high"].max(), bars["low"].min()))
    hl = pd.DataFrame(rows, columns=["date", "high", "low"]).set_index("date")
    SPX_HL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    hl.to_parquet(SPX_HL_CACHE)
    return hl


def market_days(study: fv.FinalStudy, calendar: np.ndarray) -> pd.DataFrame:
    """Per session: SPX's open (from the export's gap), 16:00 close, high and low (1-minute bars), the day's moves and VIX."""
    sp = study.spreads
    closes = br.session_closes(sp)
    opens = (sp["opening_price"] - sp["movement"]).groupby(sp["date"]).first()
    hl = spx_daily_high_low()
    m = pd.DataFrame({"open": opens, "close": closes}).join(hl).reindex(calendar)
    m["oc_pct"] = (m["close"] / m["open"] - 1) * 100
    m["range_pct"] = (m["high"] - m["low"]) / m["open"] * 100
    vix = study.panel.groupby("date")["vix30_open"].first()
    m["vix_open"] = vix.reindex(calendar)
    return m


def _frames(b: Books) -> dict[str, pd.DataFrame]:
    return {"IC": b.ic, **b.mo}


# --------------------------------------------------------------------------- #
# 1. the three books
# --------------------------------------------------------------------------- #
def setup_table(b: Books) -> pd.DataFrame:
    """Each book's size, results and structure on the common calendar."""
    rows = {}
    for k, f in _frames(b).items():
        d = b.daily[k].to_numpy()
        eq = np.cumsum(d)
        rows[k] = {"condors": len(f), "trades (spreads)": 2 * len(f), "sessions traded": f["date"].nunique(),
                   "total_pl": f["pl"].sum(), "sharpe": br.sharpe(d),
                   "max_drawdown": (eq - np.maximum.accumulate(eq)).min(),
                   "win_rate_condor": (f["pl"] > 0).mean() * 100, "avg_pl_condor": f["pl"].mean(),
                   "median_credit": f["premium"].median(), "median_short_call_pts": f["otm_call"].median(),
                   "median_short_put_pts": f["otm_put"].median(),
                   "entry_window": f"{f['entry_time'].min()[:5]}-{f['entry_time'].max()[:5]}",
                   "thursdays_traded": int((f["dow_code"] == fv.THURSDAY).sum() > 0)}
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. the correlation, measured like the backtester
# --------------------------------------------------------------------------- #
def _boot_corr(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    xs, ys = x[idx], y[idx]
    xs = xs - xs.mean(1, keepdims=True)
    ys = ys - ys.mean(1, keepdims=True)
    r = (xs * ys).sum(1) / np.sqrt((xs ** 2).sum(1) * (ys ** 2).sum(1))
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def correlation_table(b: Books) -> pd.DataFrame:
    """Each pair's correlation: daily (the backtester's), with a session-bootstrap interval, then weekly and monthly."""
    dly = b.daily
    wk = dly.groupby(dly.index.to_period("W")).sum()
    mo = dly.groupby(dly.index.to_period("M")).sum()
    rows = []
    for a, c in PAIRS:
        lo, hi = _boot_corr(dly[a].to_numpy(), dly[c].to_numpy())
        rows.append({"pair": f"{a} vs {c}", "daily": dly[a].corr(dly[c]), "daily_ci_low": lo, "daily_ci_high": hi,
                     "daily_rank (spearman)": dly[a].corr(dly[c], method="spearman"),
                     "weekly": wk[a].corr(wk[c]), "monthly": mo[a].corr(mo[c])})
    return pd.DataFrame(rows).set_index("pair")


def rolling_correlation(b: Books, window: int = ROLL) -> pd.DataFrame:
    """Each pair's daily-P/L correlation over a trailing ``window`` of sessions."""
    return pd.DataFrame({f"{a} vs {c}": b.daily[a].rolling(window).corr(b.daily[c]) for a, c in PAIRS})


# --------------------------------------------------------------------------- #
# 3. where the low correlation comes from: shared days x how they move on them
# --------------------------------------------------------------------------- #
def overlap_table(b: Books) -> pd.DataFrame:
    """For each pair: days each trades, days both trade, and the correlation split into its two parts.

    With daily P/L centred on its mean, corr = (shared-day co-movement) / sqrt(var_a * var_b). On a day only
    one book trades the other's P/L is zero, so that day adds variance but almost no covariance. So:

        corr ~= corr on shared days x sqrt(share of a's variance on shared days x share of b's)

    The ``approx`` column is that product; it lands close to the actual correlation.
    """
    rows = []
    dly = b.daily
    traded = dly != 0
    for a, c in PAIRS:
        both = traded[a] & traded[c]
        xa, xc = dly[a] - dly[a].mean(), dly[c] - dly[c].mean()
        share_a = (xa[both] ** 2).sum() / (xa ** 2).sum()
        share_c = (xc[both] ** 2).sum() / (xc ** 2).sum()
        shared = (xa[both] * xc[both]).sum() / np.sqrt((xa[both] ** 2).sum() * (xc[both] ** 2).sum())
        rows.append({"pair": f"{a} vs {c}", f"days {a} trades": int(traded[a].sum()),
                     f"days the other trades": int(traded[c].sum()), "days both trade": int(both.sum()),
                     "only the first": int((traded[a] & ~traded[c]).sum()),
                     "only the second": int((traded[c] & ~traded[a]).sum()),
                     "neither": int((~traded[a] & ~traded[c]).sum()),
                     "corr on shared days": dly.loc[both, a].corr(dly.loc[both, c]),
                     "first's variance on shared days %": share_a * 100,
                     "second's variance on shared days %": share_c * 100,
                     "approx = shared corr x sqrt(shares)": shared * np.sqrt(share_a * share_c),
                     "actual correlation": dly[a].corr(dly[c])})
    return pd.DataFrame(rows).set_index("pair")


# --------------------------------------------------------------------------- #
# 4. why they trade different days: opposite entry signals
# --------------------------------------------------------------------------- #
def signal_crosstab(b: Books) -> pd.DataFrame:
    """Each book's condors on the two signals that point opposite ways: SPX vs its open and SPX vs its SMA5."""
    rows = {}
    for k, f in _frames(b).items():
        m, w = f["spx_move_pct"], f["spx_vs_sma5_prior_pct"]
        rows[k] = {"condors": len(f), "SPX at/below its open (IC needs this) %": (m <= 0).mean() * 100,
                   "SPX >= 0.5% above SMA5 (MO needs this) %": (w >= W_MIN).mean() * 100,
                   "both at once %": ((m <= 0) & (w >= W_MIN)).mean() * 100,
                   "median move since open %": m.median(), "median vs SMA5 %": w.median()}
    return pd.DataFrame(rows)


IC_FILTERS = {
    f"one of IC's {len(fv.SELECTED_TRANCHES)} entry times": lambda f: f["entry_time"].str[:5].isin(fv.SELECTED_TRANCHES),
    "not a Thursday": lambda f: f["dow_code"] != fv.THURSDAY,
    "VIX9D/VIX 0.87-1.02 at entry": lambda f: f[fv.TS_9D_30].between(fv.STRATEGY["ts_low"], fv.STRATEGY["ts_high"]),
    "SPX above its 10-day SMA": lambda f: f["spx_vs_sma10_pct"] > fv.STRATEGY["sma10_min"],
    "short/long ratio <= 4": lambda f: f["slratio_condor"] <= fv.STRATEGY["max_slratio"],
    "SPX at or below its open": lambda f: f["spx_move_pct"] <= fv.STRATEGY["max_move"],
}


def ic_filters_on_mo(b: Books, book: str = "MO1") -> pd.DataFrame:
    """MO's condors put through the IC strategy's filters one at a time (cumulative) and each on its own."""
    f = b.mo[book]
    keep = pd.Series(True, index=f.index)
    rows = [{"filter": f"{book} as traded", "condors left": len(f), "sessions left": f["date"].nunique(),
             "passes this filter alone %": 100.0}]
    for name, rule in IC_FILTERS.items():
        alone = rule(f).fillna(False)
        keep &= alone
        rows.append({"filter": f"+ {name}", "condors left": int(keep.sum()), "sessions left": f.loc[keep, "date"].nunique(),
                     "passes this filter alone %": alone.mean() * 100})
    return pd.DataFrame(rows).set_index("filter")


def w_on_ic(b: Books) -> pd.DataFrame:
    """The other direction: IC's condors through MO's signal W and entry window."""
    f = b.ic
    w = f["spx_vs_sma5_prior_pct"] >= W_MIN
    win = f["entry_time"].str[:5].between("12:13", "14:13")
    return pd.DataFrame({"IC condors": [len(f)], "pass signal W %": [w.mean() * 100],
                         "in MO's 12:13-14:13 window %": [win.mean() * 100],
                         "both %": [(w & win).mean() * 100]})


# --------------------------------------------------------------------------- #
# 5. shared days: why they still only move partly together
# --------------------------------------------------------------------------- #
def shared_day_table(b: Books, other: str = "MO1") -> pd.DataFrame:
    """On days both trade: the day's setup for each book, and how often the two P/Ls agree in sign."""
    dly = b.daily
    both = (dly["IC"] != 0) & (dly[other] != 0)
    days = dly.index[both]
    ic = b.ic[b.ic["date"].isin(days)]
    mo = b.mo[other][b.mo[other]["date"].isin(days)]
    same_minute = len(set(zip(ic["date"], ic["entry_time"])) & set(zip(mo["date"], mo["entry_time"])))
    agree = np.sign(dly.loc[both, "IC"]) == np.sign(dly.loc[both, other])
    out = pd.DataFrame({
        "IC": {"condors on shared days": len(ic), "condors a day": len(ic) / len(days),
               "median move since open at entry %": ic["spx_move_pct"].median(),
               "median short-strike distance (pts)": np.median(np.r_[ic["otm_call"], ic["otm_put"]]),
               "avg P/L per shared day": dly.loc[both, "IC"].mean()},
        other: {"condors on shared days": len(mo), "condors a day": len(mo) / len(days),
                "median move since open at entry %": mo["spx_move_pct"].median(),
                "median short-strike distance (pts)": np.median(np.r_[mo["otm_call"], mo["otm_put"]]),
                "avg P/L per shared day": dly.loc[both, other].mean()},
    })
    out.attrs.update(shared_days=int(both.sum()), same_minute_entries=same_minute,
                     both_win=int(((dly.loc[both, "IC"] > 0) & (dly.loc[both, other] > 0)).sum()),
                     both_lose=int(((dly.loc[both, "IC"] < 0) & (dly.loc[both, other] < 0)).sum()),
                     sign_agree_pct=float(agree.mean() * 100))
    return out


def same_minute_table(b: Books, other: str = "MO1") -> pd.DataFrame:
    """Condors IC and ``other`` opened on the same day at the same minute: how alike the two trades are."""
    j = b.ic.merge(b.mo[other], on=["date", "entry_time"], suffixes=("_ic", "_mo"))
    ic, mo = j["pl_ic"], j["pl_mo"]
    out = pd.DataFrame({"value": {
        "same-minute pairs": len(j),
        "condor P/L correlation": ic.corr(mo),
        "IC avg P/L": ic.mean(), f"{other} avg P/L": mo.mean(),
        "IC win rate %": (ic > 0).mean() * 100, f"{other} win rate %": (mo > 0).mean() * 100,
        "both win %": ((ic > 0) & (mo > 0)).mean() * 100,
        f"IC wins, {other} loses %": ((ic > 0) & (mo <= 0)).mean() * 100,
        f"{other} wins, IC loses %": ((mo > 0) & (ic <= 0)).mean() * 100,
        "both lose %": ((ic <= 0) & (mo <= 0)).mean() * 100,
        "IC median short-strike distance (pts)": np.median(np.r_[j["otm_call_ic"], j["otm_put_ic"]]),
        f"{other} median short-strike distance (pts)": np.median(np.r_[j["otm_call_mo"], j["otm_put_mo"]]),
    }})
    return out


def tail_table(b: Books, q: float = 0.05) -> pd.DataFrame:
    """Each book's worst 5% of trading days: what the other books did on the same days."""
    rows = []
    dly = b.daily
    for k in BOOKS:
        traded = dly[k] != 0
        cut = dly.loc[traded, k].quantile(q)
        worst = dly.index[traded & (dly[k] <= cut)]
        row = {"book": k, "worst days": len(worst), "its avg P/L on them": dly.loc[worst, k].mean()}
        for o in BOOKS:
            if o == k:
                continue
            on = dly.loc[worst, o]
            row[f"{o}: traded %"] = (on != 0).mean() * 100
            row[f"{o}: avg P/L"] = on.mean()
            row[f"{o}: lost too %"] = (on < 0).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows).set_index("book")


# --------------------------------------------------------------------------- #
# 6. together in one book
# --------------------------------------------------------------------------- #
COMBOS = {"IC": ["IC"], "MO1": ["MO1"], "MO2": ["MO2"], "IC + MO1": ["IC", "MO1"], "IC + MO2": ["IC", "MO2"],
          "MO1 + MO2": ["MO1", "MO2"], "IC + MO1 + MO2": ["IC", "MO1", "MO2"]}


def _sharpe_capital(daily: np.ndarray, capital: float) -> float:
    """The backtester's Sharpe with a given starting capital (each book is given its own $200k)."""
    ex = np.asarray(daily, float) / capital - br.RISK_FREE / br.TRADING_DAYS
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(br.TRADING_DAYS))


def combined_table(b: Books) -> pd.DataFrame:
    """Each book alone and in pairs (one contract each, $200k of capital per book): P/L, Sharpe, drawdown,
    and the drawdown's date. Giving each book its own capital keeps the comparison fair: with the risk-free
    hurdle, two books on one $200k would look better just for running more money."""
    rows = {}
    for name, books in COMBOS.items():
        cap = br.START_CAPITAL * len(books)
        d = b.daily[books].sum(axis=1)
        eq = d.cumsum()
        dd = eq - eq.cummax()
        first = pd.Series(b.calendar).dt.year.to_numpy() <= fv.SPLIT_YEAR
        rows[name] = {"capital": cap, "total_pl": d.sum(), "sharpe": _sharpe_capital(d.to_numpy(), cap),
                      "sharpe_to_2024": _sharpe_capital(d.to_numpy()[first], cap),
                      "sharpe_from_2025": _sharpe_capital(d.to_numpy()[~first], cap),
                      "return_on_capital %": d.sum() / cap * 100,
                      "max_drawdown": dd.min(), "worst drawdown ends": dd.idxmin().date(),
                      "losing days %": (d[d != 0] < 0).mean() * 100, "worst day": d.min()}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def margin_table(b: Books) -> pd.DataFrame:
    """Margin the portfolio ties up in a day: the sum of every condor opened that day (0DTE, so all of
    them are open together by the afternoon; an upper bound, since a stopped call frees its margin)."""
    m = pd.DataFrame({"IC": b.ic.groupby("date")["margin"].sum(),
                      **{k: f.groupby("date")["Margin Req."].sum() for k, f in b.mo.items()}}).reindex(b.calendar).fillna(0)
    rows = {}
    for name, books in COMBOS.items():
        x = m[books].sum(axis=1)
        rows[name] = {"capital": br.START_CAPITAL * len(books), "peak day margin": x.max(),
                      "median margin on a trading day": x[x > 0].median(),
                      "peak as % of capital": x.max() / (br.START_CAPITAL * len(books)) * 100,
                      "days all of them trade": int((m[books] > 0).all(axis=1).sum())}
    return pd.DataFrame(rows).T


def worst_portfolio_days(b: Books, books: list[str] | None = None, n: int = 10) -> pd.DataFrame:
    """The portfolio's worst days and what each book lost on them."""
    books = books or BOOKS
    total = b.daily[books].sum(axis=1)
    worst = total.nsmallest(n).index
    return b.daily.loc[worst, books].assign(portfolio=total[worst])


# --------------------------------------------------------------------------- #
# 5b. why their spikes differ: where each one wins and what makes it lose
# --------------------------------------------------------------------------- #
DAY_MOVE_BINS = [-10, -1, -0.5, -0.2, 0.2, 0.5, 1, 10]
DAY_MOVE_LABELS = ["below -1%", "-1% to -0.5%", "-0.5% to -0.2%", "-0.2% to +0.2%", "+0.2% to +0.5%", "+0.5% to +1%",
                   "above +1%"]


def profit_zone_table(b: Books) -> pd.DataFrame:
    """Where each book's condors keep their credit: the close must land between the short strikes. Measured against
    the day's open, %: the median low edge (short put), centre (entry) and high edge (short call)."""
    rows = {}
    for k, f in _frames(b).items():
        rows[k] = {"short put (low edge)": f["zone_low_pct"].median(), "entry (centre)": f["zone_center_pct"].median(),
                   "short call (high edge)": f["zone_high_pct"].median(),
                   "width": (f["zone_high_pct"] - f["zone_low_pct"]).median(),
                   "short strikes from spot, %": (f["otm_call"] / f["spot"] * 100).median()}
    out = pd.DataFrame(rows).T
    lo = max(rows["IC"]["short put (low edge)"], rows["MO1"]["short put (low edge)"])
    hi = min(rows["IC"]["short call (high edge)"], rows["MO1"]["short call (high edge)"])
    out.attrs["ic_mo1_overlap"] = hi - lo   # negative: the median zones don't overlap
    return out


def pl_by_day_move(b: Books, shared_only: bool = False) -> pd.DataFrame:
    """Each book's average daily P/L by how SPX moved from its open to its 16:00 close, on the days it traded
    (``shared_only``: on the days IC and MO1 both traded)."""
    d = b.daily.join(b.market[["oc_pct"]])
    d["day"] = pd.cut(d["oc_pct"], DAY_MOVE_BINS, labels=DAY_MOVE_LABELS)
    out = {}
    for k in BOOKS:
        keep = d[k] != 0
        if shared_only:
            keep = (d["IC"] != 0) & (d["MO1"] != 0)
            if k == "MO2":
                continue
        g = d[keep].groupby("day", observed=False)[k]
        out[(k, "days")] = g.size()
        out[(k, "avg P/L")] = g.mean()
    return pd.DataFrame(out)


def stress_table(b: Books) -> pd.DataFrame:
    """The market's hardest days, and which book was in: the 25 widest-range days, and days SPX moved more than 1%
    from open to close, down and up."""
    d = b.daily.join(b.market[["oc_pct", "range_pct"]])
    groups = {"25 widest-range days": d.nlargest(25, "range_pct"), "SPX fell > 1% open to close": d[d["oc_pct"] <= -1],
              "SPX rose > 1% open to close": d[d["oc_pct"] >= 1]}
    rows = {}
    for name, t in groups.items():
        row = {"days": len(t)}
        for k in ("IC", "MO1", "MO2"):
            on = t[k] != 0
            row[f"{k} traded"] = int(on.sum())
            row[f"{k} avg P/L when in"] = t.loc[on, k].mean() if on.any() else np.nan
        row["IC and MO1 both in"] = int(((t["IC"] != 0) & (t["MO1"] != 0)).sum())
        row["IC and MO2 both in"] = int(((t["IC"] != 0) & (t["MO2"] != 0)).sum())
        rows[name] = row
    return pd.DataFrame(rows).T


def loss_sides(b: Books) -> pd.DataFrame:
    """Which side breaks on each book's losing condors (SPX closes past the short call or the short put), and the
    market of the days it trades."""
    rows = {}
    for k, f in _frames(b).items():
        close = f["date"].map(b.market["close"])
        call = close - f["spot"] > f["otm_call"]
        put = f["spot"] - close > f["otm_put"]
        lose = f["pl"] < 0
        worst = f["pl"] <= f["pl"].quantile(0.05)
        days = b.market.loc[b.daily.index[b.daily[k] != 0]]
        rows[k] = {"losing condors": int(lose.sum()), "of them: call side broken %": call[lose].mean() * 100,
                   "put side broken %": put[lose].mean() * 100,
                   "worst 5%: put side broken %": put[worst].mean() * 100,
                   "traded days: median VIX at the open": days["vix_open"].median(),
                   "traded days with VIX > 20 %": (days["vix_open"] > 20).mean() * 100,
                   "traded days: median range %": days["range_pct"].median(),
                   "traded days that close below the open %": (days["oc_pct"] < 0).mean() * 100}
    return pd.DataFrame(rows).T


def quarterly_pl(b: Books) -> pd.DataFrame:
    """Each book's P/L by calendar quarter."""
    q = b.daily.groupby(b.daily.index.to_period("Q")).sum()
    q.index = q.index.astype(str)
    return q


def joint_loss_days(b: Books, other: str = "MO1") -> pd.DataFrame:
    """The days both IC and ``other`` lost, with the market that day."""
    d = b.daily.join(b.market[["oc_pct", "range_pct", "vix_open"]])
    j = d[(d["IC"] < 0) & (d[other] < 0)][["IC", other, "oc_pct", "range_pct", "vix_open"]]
    return j.sort_values("IC")


def fig_spikes(b: Books, zones: pd.DataFrame, by_move: pd.DataFrame):
    """Left: where each book's condors keep their credit, against the day's open. Right: average daily P/L by the
    day's open-to-close move."""
    with plt.rc_context(fv.STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
        ax = axes[0]
        bins = np.linspace(-2, 2, 81)
        oc = b.market["oc_pct"].dropna()
        counts, _, _ = ax.hist(oc, bins=bins, color="#dedcd6", label="all 998 days: where SPX closed vs its open")
        top = counts.max()
        ax.set_ylim(0, top * 1.75)
        for i, k in enumerate(("IC", "MO1")):
            z = zones.loc[k]
            y = top * (1.55 - 0.2 * i)
            ax.fill_betweenx([y - top * 0.035, y + top * 0.035], z["short put (low edge)"], z["short call (high edge)"],
                             color=COLORS[k], alpha=0.85, lw=0)
            ax.plot([z["entry (centre)"]] * 2, [y - top * 0.035, y + top * 0.035], color="white", lw=1.5)
            ax.text(z["short call (high edge)"] + 0.05, y, f"{k}: keeps its credit if SPX closes here", va="center",
                    fontsize=8.5, color=fv.INK)
        ax.axvline(0, color=fv.INK_2, lw=0.7)
        ax.set_xlim(-2, 2.6)
        ax.set_xlabel("SPX close vs its 09:30 open, %")
        ax.set_ylabel("days")
        ax.set_title("Each book's winning zone (median condor) sits in a different place")
        ax.legend(loc="center left", fontsize=8)
        ax = axes[1]
        x = np.arange(len(by_move))
        for off, k in ((-0.2, "IC"), (0.2, "MO1")):
            ax.bar(x + off, by_move[(k, "avg P/L")], width=0.38, color=COLORS[k], label=k)
        ax.axhline(0, color=fv.INK_2, lw=0.7)
        ax.set_xticks(x, by_move.index, fontsize=8.5)
        ax.yaxis.set_major_formatter(lambda v, _: f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}")
        ax.set_xlabel("SPX open to close that day")
        ax.set_ylabel("average daily P/L on days it traded")
        ax.set_title("What kind of day pays each book, and what kind hurts it")
        ax.legend(loc="lower left")
        fig.tight_layout()
    return fig


def fig_equity(b: Books):
    with plt.rc_context(fv.STYLE):
        fig, ax = plt.subplots(figsize=(13, 4.4))
        for k in BOOKS:
            ax.plot(b.daily.index, b.daily[k].cumsum(), color=COLORS[k], lw=1.8, label=k)
        ax.axhline(0, color=fv.INK_2, lw=0.7)
        ax.yaxis.set_major_formatter(lambda v, _: fv._k(v))
        ax.set_ylabel("cumulative P/L, 1 contract")
        ax.set_title("The three books, one contract each (common calendar 2022-05 to 2026-05)")
        ax.legend(loc="upper left")
        fig.tight_layout()
    return fig


def fig_correlation(b: Books, ct: pd.DataFrame, roll: pd.DataFrame):
    with plt.rc_context({**fv.STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 2.2]})
        ax = axes[0]
        x = np.arange(len(ct))
        for off, col, color, label in ((-0.27, "daily", fv.BLUE, "daily (backtester)"), (0, "weekly", fv.ORANGE, "weekly"),
                                       (0.27, "monthly", fv.AQUA, "monthly")):
            ax.bar(x + off, ct[col], width=0.25, color=color, label=label)
        ax.errorbar(x - 0.27, ct["daily"], yerr=[ct["daily"] - ct["daily_ci_low"], ct["daily_ci_high"] - ct["daily"]],
                    fmt="none", ecolor=fv.INK_2, lw=0.8, capsize=2)
        for xi, v, hi in zip(x, ct["daily"], ct["daily_ci_high"]):
            ax.text(xi - 0.27, hi + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=fv.INK_2)
        ax.set_xticks(x, ct.index)
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(0, color=fv.INK_2, lw=0.7)
        ax.set_ylabel("correlation of P/L")
        ax.set_title("Correlation by pair")
        ax.legend(loc="upper left", fontsize=8)
        ax = axes[1]
        for (col, color) in zip(roll.columns, (fv.ORANGE, fv.AQUA, fv.INK_2)):
            ax.plot(roll.index, roll[col], color=color, lw=1.6, label=col)
        ax.axhline(0, color=fv.INK_2, lw=0.7)
        ax.set_ylim(-0.3, 1.05)
        ax.grid(axis="y", color="#e8e6e1")
        ax.set_ylabel(f"rolling {ROLL}-session correlation")
        ax.set_title("Does it ever spike? Daily correlation over a trailing six months")
        ax.legend(loc="center left", fontsize=8)
        fig.tight_layout()
    return fig


def fig_overlap(b: Books, ov: pd.DataFrame):
    with plt.rc_context(fv.STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.2, 1, 1]})
        ax = axes[0]
        x = np.arange(len(ov))
        parts = [("days both trade", fv.BLUE, "both trade"), ("only the first", "#9cc2ee", "only the first book"),
                 ("only the second", "#f5b99f", "only the second book"), ("neither", "#dedcd6", "neither")]
        bottom = np.zeros(len(ov))
        for col, color, label in parts:
            ax.bar(x, ov[col], bottom=bottom, color=color, width=0.6, label=label, edgecolor="white", lw=1)
            for xi, v, b0 in zip(x, ov[col], bottom):
                if v > 40:
                    ax.text(xi, b0 + v / 2, f"{v}", ha="center", va="center", fontsize=8, color=fv.INK)
            bottom += ov[col].to_numpy()
        ax.set_xticks(x, [i.replace(" vs ", "\nvs\n") for i in ov.index])
        ax.set_ylabel("sessions (of 998)")
        ax.set_title("How often both books trade the same day")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.5)
        for ax, other in zip(axes[1:], ("MO1", "MO2")):
            both = (b.daily["IC"] != 0) & (b.daily[other] != 0)
            d = b.daily[both]
            ax.scatter(d[other], d["IC"], s=14, color=COLORS[other], alpha=0.7, edgecolor="none")
            ax.axhline(0, color=fv.INK_2, lw=0.7)
            ax.axvline(0, color=fv.INK_2, lw=0.7)
            ax.xaxis.set_major_formatter(lambda v, _: fv._k(v))
            ax.yaxis.set_major_formatter(lambda v, _: fv._k(v))
            ax.set_xlabel(f"{other} P/L that day")
            ax.set_ylabel("IC P/L that day")
            ax.set_title(f"Shared days, IC vs {other}: corr {d['IC'].corr(d[other]):.2f} ({len(d)} days)")
        fig.tight_layout()
    return fig


def fig_signals(b: Books):
    """Each book's condors on SPX vs its open (x) and SPX vs its SMA5 (y): the two signals point opposite ways."""
    with plt.rc_context(fv.STYLE):
        fig, ax = plt.subplots(figsize=(10, 6.2))
        for k, s_ in (("MO1", 7), ("IC", 7)):
            f = _frames(b)[k]
            ax.scatter(f["spx_move_pct"], f["spx_vs_sma5_prior_pct"], s=s_, color=COLORS[k], alpha=0.35,
                       edgecolor="none", label=f"{k} condors ({len(f):,})")
        ax.axvline(0, color=fv.BLUE, lw=1.2, ls="--")
        ax.axhline(W_MIN, color=fv.ORANGE, lw=1.2, ls="--")
        ax.fill_between([-2, 0], W_MIN, 4, color="#e8e6e1", alpha=0.6, zorder=0, lw=0)
        ax.text(-1.0, 3.6, "both signals can fire here", ha="center", fontsize=8.5, color=fv.INK, fontweight="bold")
        ax.text(0.02, 0.02, "IC enters only left of the blue line\n(SPX at or below its open)", transform=ax.transAxes,
                va="bottom", fontsize=8.5, color=fv.INK)
        ax.text(0.98, 0.02, "MO enters only above the orange line\n(SPX >= 0.5% above its 5-day average)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=fv.INK)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("SPX move since the 09:30 open at entry, %")
        ax.set_ylabel("SPX at entry vs the average of the prior 5 closes, %")
        ax.set_title("Where each book enters: IC buys pullbacks, MO buys stretched rallies")
        leg = ax.legend(loc="upper right", markerscale=3)
        for h in leg.legend_handles:
            h.set_alpha(1)
        fig.tight_layout()
    return fig


def fig_filters(fl: pd.DataFrame, book: str = "MO1"):
    with plt.rc_context(fv.STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.4))
        ax = axes[0]
        y = np.arange(len(fl))[::-1]
        ax.barh(y, fl["sessions left"], color=[fv.ORANGE] + [fv.BLUE] * (len(fl) - 1), height=0.6)
        for yi, v in zip(y, fl["sessions left"]):
            ax.text(v, yi, f" {v}", va="center", fontsize=8.5, color=fv.INK_2)
        ax.set_yticks(y, fl.index)
        ax.set_xlabel("sessions left")
        ax.set_title(f"{book}'s days through IC's filters, one after another")
        ax = axes[1]
        a = fl.iloc[1:]
        y = np.arange(len(a))[::-1]
        ax.barh(y, a["passes this filter alone %"], color=fv.BLUE, height=0.6)
        for yi, v in zip(y, a["passes this filter alone %"]):
            ax.text(v, yi, f" {v:.0f}%", va="center", fontsize=8.5, color=fv.INK_2)
        ax.set_yticks(y, [i[2:] for i in a.index])
        ax.set_xlim(0, 110)
        ax.set_xlabel(f"% of {book}'s condors that pass it")
        ax.set_title("Each IC filter on its own")
        fig.tight_layout()
    return fig


def fig_combined(b: Books):
    with plt.rc_context(fv.STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(13, 6.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        for name, color, ls in (("IC", fv.BLUE, "-"), ("MO1", fv.ORANGE, "-"), ("IC + MO1", fv.INK, "-")):
            eq = b.daily[COMBOS[name]].sum(axis=1).cumsum()
            axes[0].plot(eq.index, eq, color=color, lw=1.8 if name != "IC + MO1" else 2.2, ls=ls, label=name)
            axes[1].plot(eq.index, eq - eq.cummax(), color=color, lw=1.4, label=name)
        axes[0].yaxis.set_major_formatter(lambda v, _: fv._k(v))
        axes[1].yaxis.set_major_formatter(lambda v, _: fv._k(v))
        axes[0].set_ylabel("cumulative P/L")
        axes[1].set_ylabel("drawdown")
        axes[0].set_title("IC and MO1 alone and together (one contract each)")
        axes[0].legend(loc="upper left")
        axes[1].axhline(0, color=fv.INK_2, lw=0.7)
        fig.tight_layout()
    return fig
