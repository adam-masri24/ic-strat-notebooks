"""The 2026 drawdown in the backtester's own log of the live strategy.

The log (``data/strategy/ic_strat/spx/backtest_log/ic_live_trade_log.csv``, an Option Omega export) is the strategy as
traded: the 7 tranches, no Thursdays, every condor held to expiry. It runs to 2026-09-28, past the 20-tranche export the
rest of the study is built on (which ends 2026-05-07), so it is the only view of the drawdown that started in May 2026.

- ``load_log``: one row per vertical spread, with its short strike, distance from spot and whether SPX closed through it;
- ``condors``: the call and put rows of one entry paired;
- ``reconcile``: the log against the study's rebuild, on the sessions both cover;
- ``before_vs_drawdown``: the condors of the drawdown against everything before it;
- ``worst_days`` / ``by_year``: the biggest losing days, and the trade's conditions year by year.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import STRATEGY_DATA_ROOT

from . import backtester_rules as br
from . import final_version as fv

LOG_FILE = STRATEGY_DATA_ROOT / "ic_strat" / "spx" / "backtest_log" / "ic_live_trade_log.csv"
FULL_LOSS = -600   # a condor that loses at least this much lost (close to) a full 10-wide wing


def load_log(path=LOG_FILE) -> pd.DataFrame:
    """One row per vertical spread of the backtester's log."""
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["Date Opened"])
    d["tranche"] = d["Time Opened"].str[:5]
    d["side"] = np.where(d["Legs"].str.contains(" C STO"), "call", "put")
    d["short"] = d["Legs"].str.extract(r"(\d+(?:\.\d+)?) [CP] STO")[0].astype(float)
    d["spot"], d["close"], d["pl"], d["credit"] = d["Opening Price"], d["Closing Price"], d["P/L"], d["Premium"]
    d["dist_pct"] = (d["short"] - d["spot"]).abs() / d["spot"] * 100
    d["breached"] = np.where(d["side"] == "call", d["close"] > d["short"], d["close"] < d["short"])
    d["move_after_pct"] = (d["close"] / d["spot"] - 1) * 100
    d["move_since_open_pct"] = d["Movement"] / (d["spot"] - d["Movement"]) * 100
    d["vix"] = d["Opening VIX"]
    return d


def condors(log: pd.DataFrame) -> pd.DataFrame:
    """The call and put spread of each entry as one condor."""
    g = log.groupby(["date", "tranche"])
    c = g.agg(pl=("pl", "sum"), credit=("credit", "sum"), spot=("spot", "first"), close=("close", "first"),
              vix=("vix", "first"), move_after_pct=("move_after_pct", "first"),
              move_since_open_pct=("move_since_open_pct", "first"), dist_pct=("dist_pct", "mean")).reset_index()
    sides = log.pivot_table(index=["date", "tranche"], columns="side", values="pl", aggfunc="sum").reset_index()
    return c.merge(sides, on=["date", "tranche"])


def daily_matrix(log: pd.DataFrame) -> pd.DataFrame:
    """Session x tranche P/L on every weekday from the log's first to its last session (no-trade days as zero)."""
    days = pd.bdate_range(log["date"].min(), log["date"].max())
    return log.pivot_table(index="date", columns="tranche", values="pl", aggfunc="sum").reindex(days).fillna(0.0)


def reconcile(log: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """The log against the study's rebuild of the same strategy, on the sessions both cover."""
    rb = panel[fv.version_mask(panel, **fv.STRATEGY)]
    lo, hi = log["date"].min(), min(log["date"].max(), panel["date"].max())
    rows = {}
    for name, frame, col in (("backtester's log", log, "pl"), ("the study's rebuild", rb, fv.PL)):
        f = frame[(frame["date"] >= lo) & (frame["date"] <= hi)]
        trades = len(f) if name == "backtester's log" else 2 * len(f)
        days = pd.bdate_range(lo, hi)
        dly = f.groupby("date")[col].sum().reindex(days, fill_value=0.0)
        eq = dly.cumsum()
        rows[name] = {"trades": trades, "sessions": f["date"].nunique(), "total_pl": f[col].sum(), "sharpe": br.sharpe(dly),
                      "max_drawdown": float((eq - np.maximum.accumulate(np.maximum(eq, 0))).min())}
    out = pd.DataFrame(rows).T
    out.attrs["window"] = (lo, hi)
    return out


def before_vs_drawdown(c: pd.DataFrame, log: pd.DataFrame, start, end) -> pd.DataFrame:
    """The drawdown's condors against everything before it: P/L, win rate, credit, VIX, moves and breaches."""
    def stats(cc, ll):
        return {"condors": len(cc), "sessions": cc["date"].nunique(), "total_pl": cc["pl"].sum(), "avg_pl": cc["pl"].mean(),
                "win_rate_pct": (cc["pl"] > 0).mean() * 100, "avg_win": cc.loc[cc["pl"] > 0, "pl"].mean(),
                "avg_loss": cc.loc[cc["pl"] <= 0, "pl"].mean(), "full_wing_losses": int((cc["pl"] < FULL_LOSS).sum()),
                "credit_per_condor": cc["credit"].mean(), "vix_at_entry": cc["vix"].mean(),
                "abs_move_after_entry_pct": cc["move_after_pct"].abs().mean(),
                "short_call_breached_pct": ll.loc[ll["side"] == "call", "breached"].mean() * 100,
                "short_put_breached_pct": ll.loc[ll["side"] == "put", "breached"].mean() * 100,
                "call_side_avg_pl": cc["call"].mean(), "put_side_avg_pl": cc["put"].mean()}
    b, dd = c["date"] < start, (c["date"] >= start) & (c["date"] <= end)
    lb, ldd = log["date"] < start, (log["date"] >= start) & (log["date"] <= end)
    return pd.DataFrame({"before the drawdown": stats(c[b], log[lb]), "the drawdown": stats(c[dd], log[ldd])})


def worst_days(c: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """The log's worst sessions: condors, how many lost a full wing, which side broke, VIX and SPX after entry."""
    g = c.groupby("date").agg(pl=("pl", "sum"), condors=("pl", "size"), full_wing=("pl", lambda x: int((x < FULL_LOSS).sum())),
                              call_side=("call", "sum"), put_side=("put", "sum"), vix=("vix", "first"),
                              spx_after_entry_pct=("move_after_pct", "mean"))
    g["weekday"] = g.index.day_name()
    return g.nsmallest(n, "pl")


def by_year(log: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """Year by year: SPX level, credit and short-strike distance, VIX, breaches, P/L and the worst day."""
    y = log.groupby(log["date"].dt.year).agg(spx=("spot", "median"), credit_per_spread=("credit", "median"),
                                             short_dist_pct=("dist_pct", "median"), vix=("vix", "median"),
                                             breached_pct=("breached", "mean"))
    y["breached_pct"] *= 100
    cy = c.groupby(c["date"].dt.year)
    y["condors"] = cy.size()
    y["avg_pl_condor"] = cy["pl"].mean()
    y["full_wing_losses"] = cy["pl"].apply(lambda x: int((x < FULL_LOSS).sum()))
    y["total_pl"] = cy["pl"].sum()
    y["worst_day"] = c.groupby([c["date"].dt.year, "date"])["pl"].sum().groupby(level=0).min()
    y.index.name = "year"
    return y


def fig_by_year(y: pd.DataFrame):
    """Credit per spread, short-strike distance, full-wing losses and the worst day, by year (2026 in red)."""
    with plt.rc_context(fv.STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(17, 4))
        x = np.arange(len(y))
        last = np.array([str(i) == str(y.index.max()) for i in y.index])
        for ax, col, title, fmt in ((axes[0], "credit_per_spread", "Median credit per spread ($)", "{:.0f}"),
                                    (axes[1], "short_dist_pct", "Median short strike from spot (%)", "{:.2f}"),
                                    (axes[2], "full_wing_losses", "Condors losing a full wing", "{:.0f}"),
                                    (axes[3], "worst_day", "Worst day ($)", "{:,.0f}")):
            v = y[col].astype(float)
            ax.bar(x, v, width=0.6, color=[fv.RED if l else fv.GREY for l in last])
            for xi, val in zip(x, v):
                ax.text(xi, val, fmt.format(val), ha="center", va="bottom" if val >= 0 else "top", fontsize=8, color=fv.INK_2)
            ax.set_xticks(x, [f"{i}{' (to Sep)' if l else ''}" for i, l in zip(y.index, last)], fontsize=8)
            ax.set_title(title)
            ax.axhline(0, color=fv.INK_2, lw=0.6)
        fig.suptitle("The backtester's log, year by year (2026 in red)", x=0.01, ha="left")
        fig.tight_layout()
    return fig
