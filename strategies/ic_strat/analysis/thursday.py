"""Why no Thursdays: is it the market on Thursdays (VIX regime, trend, the move into the entry), or how the
market moves after entry on Thursdays?

Two books: the raw book (every condor, held to expiry: the book the greedy cascade drops Thursdays from) and
the finished strategy with Thursdays allowed back in. For each:

- ``weekday_profile``: per weekday, what was known at entry (VIX, VIX9D/VIX, trend, move since the open, gap)
  and what happened after (each side's P/L, breaches, the session's range, the move after entry, and the range
  against the move VIX implied);
- ``by_regime``: Thursday against the other days inside the same VIX / VIX9D/VIX / trend regime;
- ``regression``: condor P/L on a Thursday dummy, with no controls, with every entry feature, and with the
  session's realized range added (an outcome, to test the mechanism, not a filter);
- ``concentration``: how much of Thursday's loss sits in its worst sessions, and Thursday by year.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import final_version as fv

THU = fv.THURSDAY
DAYS = fv.WEEKDAYS
# What the strategy can know at entry; the controls in ``regression``.
ENTRY_CONTROLS = {"vix30_open": "VIX at the open", fv.TS_9D_30: "VIX9D/VIX at entry", "spx_move_pct": "move since the open",
                  "spx_vs_sma10_pct": "SPX vs its SMA10", "gap_pct": "opening gap", "credit": "credit",
                  "mins_from_open": "entry time"}
REALIZED = "range_vs_implied"


def books(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The raw book and the finished strategy with Thursdays allowed, with the realized-vs-implied range added."""
    p = panel.copy()
    # The session's high-low range against the daily move VIX implied at the open (VIX / sqrt(252)).
    p[REALIZED] = p["day_range_pct"] / (p["vix30_open"] / np.sqrt(252))
    return {"raw book": p,
            "strategy with Thursdays": p[fv.version_mask(p, **{**fv.STRATEGY, "skip_thursday": False})]}


def weekday_profile(book: pd.DataFrame) -> pd.DataFrame:
    """Per weekday: entry conditions and outcomes, per condor (P/L, sides, breaches) and per session (range, VIX)."""
    rows = []
    for code, day in enumerate(DAYS):
        g = book[book["dow_code"] == code]
        if g.empty:
            continue
        s = g.groupby("date").agg(vix=("vix30_open", "first"), rng=("day_range_pct", "first"), rvi=(REALIZED, "first"),
                                  pl=(fv.PL, "sum"))
        rows.append({
            "weekday": day, "sessions": len(s), "trades": 2 * len(g), "avg_pl": g[fv.PL].mean(),
            "avg_pl_call": g["pl_call"].mean(), "avg_pl_put": g["pl_put"].mean(),
            "call_breached_pct": g["call_breached"].mean() * 100, "put_breached_pct": g["put_breached"].mean() * 100,
            "vix_open": s["vix"].mean(), "ts_9d_30_entry": g[fv.TS_9D_30].mean(),
            "above_sma10_pct": (g["spx_vs_sma10_pct"] > 0).mean() * 100, "move_since_open": g["spx_move_pct"].mean(),
            "abs_gap": g["gap_pct"].abs().mean(), "day_range_pct": s["rng"].mean(),
            "big_range_days_pct": (s["rng"] > 1.0).mean() * 100,
            "abs_move_after_entry": g["move_after_entry_pct"].abs().mean(), "range_vs_implied": s["rvi"].mean(),
            "total_pl": g[fv.PL].sum(),
        })
    return pd.DataFrame(rows).set_index("weekday")


REGIMES = {
    "vix30_open": ("VIX at the open", [0, 15, 18, 22, 100]),
    fv.TS_9D_30: ("VIX9D/VIX at entry", [0, 0.9, 0.95, 1.0, 5]),
    "spx_vs_sma10_pct": ("SPX vs its SMA10, %", [-50, -1, 0, 1, 50]),
}


def by_regime(book: pd.DataFrame) -> pd.DataFrame:
    """Thursday against the other days inside each regime bucket: avg P/L per condor and the difference."""
    rows = []
    thu = book["dow_code"] == THU
    for feat, (name, bins) in REGIMES.items():
        for iv, g in book.groupby(pd.cut(book[feat], bins), observed=True):
            t, o = g[thu.loc[g.index]], g[~thu.loc[g.index]]
            rows.append({"regime": name, "bucket": fv._bin_label(iv, bins[0], bins[-1]),
                         "thursday_trades": 2 * len(t), "thursday_avg_pl": t[fv.PL].mean(),
                         "other_days_avg_pl": o[fv.PL].mean(), "difference": t[fv.PL].mean() - o[fv.PL].mean(),
                         "thursday_range_vs_implied": t.groupby("date")[REALIZED].first().mean(),
                         "other_range_vs_implied": o.groupby("date")[REALIZED].first().mean()})
    return pd.DataFrame(rows).set_index(["regime", "bucket"])


def _ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and standard errors clustered by ``groups`` (the session)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        sg = X[m].T @ resid[m]
        meat += np.outer(sg, sg)
    n, k, G = len(y), X.shape[1], len(np.unique(groups))
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return beta, np.sqrt(np.diag(V))


MODELS = {
    "Thursday alone": [],
    "+ every entry feature": list(ENTRY_CONTROLS),
    "+ the session's range vs implied (an outcome)": [*ENTRY_CONTROLS, REALIZED],
}


def regression(book: pd.DataFrame) -> pd.DataFrame:
    """Condor P/L on a Thursday dummy: the Thursday effect in $ per condor, with standard errors clustered by
    session, alone, controlling for everything known at entry, and with the realized range added."""
    rows = []
    for name, ctrls in MODELS.items():
        d = book[[fv.PL, "dow_code", "date", *ctrls]].dropna()
        X = np.column_stack([np.ones(len(d)), (d["dow_code"] == THU).astype(float),
                             *[(d[c] - d[c].mean()) / d[c].std() for c in ctrls]])
        beta, se = _ols_cluster(d[fv.PL].to_numpy(float), X, d["date"].to_numpy())
        t = beta[1] / se[1]
        rows.append({"model": name, "condors": len(d), "thursday_effect": beta[1], "se": se[1], "t": t,
                     "p": 2 * stats.norm.sf(abs(t))})
    return pd.DataFrame(rows).set_index("model")


def concentration(book: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Thursday's session P/L: its worst sessions' share of the loss, and Thursday vs other days by year."""
    s = book.groupby(["date", "dow_code"])[fv.PL].sum().reset_index()
    thu = s[s["dow_code"] == THU].sort_values(fv.PL)
    worst = pd.DataFrame({"worst_n": [5, 10, 20], "pl": [thu[fv.PL].head(n).sum() for n in (5, 10, 20)]})
    worst["thursday_total"] = thu[fv.PL].sum()
    worst["rest_of_thursdays"] = worst["thursday_total"] - worst["pl"]
    s["year"] = s["date"].dt.year
    s["day"] = np.where(s["dow_code"] == THU, "Thursday", "other days")
    yr = s.pivot_table(index="year", columns="day", values=fv.PL, aggfunc="mean")
    yr["thursday_sessions"] = s[s["day"] == "Thursday"].groupby("year").size()
    medians = s.groupby("day")[fv.PL].agg(["median", "mean", lambda x: (x > 0).mean() * 100])
    medians.columns = ["median_session_pl", "mean_session_pl", "winning_sessions_pct"]
    return {"worst": worst.set_index("worst_n"), "by_year": yr, "sessions": medians,
            "worst_days": thu.head(10).set_index("date")[[fv.PL]]}


def fig_weekdays(prof: pd.DataFrame, title: str):
    """Per weekday: each side's avg P/L, the session's range against VIX's implied move, and VIX at the open."""
    with plt.rc_context(fv.STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.3))
        x = np.arange(len(prof))
        labels = [d[:3] for d in prof.index]
        thu = np.array([d == "Thursday" for d in prof.index])
        ax = axes[0]
        ax.bar(x - 0.19, prof["avg_pl_call"], width=0.36, color=fv.BLUE, label="call side")
        ax.bar(x + 0.19, prof["avg_pl_put"], width=0.36, color=fv.ORANGE, label="put side")
        ax.axhline(0, color=fv.INK_2, lw=0.7)
        ax.set_title("Avg P/L per condor, each side ($)")
        ax.legend(loc="lower left")
        for ax, col, ttl, fmt in ((axes[1], "range_vs_implied", "Session range / VIX-implied daily move", "{:.2f}"),
                                  (axes[2], "vix_open", "VIX at the open", "{:.1f}")):
            ax.bar(x, prof[col], width=0.6, color=[fv.RED if t else fv.GREY for t in thu])
            for xi, v in zip(x, prof[col]):
                ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=fv.INK_2)
            ax.set_title(ttl)
        for ax in axes:
            ax.set_xticks(x, labels)
        fig.suptitle(title, x=0.01, ha="left")
        fig.tight_layout()
    return fig


def fig_regimes(reg: pd.DataFrame, title: str):
    """Thursday vs the other days inside each regime bucket (avg P/L per condor)."""
    regimes = reg.index.get_level_values(0).unique()
    with plt.rc_context(fv.STYLE):
        fig, axes = plt.subplots(1, len(regimes), figsize=(17, 4.3))
        for ax, r in zip(axes, regimes):
            t = reg.loc[r]
            x = np.arange(len(t))
            ax.bar(x - 0.19, t["other_days_avg_pl"], width=0.36, color=fv.GREY, label="other days")
            ax.bar(x + 0.19, t["thursday_avg_pl"], width=0.36, color=fv.RED, label="Thursday")
            ax.axhline(0, color=fv.INK_2, lw=0.7)
            ax.set_xticks(x, t.index, fontsize=8)
            ax.set_title(r)
        axes[0].set_ylabel("avg P/L per condor ($)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper right", ncol=2, frameon=False)
        fig.suptitle(title, x=0.01, ha="left")
        fig.tight_layout()
    return fig


@dataclass
class ThursdayStudy:
    profiles: dict[str, pd.DataFrame]
    regimes: dict[str, pd.DataFrame]
    regressions: dict[str, pd.DataFrame]
    concentration: dict[str, dict[str, pd.DataFrame]]


def compute(panel: pd.DataFrame) -> ThursdayStudy:
    bk = books(panel)
    return ThursdayStudy(profiles={k: weekday_profile(b) for k, b in bk.items()},
                         regimes={k: by_regime(b) for k, b in bk.items()},
                         regressions={k: regression(b) for k, b in bk.items()},
                         concentration={k: concentration(b) for k, b in bk.items()})
