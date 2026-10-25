"""The strategy built as a greedy cascade: at each step, the setting that raises Sharpe most is added.

The raw export (every condor, every session, held to expiry) gets the strategy's seven settings one at
a time. At each step every setting not yet added is tried on the book built so far, and the one that
gives the highest Sharpe is kept (``greedy``). The tranche setting is the fixed set of 15 forward-
selected tranches (``final_version.SELECTED_TRANCHES``). The order that comes out is ``ORDER``:

1. no Thursdays
2. max SPX move up since the open 0%
3. SPX at entry above its 10-day SMA
4. VIX9D / VIX 0.87-1.02 at entry
5. the 7 tranches (11:33, 11:53, 12:13, 13:13, 13:53, 14:13, 14:53)
6. short/long ratio of the whole condor <= 4

The strategy holds every condor to expiry; the call stop it no longer uses is studied on the finished
strategy (``final_version.stop_scan``), not as a step.

Each step is analysed **on the book built so far**: the earlier settings on, the later ones off.
Until the stop is added, every condor is held to expiry. Every version is scored the same way as in
``final_version`` (all 998 sessions, the backtester's Sharpe, each half separately), and one trade is
one vertical spread, as the backtester counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import backtester_rules as br
from . import final_version as fv
from . import regime_study as rs


@dataclass(frozen=True)
class Step:
    key: str
    label: str
    change: dict


SETTINGS = {
    "band": Step("band", "VIX9D/VIX 0.87-1.02 at entry", {"ts_low": fv.STRATEGY["ts_low"], "ts_high": fv.STRATEGY["ts_high"]}),
    "move": Step("move", "max move up 0%", {"max_move": fv.STRATEGY["max_move"]}),
    "sma10": Step("sma10", "above the 10-day SMA", {"sma10_min": fv.STRATEGY["sma10_min"]}),
    "ratio": Step("ratio", f"short/long ratio <= {fv.MAX_SLRATIO:g}", {"max_slratio": fv.MAX_SLRATIO}),
    "thursday": Step("thursday", "no Thursdays", {"skip_thursday": True}),
    "tranches": Step("tranches", f"{len(fv.SELECTED_TRANCHES)} forward-selected tranches", {"drop": fv.MORNING}),
}
# The greedy order (``greedy`` recomputes it and flags any difference).
ORDER = ["thursday", "move", "sma10", "band", "tranches", "ratio"]
STEPS = [SETTINGS[k] for k in ORDER]
KEYS = [s.key for s in STEPS]


def greedy(panel: pd.DataFrame, cal: fv.Calendars) -> pd.DataFrame:
    """The greedy cascade: at each step, try every setting not yet added on the book built so far and add the one
    with the highest Sharpe. One row per step: each candidate's Sharpe (blank once added); ``attrs`` hold the order
    and each trial's trades and P/L."""
    cfg, left, order = {}, list(SETTINGS), []
    rows, detail = [], []
    while left:
        trial = {}
        for k in left:
            st = fv.score(panel, fv.version_mask(panel, **{**cfg, **SETTINGS[k].change}), cal, fv.PL)
            trial[k] = st
            detail.append({"step": len(order) + 1, "setting": k, "trades": st["trades"], "total_pl": st["total_pl"],
                           "sharpe": st["sharpe"]})
        best = max(trial, key=lambda k: trial[k]["sharpe"])
        rows.append({k: trial[k]["sharpe"] if k in trial else np.nan for k in SETTINGS} | {"picked": best})
        cfg.update(SETTINGS[best].change)
        order.append(best)
        left.remove(best)
    out = pd.DataFrame(rows, index=pd.Index(range(1, len(rows) + 1), name="step"))[["picked", *SETTINGS]]
    out.attrs.update(order=order, matches=order == ORDER, detail=pd.DataFrame(detail))
    return out


def greedy_view(g: pd.DataFrame):
    """The greedy trials: each step's Sharpe for every setting tried (blank once added), the one added highlighted."""
    labels = {k: v.label for k, v in SETTINGS.items()}
    view = g.rename(columns=labels).assign(picked=g["picked"].map(labels))

    def mark(row):
        return ["background-color: #d6e6f9; font-weight: bold" if c == row["picked"] else "" for c in row.index]

    return view.style.apply(mark, axis=1).format(precision=2, na_rep="")


def before(key: str) -> tuple[dict, str]:
    """The filters on before ``key`` is added, and the P/L column (held to expiry until the stop step)."""
    cfg = {}
    for s in STEPS[:KEYS.index(key)]:
        cfg.update(s.change)
    return cfg, fv.PL


def after(key: str) -> tuple[dict, str]:
    """The filters on once ``key`` is added, and the P/L column."""
    cfg = {}
    for s in STEPS[:KEYS.index(key) + 1]:
        cfg.update(s.change)
    return cfg, fv.PL


def _sides(book: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """Each condor's call and put P/L: the call with the stop once it is on, otherwise held to expiry."""
    call = book[col] - book["pl_put"]
    return call, book["pl_put"]


def _stats(panel: pd.DataFrame, keep: pd.Series, col: str, cal: fv.Calendars) -> dict:
    book = panel[keep]
    st = fv.score(panel, keep, cal, col)
    call, put = _sides(book, col)
    sides = pd.concat([call, put]).dropna()
    return {"trades": st["trades"], "sessions": st["sessions"], "total_pl": st["total_pl"], "sharpe": st["sharpe"],
            fv.H1: st[fv.H1], fv.H2: st[fv.H2], "max_drawdown": st["max_drawdown"],
            "win_rate_trade": float((sides > 0).mean() * 100), "win_rate_condor": st["win_rate"],
            "avg_pl_condor": st["avg_pl"]}


# --------------------------------------------------------------------------- #
# the cascade
# --------------------------------------------------------------------------- #
def cascade_table(panel: pd.DataFrame, cal: fv.Calendars) -> pd.DataFrame:
    """The raw book, then each setting added: trades, P/L, Sharpe, drawdown and win rate, and the change at each step."""
    rows = {"raw book": _stats(panel, pd.Series(True, index=panel.index), fv.PL_EXPIRY, cal)}
    for i, s in enumerate(STEPS, start=1):
        cfg, col = after(s.key)
        rows[f"{i}. + {s.label}"] = _stats(panel, fv.version_mask(panel, **cfg), col, cal)
    out = pd.DataFrame(rows).T
    for c in ("trades", "total_pl", "sharpe"):
        out[f"{c}_change"] = out[c].diff()
    out.index.name = "step"
    return out


def note(ct: pd.DataFrame, key: str, what: str = "") -> str:
    """A chart title for one step: what is shown, and the trades, P/L and Sharpe before -> after the step."""
    i = KEYS.index(key) + 1
    a, b = ct.iloc[i - 1], ct.iloc[i]
    head = f"Step {i}, + {STEPS[i - 1].label}"
    text = (f"{head}{': ' + what if what else ''}\ntrades {int(a['trades']):,} → {int(b['trades']):,}   |   "
            f"P/L {fv._k(a['total_pl'])} → {fv._k(b['total_pl'])}   |   Sharpe {a['sharpe']:.2f} → {b['sharpe']:.2f}")
    return text.replace("$", r"\$")   # matplotlib reads a pair of $ as maths


def fig_cascade(ct: pd.DataFrame):
    """Sharpe, total P/L and trades as each setting is added."""
    with plt.rc_context(fv.STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        x = np.arange(len(ct))
        labels = ["raw book"] + [f"+ {s.label}" for s in STEPS]
        last = len(ct) - 1
        for ax, col, title, fmt in ((axes[0], "sharpe", "Sharpe", "{:.2f}"), (axes[1], "total_pl", "Total P/L", None),
                                    (axes[2], "trades", "Trades (one per vertical spread)", "{:,.0f}")):
            vals = ct[col].astype(float)
            ax.bar(x, vals, width=0.62, color=[fv.BLUE if i == last else (fv.RED if v < 0 else fv.GREY)
                                                for i, v in enumerate(vals)])
            for xi, v in zip(x, vals):
                ax.text(xi, v, fmt.format(v) if fmt else fv._k(v), ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=7.5, color=fv.INK_2)
            ax.axhline(0, color=fv.INK_2, lw=0.7)
            ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
            ax.set_title(title)
            if col == "total_pl":
                ax.yaxis.set_major_formatter(lambda v, _: fv._k(v))
            elif col == "trades":
                ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
        f = ct.iloc[-1]
        fig.suptitle(f"The greedy cascade: at each step the setting that adds most Sharpe (blue = the strategy: {int(f['trades']):,} "
                     f"trades, {fv._k(f['total_pl'])}, Sharpe {f['sharpe']:.2f})".replace("$", r"\$"), x=0.01, ha="left")
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# per step: the feature's profile and its threshold, on the book built so far
# --------------------------------------------------------------------------- #
def profile(panel: pd.DataFrame, feature: str, key: str) -> pd.DataFrame:
    """P/L across ``feature``'s range on the book before step ``key`` (same columns as ``final_version``'s profiles)."""
    _, bins, kept, _ = fv.PROFILES[feature]
    cfg, col = before(key)
    book = panel[fv.version_mask(panel, **cfg)]
    call, put = _sides(book, col)
    rows = []
    for iv, grp in book.groupby(pd.cut(book[feature], bins), observed=True):
        pl = grp[col]
        lo, hi = rs.session_bootstrap_mean(pl, grp["date"])
        first = grp["date"].dt.year <= fv.SPLIT_YEAR
        rows.append({
            "bin": fv._bin_label(iv, bins[0], bins[-1]), "condors": len(grp), "trades": 2 * len(grp),
            "sessions": grp["date"].nunique(), "avg_pl": pl.mean(), "ci_low": lo, "ci_high": hi,
            "win_rate": (pl > 0).mean() * 100, f"avg_pl_to_{fv.SPLIT_YEAR}": pl[first].mean(),
            f"avg_pl_from_{fv.SPLIT_YEAR + 1}": pl[~first].mean(), "total_pl": pl.sum(),
            "avg_pl_call": call.loc[grp.index].mean(), "avg_pl_put": put.loc[grp.index].mean(),
            "call_breached_pct": grp["call_breached"].mean() * 100, "put_breached_pct": grp["put_breached"].mean() * 100,
            "abs_move_after_entry_pct": grp["move_after_entry_pct"].abs().mean(),
            "day_range_pct": grp["day_range_pct"].mean(),
            "in_strategy": kept is not None and bool(iv.left >= kept[0] - 1e-9 and iv.right <= kept[1] + 1e-9),
        })
    return pd.DataFrame(rows).set_index("bin")


def sweep(panel: pd.DataFrame, cal: fv.Calendars, name: str, key: str) -> pd.DataFrame:
    """One threshold moved on the book before step ``key`` (the band's floor and cap each move with the other fixed)."""
    cfg, col = before(key)
    if name in ("ts_low", "ts_high"):
        cfg = {**cfg, "ts_low": fv.STRATEGY["ts_low"], "ts_high": fv.STRATEGY["ts_high"]}
    rows = {fv._fmt(v): fv.score(panel, fv.version_mask(panel, **{**cfg, name: v}), cal, col) for v in fv.SWEEPS[name]}
    t = pd.DataFrame(rows).T[["sessions", "trades", "total_pl", "sharpe", "max_drawdown", fv.H1, fv.H2]]
    t.index.name = name
    t["is_strategy"] = [v == fv._fmt(fv.STRATEGY[name]) for v in t.index]
    return t


SMA_RULES = {"SMA10": "spx_vs_sma10_pct", "SMA20": "spx_vs_sma20_pct"}


def sma_sweeps(panel: pd.DataFrame, cal: fv.Calendars) -> dict[str, pd.DataFrame]:
    """At the SMA step: SPX at entry more than x% above its 10-day SMA, and the same for the 20-day, at every
    threshold of the SMA sweep (the book before the step; "off" = no trend rule)."""
    cfg, col = before("sma10")
    out = {}
    for name, feat in SMA_RULES.items():
        rows = {fv._fmt(v): fv.score(panel, fv.version_mask(panel, **cfg, trend=None if v is None else panel[feat] > v),
                                     cal, col) for v in fv.SWEEPS["sma10_min"]}
        t = pd.DataFrame(rows).T[["sessions", "trades", "total_pl", "sharpe", "max_drawdown", fv.H1, fv.H2]]
        t.index.name = f"{name.lower()}_min"
        out[name] = t
    return out


def fig_sma_sweeps(sw: dict[str, pd.DataFrame], note: str):
    """SMA10 against SMA20 at every threshold: Sharpe, total P/L and max drawdown (the strategy: SMA10 at 0%)."""
    colors, markers = {"SMA10": fv.BLUE, "SMA20": fv.ORANGE}, {"SMA10": "o", "SMA20": "s"}
    with plt.rc_context(fv.STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
        for name, t in sw.items():
            t = t[t.index != "off"]
            x = np.arange(len(t))
            for ax, c in zip(axes, ("sharpe", "total_pl", "max_drawdown")):
                ax.plot(x, t[c].astype(float), color=colors[name], marker=markers[name], ms=5.5, lw=1.8,
                        label=f"SPX above its {name[3:]}-day SMA by more than x%")
                ax.set_xticks(x, [f"{v}%" for v in t.index])
        i0 = list(sw["SMA10"].index).index("0")
        for ax, title in zip(axes, ("Sharpe", "Total P/L", "Max drawdown")):
            fv._mark(ax, i0, "")
            ax.set_title(f"{title} (shaded: the strategy, SMA10 at 0%)" if title == "Sharpe" else title)
            ax.set_xlabel("x")
        for ax in axes[1:]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fv._k(v)))
        axes[0].legend(loc="lower center")
        fig.suptitle(note, x=0.01, ha="left")
        fig.tight_layout()
    return fig


def trend(panel: pd.DataFrame, cal: fv.Calendars) -> pd.DataFrame:
    """Every trend rule in the SMA10 rule's place, on the book before the SMA step."""
    cfg, col = before("sma10")
    rows = {name: fv.score(panel, fv.version_mask(panel, **cfg) if rule is None
                           else fv.version_mask(panel, **cfg, trend=rule), cal, col)
            for name, rule in fv.trend_rules(panel).items()}
    out = pd.DataFrame(rows).T[["sessions", "trades", "total_pl", "sharpe", "max_drawdown", fv.H1, fv.H2]]
    out["sharpe_vs_strategy"] = out["sharpe"] - out.loc[fv.STRATEGY_TREND, "sharpe"]
    return out


def ab_t_tests(a: np.ndarray, b: np.ndarray, n_boot: int = fv.AB_BOOT, seed: int = fv.SEED) -> dict:
    """Significance of version b against version a, on the same sessions (positive t = b better).

    * ``t_daily_pl`` / ``p_daily_pl``: paired t-test on the session-by-session P/L difference (b - a), every session
      (no-trade days as zero), two-sided;
    * ``t_sharpe`` / ``p_sharpe``: the Sharpe difference over its standard error from a paired session bootstrap
      (the same draws as ``final_version._paired_boot``), two-sided against a normal.
    """
    t_pl, p_pl = stats.ttest_rel(b, a)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n_boot, len(a)))
    se = (br.sharpe_rows(b[idx]) - br.sharpe_rows(a[idx])).std(ddof=1)
    t_sh = (br.sharpe(b) - br.sharpe(a)) / se
    return {"t_daily_pl": float(t_pl), "p_daily_pl": float(p_pl), "t_sharpe": float(t_sh),
            "p_sharpe": float(2 * stats.norm.sf(abs(t_sh)))}


def sma_ab(panel: pd.DataFrame, cal: fv.Calendars, tranches: list[str] | None = None) -> pd.DataFrame:
    """With every other setting on (the finished strategy, 400% call stop): above the 10-day SMA (A, the strategy)
    against the 20-day (B) and both, with a paired session bootstrap of the Sharpe difference. ``attrs`` hold the trades only
    one rule takes and each version's Sharpe and P/L by year."""
    cfg, col = {**fv.STRATEGY, "sma10_min": None}, fv.PL
    if tranches is not None:   # the same test on another tranche set
        cfg["drop"] = tuple(f"{t}:00" for t in fv.ALL_TRANCHES if t not in tranches)
    above10, above20 = panel["spx_vs_sma10_pct"] > 0, panel["spx_vs_sma20_pct"] > 0
    versions = {"A: above 10-day SMA (strategy)": fv.version_mask(panel, **cfg, trend=above10),
                "B: above 20-day SMA": fv.version_mask(panel, **cfg, trend=above20),
                "above both": fv.version_mask(panel, **cfg, trend=above10 & above20)}
    daily = {k: rs.daily_series(panel.loc[m, col], panel.loc[m, "date"], cal.all) for k, m in versions.items()}
    base = next(iter(versions))
    rows = {}
    for k, m in versions.items():
        lo, hi, p = (np.nan, np.nan, np.nan) if k == base else fv._paired_boot(daily[base].to_numpy(), daily[k].to_numpy())
        tt = (dict.fromkeys(["t_daily_pl", "p_daily_pl", "t_sharpe", "p_sharpe"], np.nan) if k == base
              else ab_t_tests(daily[base].to_numpy(), daily[k].to_numpy()))
        st = _stats(panel, m, col, cal)
        rows[k] = {**st, "win_rate_spread": st["win_rate_trade"],
                   "sharpe_vs_A": br.sharpe(daily[k]) - br.sharpe(daily[base]), "ci_low": lo, "ci_high": hi,
                   "p_better_than_A": p, **tt}
    out = pd.DataFrame(rows).T
    years = pd.DatetimeIndex(cal.all).year
    by_year = pd.DataFrame({
        **{f"sharpe {k.split(':')[0]}": [br.sharpe(daily[k][years == y]) for y in sorted(set(years))] for k in list(versions)[:2]},
        **{f"P/L {k.split(':')[0]}": [daily[k][years == y].sum() for y in sorted(set(years))] for k in list(versions)[:2]},
    }, index=pd.Index([str(y) for y in sorted(set(years))], name="year"))
    a_only = versions[base] & ~above20
    b_only = versions["B: above 20-day SMA"] & ~above10
    out.attrs.update(only_sma10=2 * int(a_only.sum()), only_sma20=2 * int(b_only.sum()),
                     only_sma10_pl=float(panel.loc[a_only, col].sum()), only_sma20_pl=float(panel.loc[b_only, col].sum()),
                     by_year=by_year)
    return out


# --------------------------------------------------------------------------- #
# step 4: the short/long ratio
# --------------------------------------------------------------------------- #
RATIO = "slratio_condor"


def ratio_quintiles(panel: pd.DataFrame) -> pd.DataFrame:
    cfg, col = before("ratio")
    book = panel[fv.version_mask(panel, **cfg)]
    first = book["date"].dt.year <= fv.SPLIT_YEAR
    rows = []
    for i, (level, grp) in enumerate(book.groupby(pd.qcut(book[RATIO], 5), observed=True)):
        f = first.loc[grp.index]
        rows.append({"side": "whole condor", "quintile": i + 1, "range": f"{level.left:.2f} to {level.right:.2f}",
                     "condors": len(grp), "trades": 2 * len(grp), "sessions": grp["date"].nunique(),
                     "win_rate": (grp[col] > 0).mean() * 100, "avg_pl": grp[col].mean(), "total_pl": grp[col].sum(),
                     f"avg_pl_to_{fv.SPLIT_YEAR}": grp.loc[f, col].mean(),
                     f"avg_pl_from_{fv.SPLIT_YEAR + 1}": grp.loc[~f, col].mean()})
    return pd.DataFrame(rows)


def ratio_caps(panel: pd.DataFrame, cal: fv.Calendars) -> pd.DataFrame:
    cfg, col = before("ratio")
    base = fv.version_mask(panel, **cfg)
    rows = []
    for cap in fv.SL_CAPS:
        keep = base if cap is None else base & (panel[RATIO] <= cap)
        rows.append({"side": "whole condor", "max_ratio": fv._fmt(cap), "kept_pct": keep.sum() / base.sum() * 100,
                     **fv.score(panel, keep, cal, col)})
    return pd.DataFrame(rows).set_index(["side", "max_ratio"])[["kept_pct", "trades", "total_pl", "sharpe", "max_drawdown",
                                                                fv.H1, fv.H2]]


def ratio_correlation(panel: pd.DataFrame, n_boot: int = 500, seed: int = fv.SEED) -> dict[str, pd.DataFrame]:
    """The ratio's rank correlation with P/L on the book before the ratio step (with an interval), its hump,
    what it moves with, and its correlation by year."""
    cfg, col = before("ratio")
    book = panel[fv.version_mask(panel, **cfg)]
    rng = np.random.default_rng(seed)
    groups = book.groupby("date").indices
    days = np.array(list(groups))
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[d] for d in rng.choice(days, len(days))])
        s = book.iloc[idx]
        boots.append(s[RATIO].corr(s[col], method="spearman"))
    daily = book.groupby("date").agg(r=(RATIO, "mean"), pl=(col, "sum"))
    cut = book[RATIO].quantile(0.6)
    lo, hi = book[book[RATIO] <= cut], book[book[RATIO] > cut]
    summary = pd.DataFrame({"whole condor": {
        "spearman_condor": book[RATIO].corr(book[col], method="spearman"), "ci_low": np.percentile(boots, 2.5),
        "ci_high": np.percentile(boots, 97.5), "spearman_session": daily["r"].corr(daily["pl"], method="spearman"),
        "pearson_condor": book[RATIO].corr(book[col]), "p60_ratio": cut,
        "spearman_below_p60": lo[RATIO].corr(lo[col], method="spearman"),
        "spearman_above_p60": hi[RATIO].corr(hi[col], method="spearman")}}).T
    partial = pd.DataFrame({"whole condor": {ctrl: fv._partial_spearman(book, RATIO, col, cols)
                                             for ctrl, cols in fv.SL_CONTROLS.items()}})
    partial.index.name = "controlling for"
    related = book[[RATIO] + fv.SL_RELATED].corr(method="spearman").loc[fv.SL_RELATED, [RATIO]]
    related.columns = ["whole condor"]
    by_year = book.groupby(book["date"].dt.year).apply(lambda g: g[RATIO].corr(g[col], method="spearman")).to_frame("whole condor")
    return {"summary": summary, "partial": partial, "related": related, "by_year": by_year}


# --------------------------------------------------------------------------- #
# step 5: weekdays
# --------------------------------------------------------------------------- #
def weekdays(panel: pd.DataFrame) -> pd.DataFrame:
    cfg, col = before("thursday")
    book = panel[fv.version_mask(panel, **cfg)]
    daily = book.groupby(["date", "dow"])[col].sum().reset_index()
    rows = []
    for day in fv.WEEKDAYS:
        d = daily.loc[daily["dow"] == day, col]
        lo, hi = fv._session_mean_ci(d)
        rows.append({"weekday": day, "sessions": len(d), "trades": 2 * int((book["dow"] == day).sum()),
                     "avg_session_pl": d.mean(), "ci_low": lo, "ci_high": hi,
                     "winning_sessions_pct": (d > 0).mean() * 100, "total_pl": d.sum()})
    return pd.DataFrame(rows).set_index("weekday")


# --------------------------------------------------------------------------- #
# step 6: the tranches
# --------------------------------------------------------------------------- #
def _tranche_book(panel: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    cfg, col = before("tranches")
    book = panel[fv.version_mask(panel, **cfg)].copy()
    book["tranche"] = book["entry_time"].str[:5]
    return book, col


def _daily_by_tranche(book: pd.DataFrame, col: str, cal: fv.Calendars) -> pd.DataFrame:
    return book.pivot_table(index="date", columns="tranche", values=col, aggfunc="sum").reindex(
        cal.all, fill_value=0).reindex(columns=fv.ALL_TRANCHES, fill_value=0).fillna(0)


def tranche_table(panel: pd.DataFrame, cal: fv.Calendars) -> pd.DataFrame:
    """Each tranche on the book before the tranche step: trades, avg P/L, total, and the Sharpe change if dropped."""
    book, col = _tranche_book(panel)
    cfg, _ = before("tranches")
    keep_all = fv.version_mask(panel, **cfg)
    ref = fv.score(panel, keep_all, cal, col)
    rows = []
    for t in fv.ALL_TRANCHES:
        g = book[book["tranche"] == t]
        lo, hi = rs.session_bootstrap_mean(g[col], g["date"]) if len(g) else (np.nan, np.nan)
        without = fv.score(panel, keep_all & (panel["entry_time"].str[:5] != t), cal, col)
        rows.append({"entry_time": t, "condors": len(g), "trades": 2 * len(g), "avg_pl": g[col].mean(),
                     "ci_low": lo, "ci_high": hi, "total_pl": g[col].sum(),
                     "sharpe_if_dropped": without["sharpe"] - ref["sharpe"], "status": fv.tranche_status(t)})
    return pd.DataFrame(rows).set_index("entry_time")


def tranche_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily-P/L correlation between every pair of tranches on the book before the tranche step (all 20)."""
    book, col = _tranche_book(panel)
    pv = book.pivot_table(index="date", columns="tranche", values=col, aggfunc="sum")
    corr = pv.corr(min_periods=fv.TRANCHE_MIN_SESSIONS)
    corr.attrs["sessions"] = pv.notna().sum()
    return corr


def _set_stats(book: pd.DataFrame, daily: pd.DataFrame, tranches: list[str], first: np.ndarray, col: str) -> dict:
    d = daily[tranches].sum(axis=1).to_numpy()
    eq = np.cumsum(d)
    rows = book[book["tranche"].isin(tranches)]
    call, put = _sides(rows, col)
    sides = pd.concat([call, put]).dropna()
    return {"tranches": len(tranches), "condors": len(rows), "trades": 2 * len(rows), "total_pl": float(d.sum()),
            "sharpe": br.sharpe(d), fv.H1: br.sharpe(d[first]), fv.H2: br.sharpe(d[~first]),
            "max_drawdown": float((eq - np.maximum.accumulate(eq)).min()),
            "win_rate_spread": float((sides > 0).mean() * 100), "win_rate_condor": float((rows[col] > 0).mean() * 100),
            "avg_pl_condor": float(rows[col].mean())}


PREVIOUS_LIVE = fv.PREVIOUS_TRANCHES   # the 15 traded before the 7


def tranche_selection(panel: pd.DataFrame, cal: fv.Calendars) -> pd.DataFrame:
    """Forward selection at the tranche step: start with the tranche whose own Sharpe is highest, then add the one
    that gives the book the highest Sharpe, until all 20 are in. ``attrs["best"]`` is the set at the top."""
    book, col = _tranche_book(panel)
    daily = _daily_by_tranche(book, col, cal)
    first = pd.Series(cal.all).dt.year.to_numpy() <= fv.SPLIT_YEAR
    rows, chosen, trials = [], [], []
    for step, (t, _) in enumerate(fv._forward_path(daily, trials), start=1):
        chosen.append(t)
        rows.append({"step": step, "added": t, **_set_stats(book, daily, list(chosen), first, col)})
    out = pd.DataFrame(rows).set_index("step")
    best = int(out["sharpe"].idxmax())
    picked = list(out["added"].iloc[:best])
    # The strategy's tranche set is fixed; this selection is re-run on the book before the tranche step to show
    # what it would pick there.
    tried = pd.DataFrame(trials, index=pd.Index(range(1, len(trials) + 1), name="step"))[fv.ALL_TRANCHES]
    out.attrs.update(best=picked, best_step=best, trials=tried,
                     live=_set_stats(book, daily, PREVIOUS_LIVE, first, col), live_set=PREVIOUS_LIVE,
                     strategy=_set_stats(book, daily, fv.SELECTED_TRANCHES, first, col),
                     all20=_set_stats(book, daily, fv.ALL_TRANCHES, first, col),
                     matches_strategy=sorted(picked) == sorted(fv.SELECTED_TRANCHES))
    return out


def _pl_per_dd(d: np.ndarray) -> float:
    eq = np.cumsum(d)
    dd = float((eq - np.maximum.accumulate(np.maximum(eq, 0))).min())
    return float(d.sum() / -dd) if dd < 0 else np.nan


def tranche_selection_dd(panel: pd.DataFrame, cal: fv.Calendars) -> pd.DataFrame:
    """Forward selection of the tranches by drawdown, on its own terms: every other setting of the strategy on (the
    ratio cap included), all 20 tranches available, nothing taken from the strategy's own tranche set. Start with
    the tranche whose own P/L per $ of max drawdown is highest, then add the tranche that gives the book the highest
    P/L per $ of max drawdown, until all 20 are in. (Minimising the drawdown alone would pick the tranches that
    barely trade.) ``attrs["best"]`` is the set at the top of that ratio; ``attrs["trials"]`` each step's ratio for
    every tranche tried."""
    book = panel[fv.version_mask(panel, **{**fv.STRATEGY, "drop": ()})].copy()
    book["tranche"] = book["entry_time"].str[:5]
    col = fv.PL
    daily = _daily_by_tranche(book, col, cal)
    first = pd.Series(cal.all).dt.year.to_numpy() <= fv.SPLIT_YEAR
    m = daily.to_numpy()
    idx = {t: i for i, t in enumerate(daily.columns)}
    left, chosen, rows, trials = list(daily.columns), [], [], []
    total = np.zeros(len(daily))
    while left:
        scores = {t: _pl_per_dd(total + m[:, idx[t]]) for t in left}
        best = max(scores, key=lambda t: (np.nan_to_num(scores[t], nan=-np.inf), t))
        trials.append(scores)
        chosen.append(best)
        left.remove(best)
        total = total + m[:, idx[best]]
        rows.append({"step": len(chosen), "added": best, **_set_stats(book, daily, list(chosen), first, col),
                     "pl_per_dd": scores[best]})
    out = pd.DataFrame(rows).set_index("step")
    top = int(out["pl_per_dd"].idxmax())
    tried = pd.DataFrame(trials, index=pd.Index(range(1, len(trials) + 1), name="step"))[fv.ALL_TRANCHES]
    out.attrs.update(best=list(out["added"].iloc[:top]), best_step=top, trials=tried,
                     all20=_set_stats(book, daily, fv.ALL_TRANCHES, first, col))
    return out


def fig_selection_dd(dd: pd.DataFrame, note: str):
    """Forward selection by drawdown: P/L per $ of drawdown, max drawdown and Sharpe at each step (dotted: all 20)."""
    with plt.rc_context(fv.STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        x = dd.index.to_numpy()
        b = dd.attrs["best_step"]
        for ax, col, title in ((axes[0], "pl_per_dd", "P/L per $ of max drawdown"), (axes[1], "max_drawdown", "Max drawdown"),
                               (axes[2], "sharpe", "Sharpe")):
            ax.plot(x, dd[col], color=fv.ORANGE, marker="s", ms=5, lw=1.8)
            ax.scatter([b], [dd.loc[b, col]], s=130, facecolor="none", edgecolor=fv.INK, lw=1.3, zorder=4)
            ax.set_xticks(x, [f"+{t}" for t in dd["added"]], rotation=90, fontsize=7.5)
            ax.set_title(title)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fv._k(v)))
        axes[2].axhline(dd.attrs["all20"]["sharpe"], color=fv.INK_2, ls=":", lw=1.2)
        axes[2].set_ylim(bottom=max(-1, dd["sharpe"].min() - 0.2))
        axes[0].annotate(f"picked: {b} tranches, {dd.loc[b, 'pl_per_dd']:.1f}", (b, dd.loc[b, "pl_per_dd"]),
                         xytext=(8, -14), textcoords="offset points", fontsize=8, color=fv.INK)
        fig.suptitle(note, x=0.01, ha="left")
        fig.tight_layout()
    return fig


def tranche_walk_forward(panel: pd.DataFrame, cal: fv.Calendars) -> pd.DataFrame:
    """The selection checked out of sample: chosen on every earlier year, traded in the next, against the full-period
    pick (the strategy's set, chosen with hindsight) and all 20 tranches."""
    book, col = _tranche_book(panel)
    daily = _daily_by_tranche(book, col, cal)
    years = pd.Series(daily.index).dt.year.to_numpy()
    rows, oos = [], {"selected": [], "live": [], "all20": []}
    for y in fv.WF_TEST_YEARS:
        train, test = daily[years < y], daily[years == y]
        path = fv._forward_path(train)
        best = int(np.nanargmax([sh for _, sh in path])) + 1
        chosen = [t for t, _ in path[:best]]
        for k, ts in (("selected", chosen), ("live", fv.SELECTED_TRANCHES), ("all20", fv.ALL_TRANCHES)):
            oos[k].append(test[ts].sum(axis=1))
        rows.append({"test_year": str(y), "train": f"{daily.index[years < y].min():%Y-%m} to {y - 1}",
                     "tranches_selected": len(chosen),
                     "dropped_by_selection": ", ".join(t for t in fv.ALL_TRANCHES if t not in chosen) or "-",
                     "selected_sharpe": br.sharpe(test[chosen].sum(axis=1)),
                     "live_sharpe": br.sharpe(test[fv.SELECTED_TRANCHES].sum(axis=1)),
                     "all20_sharpe": br.sharpe(test[fv.ALL_TRANCHES].sum(axis=1)),
                     "selected_pl": float(test[chosen].sum().sum()),
                     "live_pl": float(test[fv.SELECTED_TRANCHES].sum().sum()),
                     "all20_pl": float(test[fv.ALL_TRANCHES].sum().sum())})
    j = {k: pd.concat(v).to_numpy() for k, v in oos.items()}
    rows.append({"test_year": "out of sample", "train": "expanding", "tranches_selected": np.nan, "dropped_by_selection": "-",
                 "selected_sharpe": br.sharpe(j["selected"]), "live_sharpe": br.sharpe(j["live"]),
                 "all20_sharpe": br.sharpe(j["all20"]), "selected_pl": float(j["selected"].sum()),
                 "live_pl": float(j["live"].sum()), "all20_pl": float(j["all20"].sum())})
    out = pd.DataFrame(rows).set_index("test_year")
    lo, hi, p = fv._paired_boot(j["all20"], j["selected"])
    out.attrs.update(selected_vs_all20_ci=(lo, hi), p_selected_better=p)
    return out


def tranche_sets(panel: pd.DataFrame, cal: fv.Calendars, selection: pd.DataFrame, selection_dd: pd.DataFrame) -> pd.DataFrame:
    """With every other setting on: the strategy's tranches against forward selection's peak, the previous live
    set and all 20 (trades, P/L, Sharpe per half, drawdown, and P/L per $ of drawdown)."""
    sets = {f"the strategy's {len(fv.SELECTED_TRANCHES)}": fv.SELECTED_TRANCHES,
            f"forward selection's peak ({len(selection.attrs['best'])})": selection.attrs["best"],
            f"forward selection by drawdown ({len(selection_dd.attrs['best'])})": selection_dd.attrs["best"],
            f"previous live set ({len(fv.PREVIOUS_TRANCHES)})": fv.PREVIOUS_TRANCHES, "all 20": fv.ALL_TRANCHES}
    rows = {}
    for name, ts in sets.items():
        drop = tuple(f"{t}:00" for t in fv.ALL_TRANCHES if t not in ts)
        st = fv.score(panel, fv.version_mask(panel, **{**fv.STRATEGY, "drop": drop}), cal, fv.PL)
        rows[name] = {"tranches": ", ".join(sorted(ts)) if len(ts) < 20 else "every entry time",
                      **{k: st[k] for k in ("trades", "total_pl", "sharpe", fv.H1, fv.H2, "max_drawdown")},
                      "pl_per_dd": st["total_pl"] / -st["max_drawdown"]}
    return pd.DataFrame(rows).T


def fig_tranches(tt: pd.DataFrame, note: str):
    """Avg P/L per condor by tranche on the book before the tranche step; the tranches the selection drops hatched."""
    with plt.rc_context(fv.STYLE):
        fig, ax = plt.subplots(figsize=(15, 4.4))
        x = np.arange(len(tt))
        out = (tt["status"] == "dropped").to_numpy()
        ax.bar(x, tt["avg_pl"], width=0.66, color=[fv.ORANGE if o else fv.BLUE for o in out],
               hatch=["///" if o else "" for o in out], edgecolor="white", lw=0)
        ax.errorbar(x, tt["avg_pl"], yerr=[tt["avg_pl"] - tt["ci_low"], tt["ci_high"] - tt["avg_pl"]], fmt="none",
                    ecolor=fv.INK_2, lw=0.8, capsize=2)
        ax.axhline(0, color=fv.INK_2, lw=0.7)
        ax.set_xticks(x, tt.index, rotation=90)
        ax.set_ylabel("$ per condor")
        ax.set_title("Avg P/L per condor, by tranche (whiskers: 95% session-bootstrap CI)")
        ax.legend([plt.Rectangle((0, 0), 1, 1, color=fv.BLUE),
                   plt.Rectangle((0, 0), 1, 1, facecolor=fv.ORANGE, hatch="///", edgecolor="white")],
                  ["picked by forward selection", "dropped"], loc="upper left", fontsize=8)
        fig.suptitle(note, x=0.01, ha="left")
        fig.tight_layout()
    return fig


def fig_walk_forward(fwd: pd.DataFrame, wf: pd.DataFrame, note: str):
    """The forward-selection path, and the walk-forward test years (selected each year vs the full-period set vs all 20)."""
    with plt.rc_context(fv.STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
        ax = axes[0]
        x = fwd.index.to_numpy()
        ax.plot(x, fwd["sharpe"], color=fv.BLUE, marker="o", ms=5.5, label="all sessions")
        for h, style in fv.HALF_STYLE.items():
            ax.plot(x, fwd[h], **style)
        b = fwd.attrs["best_step"]
        ax.scatter([b], [fwd.loc[b, "sharpe"]], s=130, facecolor="none", edgecolor=fv.INK, lw=1.4, zorder=4)
        ax.annotate(f"picked: {b} tranches, Sharpe {fwd.loc[b, 'sharpe']:.2f}", (b, fwd.loc[b, "sharpe"]),
                    xytext=(8, -18), textcoords="offset points", fontsize=8, color=fv.INK)
        ax.axhline(fwd.attrs["live"]["sharpe"], color=fv.INK, ls=":", lw=1.4,
                   label=f"previous live set ({len(fwd.attrs['live_set'])}): {fwd.attrs['live']['sharpe']:.2f}")
        ax.set_xticks(x, [f"+{t}" for t in fwd["added"]], rotation=90, fontsize=8)
        ax.set_ylabel("Sharpe of the book")
        ax.set_title("Forward selection: best tranche first, then the one that adds most Sharpe")
        ax.legend(loc="lower right", fontsize=8)
        ax = axes[1]
        w = wf.drop(index="out of sample")
        xx = np.arange(len(w))
        for k, (c, color, label) in enumerate((("selected_sharpe", fv.BLUE, "chosen on earlier years"),
                                                ("live_sharpe", fv.ORANGE, "the strategy's set"),
                                                ("all20_sharpe", fv.GREY, "all 20"))):
            ax.bar(xx + (k - 1) * 0.26, w[c], width=0.24, color=color, label=label)
        ax.axhline(0, color=fv.INK_2, lw=0.7)
        ax.set_xticks(xx, w.index)
        ax.set_ylabel("Sharpe in the test year")
        ax.set_title("Walk-forward: choose on the past, trade the next year")
        ax.legend(loc="upper left", fontsize=8)
        fig.suptitle(note, x=0.01, ha="left")
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# everything, computed once
# --------------------------------------------------------------------------- #
@dataclass
class Cascade:
    greedy: pd.DataFrame
    table: pd.DataFrame
    profiles: dict[str, pd.DataFrame]
    sweeps: dict[str, pd.DataFrame]
    trend: pd.DataFrame
    sma_ab: pd.DataFrame
    sma_sweeps: dict[str, pd.DataFrame]
    sma_ab_dd: pd.DataFrame
    ratio_quintiles: pd.DataFrame
    ratio_caps: pd.DataFrame
    ratio_corr: dict[str, pd.DataFrame]
    weekdays: pd.DataFrame
    tranches: pd.DataFrame
    tranche_corr: pd.DataFrame
    selection: pd.DataFrame
    walk_forward: pd.DataFrame
    tranche_sets: pd.DataFrame
    selection_dd: pd.DataFrame


def compute(panel: pd.DataFrame) -> Cascade:
    cal = fv.Calendars.of(panel)
    sel, sel_dd = tranche_selection(panel, cal), tranche_selection_dd(panel, cal)
    return Cascade(
        greedy=greedy(panel, cal),
        table=cascade_table(panel, cal),
        profiles={"band": profile(panel, fv.TS_9D_30, "band"), "move": profile(panel, "spx_move_pct", "move"),
                  "sma10": profile(panel, "spx_vs_sma10_pct", "sma10"),
                  "sma20": profile(panel, "spx_vs_sma20_pct", "sma10")},
        sweeps={"ts_low": sweep(panel, cal, "ts_low", "band"), "ts_high": sweep(panel, cal, "ts_high", "band"),
                "max_move": sweep(panel, cal, "max_move", "move"), "sma10_min": sweep(panel, cal, "sma10_min", "sma10")},
        trend=trend(panel, cal), sma_ab=sma_ab(panel, cal), sma_sweeps=sma_sweeps(panel, cal),
        ratio_quintiles=ratio_quintiles(panel), ratio_caps=ratio_caps(panel, cal), ratio_corr=ratio_correlation(panel),
        weekdays=weekdays(panel), tranches=tranche_table(panel, cal), tranche_corr=tranche_correlation(panel),
        selection=sel, walk_forward=tranche_walk_forward(panel, cal), tranche_sets=tranche_sets(panel, cal, sel, sel_dd), selection_dd=sel_dd,
        sma_ab_dd=sma_ab(panel, cal, sel_dd.attrs["best"]),
    )
