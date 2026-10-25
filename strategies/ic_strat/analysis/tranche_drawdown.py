"""Drawdowns by tranche, inside the finished strategy: where the losses come from, and where to cut next.

Everything here is on the strategy as traded (every filter on, held to expiry, ``final_version.PL``),
scored on all sessions with no-trade days as zero, Sharpe the backtester's.

- ``daily_matrix``: date x tranche P/L for the strategy's tranches.
- ``episodes``: the book's worst drawdowns (peak, trough, recovery, depth).
- ``episode_contrib``: each tranche's P/L inside each of those drawdowns.
- ``tranche_drawdown_table``: each tranche's own drawdown and what the book does without it.
- ``by_year``: each tranche's P/L by year.
- ``backward_elimination``: from the strategy's set, drop one tranche at a time, by Sharpe or by drawdown.
- ``elimination_walk_forward``: the cut chosen on earlier years only, traded in the next year.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

from . import backtester_rules as br
from . import final_version as fv

N_EPISODES = 5        # the book's worst drawdowns studied
WORST_DAYS = 20       # the book's worst sessions
CRITERIA = {"sharpe": "cut by Sharpe", "drawdown": "cut by drawdown"}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def strategy_book(panel: pd.DataFrame) -> pd.DataFrame:
    """The strategy's condors (every setting on), tranche as ``HH:MM``."""
    book = panel[fv.version_mask(panel, **fv.STRATEGY)].copy()
    book["tranche"] = book["entry_time"].str[:5]
    return book


def daily_matrix(book: pd.DataFrame, cal: fv.Calendars) -> pd.DataFrame:
    """Session x tranche P/L (with the call stop) on every session, the strategy's tranches."""
    return book.pivot_table(index="date", columns="tranche", values=fv.PL, aggfunc="sum").reindex(
        cal.all).reindex(columns=fv.SELECTED_TRANCHES).fillna(0.0)


def _underwater(d: np.ndarray) -> np.ndarray:
    eq = np.cumsum(d)
    return eq - np.maximum.accumulate(np.maximum(eq, 0))


def max_drawdown(d: np.ndarray) -> float:
    return float(_underwater(np.asarray(d, dtype=float)).min())


def _sharpe(d: np.ndarray) -> float:
    """The backtester's Sharpe, blank for fewer than two sessions (a half with no data)."""
    return br.sharpe(d) if len(d) > 1 else np.nan


def _ret_dd(d: np.ndarray) -> float:
    dd = max_drawdown(d)
    return float(d.sum() / -dd) if dd < 0 else np.nan


# --------------------------------------------------------------------------- #
# the book's drawdowns, and who is in them
# --------------------------------------------------------------------------- #
def episodes(daily: pd.DataFrame, n: int = N_EPISODES) -> pd.DataFrame:
    """The book's ``n`` deepest drawdowns: last peak, trough, recovery (first session back at the peak), depth."""
    d = daily.sum(axis=1)
    uw = pd.Series(_underwater(d.to_numpy()), index=d.index)
    under = uw < 0
    seg = (under != under.shift(fill_value=False)).cumsum()[under]
    rows = []
    for _, s in uw[under].groupby(seg):
        start, end = s.index[0], s.index[-1]
        i0 = d.index.get_loc(start)
        peak = d.index[i0 - 1] if i0 > 0 else start
        i1 = d.index.get_loc(end)
        rows.append({"peak": peak, "trough": s.idxmin(), "recovered": d.index[i1 + 1] if i1 + 1 < len(d) else pd.NaT,
                     "depth": float(s.min()), "sessions_to_trough": int(d.index.get_loc(s.idxmin()) - i0 + 1),
                     "sessions_under": len(s)})
    out = pd.DataFrame(rows).sort_values("depth").head(n).reset_index(drop=True)
    out.index = pd.Index([f"DD{i}" for i in range(1, len(out) + 1)], name="drawdown")
    return out


def episode_contrib(daily: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    """Each tranche's P/L from each drawdown's peak to its trough (the rows sum to the drawdown's depth)."""
    cols = {}
    for k, e in eps.iterrows():
        win = (daily.index > e["peak"]) & (daily.index <= e["trough"])
        cols[k] = daily[win].sum()
    out = pd.DataFrame(cols)
    out.loc["book"] = out.sum()
    return out


def tranche_drawdown_table(book: pd.DataFrame, daily: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    """Per tranche: its own P/L and drawdown (and P/L per $ of drawdown: a single tranche's Sharpe is meaningless
    here, the 4% hurdle on $200k swamps it), its share of the book's losses in the worst drawdowns and
    worst days, and what the book's Sharpe, P/L and drawdown become without it."""
    total = daily.sum(axis=1).to_numpy()
    ref = {"sharpe": br.sharpe(total), "pl": total.sum(), "dd": max_drawdown(total)}
    contrib = episode_contrib(daily, eps)
    worst = np.argsort(total)[:WORST_DAYS]
    rows = []
    for t in daily.columns:
        d = daily[t].to_numpy()
        g = book[book["tranche"] == t]
        rest = total - d
        rows.append({
            "tranche": t, "trades": 2 * len(g), "total_pl": d.sum(), "max_drawdown": max_drawdown(d),
            "pl_per_dd": _ret_dd(d), "worst_day": d.min(),
            "pl_in_worst_dd": contrib.loc[t, "DD1"],
            "share_of_worst_dd_pct": 100 * contrib.loc[t, "DD1"] / contrib.loc["book", "DD1"],
            f"pl_on_{WORST_DAYS}_worst_days": d[worst].sum(),
            "pl_in_top_dds": contrib.loc[t].sum(),
            "sharpe_if_dropped": br.sharpe(rest) - ref["sharpe"],
            "pl_if_dropped": rest.sum() - ref["pl"],
            "max_dd_if_dropped": max_drawdown(rest) - ref["dd"],
        })
    return pd.DataFrame(rows).set_index("tranche")


def by_year(daily: pd.DataFrame) -> pd.DataFrame:
    """Tranche x year total P/L (2022 and 2026 are part years), with each tranche's losing years counted."""
    out = daily.groupby(daily.index.year).sum().T
    out.columns = [str(c) for c in out.columns]
    out["losing_years"] = (out < 0).sum(axis=1)
    return out


# --------------------------------------------------------------------------- #
# cutting further: backward elimination, and a walk-forward check of it
# --------------------------------------------------------------------------- #
def _score(d: np.ndarray, criterion: str) -> float:
    """Higher is better: Sharpe, or the max drawdown (closer to zero)."""
    return br.sharpe(d) if criterion == "sharpe" else max_drawdown(d)


def _elimination_path(daily: pd.DataFrame, criterion: str) -> list[str]:
    """Drop order: at each step remove the tranche whose removal gives the best book by ``criterion``."""
    left, order = list(daily.columns), []
    m = daily.to_numpy()
    idx = {t: i for i, t in enumerate(daily.columns)}
    while len(left) > 1:
        total = m[:, [idx[t] for t in left]].sum(axis=1)
        scores = {t: _score(total - m[:, idx[t]], criterion) for t in left}
        worst = max(scores, key=lambda t: (np.nan_to_num(scores[t], nan=-np.inf), t))
        order.append(worst)
        left.remove(worst)
    return order


def _pick(path: pd.DataFrame, criterion: str) -> int:
    """Where to stop: the highest Sharpe, or (cutting by drawdown) the best P/L per $ of drawdown."""
    return int(path["sharpe"].idxmax() if criterion == "sharpe" else path["pl_per_dd"].idxmax())


def backward_elimination(daily: pd.DataFrame, criterion: str) -> pd.DataFrame:
    """From the strategy's set, drop one tranche at a time; one row per number of tranches kept (all = the strategy)."""
    first = pd.Series(daily.index).dt.year.to_numpy() <= fv.SPLIT_YEAR
    order = _elimination_path(daily, criterion)
    rows, kept = [], list(daily.columns)
    for removed in [None, *order]:
        if removed:
            kept.remove(removed)
        d = daily[kept].sum(axis=1).to_numpy()
        rows.append({"kept": len(kept), "removed": removed or "-", "total_pl": d.sum(), "sharpe": br.sharpe(d),
                     fv.H1: _sharpe(d[first]), fv.H2: _sharpe(d[~first]), "max_drawdown": max_drawdown(d),
                     "pl_per_dd": _ret_dd(d), "tranches": ", ".join(kept)})
    out = pd.DataFrame(rows).set_index("kept")
    best = _pick(out, criterion)
    out.attrs.update(criterion=criterion, best=best, best_set=out.loc[best, "tranches"].split(", "),
                     cut=[t for t in daily.columns if t not in out.loc[best, "tranches"].split(", ")])
    return out


def elimination_walk_forward(daily: pd.DataFrame, criterion: str) -> pd.DataFrame:
    """For each test year: cut on every earlier year only (same rule and stopping point), trade the next year,
    against keeping the strategy's set."""
    years = pd.Series(daily.index).dt.year.to_numpy()
    rows, oos_cut, oos_all = [], [], []
    for y in fv.WF_TEST_YEARS:
        train, test = daily[years < y], daily[years == y]
        be = backward_elimination(train, criterion)
        keep = be.attrs["best_set"]
        dc, da = test[keep].sum(axis=1).to_numpy(), test.sum(axis=1).to_numpy()
        oos_cut.append(dc)
        oos_all.append(da)
        rows.append({"test_year": str(y), "kept": len(keep), "cut": ", ".join(be.attrs["cut"]) or "-",
                     "cut_sharpe": br.sharpe(dc), "strategy_set_sharpe": br.sharpe(da), "cut_pl": dc.sum(), "strategy_set_pl": da.sum(),
                     "cut_max_dd": max_drawdown(dc), "strategy_set_max_dd": max_drawdown(da)})
    c, a = np.concatenate(oos_cut), np.concatenate(oos_all)
    rows.append({"test_year": "out of sample", "kept": np.nan, "cut": "-", "cut_sharpe": br.sharpe(c),
                 "strategy_set_sharpe": br.sharpe(a), "cut_pl": c.sum(), "strategy_set_pl": a.sum(),
                 "cut_max_dd": max_drawdown(c), "strategy_set_max_dd": max_drawdown(a)})
    out = pd.DataFrame(rows).set_index("test_year")
    lo, hi, p = fv._paired_boot(a, c)
    out.attrs.update(criterion=criterion, cut_vs_strategy_set_ci=(lo, hi), p_cut_better=p)
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_underwater(daily: pd.DataFrame, eps: pd.DataFrame):
    """The strategy's equity curve and its underwater curve, the worst drawdowns shaded and numbered."""
    d = daily.sum(axis=1)
    eq, uw = d.cumsum(), pd.Series(_underwater(d.to_numpy()), index=d.index)
    with plt.rc_context(fv.STYLE):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
        a1.plot(eq.index, eq, color=fv.BLUE, lw=1.6)
        a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fv._k(v)))
        a1.set_title(f"Equity: {fv._k(eq.iloc[-1])} over {len(d)} sessions")
        a2.fill_between(uw.index, uw, 0, color=fv.RED, alpha=0.35, lw=0)
        a2.plot(uw.index, uw, color=fv.RED, lw=0.9)
        a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fv._k(v)))
        a2.set_title("Underwater: distance below the last equity peak")
        for k, e in eps.iterrows():
            end = e["recovered"] if pd.notna(e["recovered"]) else d.index[-1]
            for ax in (a1, a2):
                ax.axvspan(e["peak"], end, color=fv.GREY, alpha=0.18, lw=0)
            a2.annotate(f"{k}\n{fv._k(e['depth'])}", (e["trough"], e["depth"]), xytext=(0, -4),
                        textcoords="offset points", ha="center", va="top", fontsize=7.5, color=fv.INK)
        a2.set_ylim(eps["depth"].min() * 1.45, 0)
        fig.suptitle(f"The strategy's drawdowns ({int(daily.columns.size)} tranches, held to expiry): "
                     f"the {len(eps)} deepest shaded, peak to recovery", x=0.01, ha="left")
        fig.tight_layout()
    return fig


_DIVERGING = LinearSegmentedColormap.from_list("loss_gain", [fv.RED, "#f4f3ef", fv.BLUE])


def fig_episode_heatmap(contrib: pd.DataFrame, eps: pd.DataFrame):
    """Tranche x drawdown: each tranche's P/L from peak to trough (red = lost money in that drawdown)."""
    m = contrib.drop(index="book")
    with plt.rc_context(fv.STYLE):
        fig, ax = plt.subplots(figsize=(10, 7.2))
        lim = np.abs(m.to_numpy()).max()
        ax.imshow(m.to_numpy(), cmap=_DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.iat[i, j]
                ax.text(j, i, f"{v / 1000:+.1f}k", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v) > 0.6 * lim else fv.INK)
        labels = [f"{k}: {fv._k(e['depth'])}\n{e['peak']:%Y-%m-%d} to {e['trough']:%Y-%m-%d}" for k, e in eps.iterrows()]
        ax.set_xticks(range(m.shape[1]), labels, fontsize=7.5)
        ax.set_yticks(range(m.shape[0]), m.index)
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("Each tranche's P/L from peak to trough in the book's deepest drawdowns ($; red = lost)")
        fig.tight_layout()
    return fig


def fig_by_year(yr: pd.DataFrame):
    """Tranche x year P/L heatmap (red = a losing year)."""
    m = yr.drop(columns="losing_years")
    with plt.rc_context(fv.STYLE):
        fig, ax = plt.subplots(figsize=(8, 7))
        lim = np.abs(m.to_numpy()).max()
        ax.imshow(m.to_numpy(), cmap=_DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.iat[i, j]
                ax.text(j, i, f"{v / 1000:+.1f}k", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 * lim else fv.INK)
        ax.set_xticks(range(m.shape[1]), [f"{c}{' (part)' if c in ('2022', '2026') else ''}" for c in m.columns])
        ax.set_yticks(range(m.shape[0]), [f"{t}  ({n} losing)" if n else t for t, n in zip(m.index, yr["losing_years"])])
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("Each tranche's P/L by year ($; red = a losing year)")
        fig.tight_layout()
    return fig


def fig_tranche_underwater(daily: pd.DataFrame):
    """Small multiples: each tranche's own underwater curve, on one shared scale."""
    n = daily.shape[1]
    cols = 5
    rows = int(np.ceil(n / cols))
    uws = {t: pd.Series(_underwater(daily[t].to_numpy()), index=daily.index) for t in daily.columns}
    low = min(u.min() for u in uws.values())
    with plt.rc_context(fv.STYLE):
        fig, axes = plt.subplots(rows, cols, figsize=(16, 2.3 * rows), sharex=True, sharey=True)
        for ax, t in zip(axes.flat, daily.columns):
            u = uws[t]
            ax.fill_between(u.index, u, 0, color=fv.RED, alpha=0.35, lw=0)
            ax.plot(u.index, u, color=fv.RED, lw=0.7)
            ax.set_title(f"{t}: {fv._k(daily[t].sum())}, max DD {fv._k(u.min())}".replace("$", r"\$"), fontsize=9)
            ax.set_ylim(low * 1.08, 0)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fv._k(v)))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(axis="x", labelsize=7.5)
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        fig.suptitle("Each tranche's own underwater curve (total P/L and max drawdown in the title; one shared scale)",
                     x=0.01, ha="left")
        fig.tight_layout()
    return fig


def fig_elimination(paths: dict[str, pd.DataFrame]):
    """Backward elimination from the strategy's set: Sharpe, P/L and max drawdown by the number of tranches kept."""
    colors = {"sharpe": fv.BLUE, "drawdown": fv.ORANGE}
    markers = {"sharpe": "o", "drawdown": "s"}
    with plt.rc_context(fv.STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
        for crit, p in paths.items():
            x = p.index.to_numpy()
            for ax, col in zip(axes, ("sharpe", "total_pl", "max_drawdown")):
                ax.plot(x, p[col], color=colors[crit], marker=markers[crit], ms=5, lw=1.6,
                        label=f"{CRITERIA[crit]} (picked: {p.attrs['best']} kept)")
                b = p.attrs["best"]
                ax.scatter([b], [p.loc[b, col]], s=120, facecolor="none", edgecolor=fv.INK, lw=1.2, zorder=4)
        for ax, title in zip(axes, ("Sharpe", "Total P/L", "Max drawdown")):
            ax.set_title(title)
            ax.set_xticks(range(1, len(fv.SELECTED_TRANCHES) + 1))
            ax.set_xlabel("tranches kept")
            ax.invert_xaxis()
            ax.axvline(len(fv.SELECTED_TRANCHES), color=fv.INK_2, ls=":", lw=1)
        for ax in axes[1:]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fv._k(v)))
        top = max(p["sharpe"].max() for p in paths.values())
        axes[0].set_ylim(-1, top + 0.3)
        axes[0].set_title("Sharpe (below -1 off the chart)")
        axes[0].legend(loc="lower left")
        n = len(fv.SELECTED_TRANCHES)
        fig.suptitle(f"Cutting the {n} one tranche at a time (dotted: the strategy's {n}; circled: where each rule stops)",
                     x=0.01, ha="left")
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# everything, computed once
# --------------------------------------------------------------------------- #
@dataclass
class TrancheDrawdowns:
    daily: pd.DataFrame
    episodes: pd.DataFrame
    contrib: pd.DataFrame
    table: pd.DataFrame
    by_year: pd.DataFrame
    paths: dict[str, pd.DataFrame]
    walk_forward: dict[str, pd.DataFrame]


def compute(panel: pd.DataFrame) -> TrancheDrawdowns:
    cal = fv.Calendars.of(panel)
    book = strategy_book(panel)
    daily = daily_matrix(book, cal)
    eps = episodes(daily)
    return TrancheDrawdowns(
        daily=daily, episodes=eps, contrib=episode_contrib(daily, eps),
        table=tranche_drawdown_table(book, daily, eps), by_year=by_year(daily),
        paths={c: backward_elimination(daily, c) for c in CRITERIA},
        walk_forward={c: elimination_walk_forward(daily, c) for c in CRITERIA},
    )
