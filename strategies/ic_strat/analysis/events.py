"""Event-day blackouts against the no-Thursday rule: is the Thursday loss just FOMC, CPI, NFP, expiries and holidays?

Every other setting of the strategy is on. The calendar (``event_calendar.json``) lists FOMC, CPI and NFP releases, monthly opex,
quad witching, VIX expiry, NYSE holidays and early closes; the blackout skips every listed date that is a session.

- ``event_days``: which event types fall on each session, and on which weekday;
- ``blackout_ab``: the strategy (no Thursdays) against Thursdays back in, with and without the blackout, with a paired session
  bootstrap and t-tests against the strategy;
- ``thursday_split``: Thursday sessions with and without an event (on the day or the day before) against the other weekdays.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import backtester_rules as br
from . import cascade as cs
from . import final_version as fv
from . import regime_study as rs

CALENDAR_FILE = Path(__file__).with_name("event_calendar.json")
# The blackout: every listed date that is a session (NYSE holidays have none, so they never bind).
SAME_DAY = ["fomc", "cpi", "nfp", "opex", "quad_witching", "vix_expiry", "nyse_early_closes"]


def load_calendar() -> dict[str, pd.DatetimeIndex]:
    return {k: pd.DatetimeIndex(pd.to_datetime(v)) for k, v in json.loads(CALENDAR_FILE.read_text()).items()}


def event_days(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Session x event flags: each event type on the day, the day after an FOMC (always a Thursday, a diagnostic, not part of
    the blackout), and ``any_event`` (the blackout: any listed date)."""
    cal = load_calendar()
    s = pd.DatetimeIndex(sorted(sessions))
    out = pd.DataFrame(index=s)
    for k in SAME_DAY:
        out[k] = s.isin(cal[k])
    prev = pd.Series(s, index=s).shift(1)
    out["day_after_fomc"] = prev.isin(cal["fomc"]).to_numpy()
    # NYSE holidays have no session: the sessions either side of one are the tradable effect (not in the live blackout).
    hol = cal["nyse_holidays"][(cal["nyse_holidays"] > s.min()) & (cal["nyse_holidays"] < s.max())]
    i = s.searchsorted(hol)
    out["around_holiday"] = s.isin(s[np.unique(np.r_[i - 1, i])])
    out["any_event"] = out[SAME_DAY].any(axis=1)
    out["weekday"] = s.dayofweek
    return out


def by_weekday(ev: pd.DataFrame) -> pd.DataFrame:
    """How many sessions of each event type fall on each weekday."""
    cols = SAME_DAY + ["around_holiday", "day_after_fomc", "any_event"]
    t = ev.groupby("weekday")[cols].sum().T
    t.columns = [fv.WEEKDAYS[d] if isinstance(fv.WEEKDAYS, dict) else ["Mon", "Tue", "Wed", "Thu", "Fri"][d] for d in t.columns]
    t["sessions"] = t.sum(axis=1)
    return t


def blackout_ab(panel: pd.DataFrame, cal: fv.Calendars) -> pd.DataFrame:
    """Everything else on: A the strategy (no Thursdays); B Thursdays back in; C Thursdays back in with every event day blacked
    out; D no Thursdays and the blackout; E Thursdays back in except the day after FOMC. Paired bootstrap and t-tests vs A."""
    ev = event_days(pd.DatetimeIndex(cal.all))
    black = panel["date"].isin(ev.index[ev["any_event"]])
    after_fomc = panel["date"].isin(ev.index[ev["day_after_fomc"]])
    strat = fv.version_mask(panel, **fv.STRATEGY)
    thu_in = fv.version_mask(panel, **{**fv.STRATEGY, "skip_thursday": False})
    versions = {"A: no Thursdays (strategy)": strat,
                "B: Thursdays back in": thu_in,
                "C: Thursdays back in, event days blacked out": thu_in & ~black,
                "D: no Thursdays, event days blacked out": strat & ~black,
                "E: Thursdays back in except the day after FOMC": thu_in & ~(after_fomc & (panel["dow_code"] == fv.THURSDAY))}
    daily = {k: rs.daily_series(panel.loc[m, fv.PL], panel.loc[m, "date"], cal.all).to_numpy() for k, m in versions.items()}
    ref = next(iter(versions))
    rows = {}
    for k, m in versions.items():
        st = cs._stats(panel, m, fv.PL, cal)
        lo, hi, pb = (np.nan,) * 3 if k == ref else fv._paired_boot(daily[ref], daily[k])
        tt = (dict.fromkeys(["t_daily_pl", "p_daily_pl", "t_sharpe", "p_sharpe"], np.nan) if k == ref
              else cs.ab_t_tests(daily[ref], daily[k]))
        rows[k] = {**st, "sharpe_vs_A": br.sharpe(daily[k]) - br.sharpe(daily[ref]), "ci_low": lo, "ci_high": hi,
                   "p_better_than_A": pb, **tt}
    return pd.DataFrame(rows).T


def thursday_split(panel: pd.DataFrame, cal: fv.Calendars) -> pd.DataFrame:
    """With Thursdays back in (everything else on): session P/L on Thursdays and on the other weekdays, split by whether the
    session is an event day (the blackout) or the day after an FOMC, with a 95% session-bootstrap interval of the mean."""
    ev = event_days(pd.DatetimeIndex(cal.all))
    book = panel[fv.version_mask(panel, **{**fv.STRATEGY, "skip_thursday": False})]
    s = book.groupby("date")[fv.PL].sum().to_frame("pl")
    s["thursday"] = s.index.dayofweek == 3
    s["event"] = ev.reindex(s.index)["any_event"].to_numpy()
    s["after_fomc"] = ev.reindex(s.index)["day_after_fomc"].to_numpy()
    groups = {"Thursday, event day": s["thursday"] & s["event"],
              "Thursday, day after FOMC": s["thursday"] & s["after_fomc"],
              "Thursday, no event": s["thursday"] & ~s["event"] & ~s["after_fomc"],
              "other days, event day": ~s["thursday"] & s["event"],
              "other days, no event": ~s["thursday"] & ~s["event"]}
    rows = {}
    rng = np.random.default_rng(fv.SEED)
    for k, m in groups.items():
        v = s.loc[m, "pl"].to_numpy()
        boots = rng.choice(v, (2000, len(v))).mean(axis=1) if len(v) else np.array([np.nan])
        rows[k] = {"sessions": len(v), "total_pl": v.sum(), "avg_session_pl": v.mean() if len(v) else np.nan,
                   "ci_low": np.quantile(boots, 0.025), "ci_high": np.quantile(boots, 0.975),
                   "winning_sessions_pct": (v > 0).mean() * 100 if len(v) else np.nan}
    return pd.DataFrame(rows).T


def fig_blackout(ab: pd.DataFrame, split: pd.DataFrame):
    """Left: Sharpe of each version (A = the strategy). Right: average session P/L by group, with 95% intervals."""
    with plt.rc_context(fv.STYLE):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1.1]})
        labels = ["A: no Thu\n(strategy)", "B: Thu in", "C: Thu in,\nblackout", "D: no Thu,\nblackout", "E: Thu in,\nno day after FOMC"]
        v = ab["sharpe"].astype(float).to_numpy()
        x = np.arange(len(v))
        a1.bar(x, v, width=0.6, color=[fv.BLUE] + [fv.GREY] * (len(v) - 1))
        for xi, val in zip(x, v):
            a1.text(xi, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, color=fv.INK_2)
        a1.set_xticks(x, labels, fontsize=8)
        a1.set_title("Sharpe, everything else on")
        t = split
        y = np.arange(len(t))[::-1]
        a2.barh(y, t["avg_session_pl"], height=0.55, color=[fv.ORANGE if "Thursday" in k else fv.GREY for k in t.index])
        a2.errorbar(t["avg_session_pl"], y, xerr=[t["avg_session_pl"] - t["ci_low"], t["ci_high"] - t["avg_session_pl"]],
                    fmt="none", ecolor=fv.INK_2, lw=0.9, capsize=3)
        a2.set_yticks(y, [f"{k} ({int(n)})" for k, n in zip(t.index, t["sessions"])], fontsize=8.5)
        a2.axvline(0, color=fv.INK_2, lw=0.7)
        a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{'-' if val < 0 else ''}${abs(val):,.0f}"))
        a2.set_title("Average session P/L with Thursdays back in (sessions), 95% interval")
        fig.suptitle("Do event-day blackouts explain the Thursday loss?", x=0.01, ha="left")
        fig.tight_layout()
    return fig


def blackout_before_after(panel: pd.DataFrame, cal: fv.Calendars) -> dict[str, pd.DataFrame]:
    """The live strategy before and after the event-day blackout was added: the headline stats with the paired bootstrap and
    t-tests, Sharpe and P/L by year, what each event type removes (sessions and P/L, a session can carry several), and the daily
    P/L of both versions."""
    ev = event_days(pd.DatetimeIndex(cal.all))
    before = fv.version_mask(panel, **fv.STRATEGY)
    after = before & ~panel["date"].isin(ev.index[ev["any_event"]])
    versions = {"before: no blackout": before, "after: event days blacked out": after}
    daily = {k: rs.daily_series(panel.loc[m, fv.PL], panel.loc[m, "date"], cal.all) for k, m in versions.items()}
    rows = {}
    for k, m in versions.items():
        st = cs._stats(panel, m, fv.PL, cal)
        rows[k] = {**st, "return_on_dd": st["total_pl"] / -st["max_drawdown"]}
    stats = pd.DataFrame(rows).T
    lo, hi, pb = fv._paired_boot(daily["before: no blackout"].to_numpy(), daily["after: event days blacked out"].to_numpy())
    tt = cs.ab_t_tests(daily["before: no blackout"].to_numpy(), daily["after: event days blacked out"].to_numpy())
    stats.attrs.update(ci_low=lo, ci_high=hi, p_after_better=pb, **tt)
    years = pd.DatetimeIndex(cal.all).year
    yr = pd.DataFrame({f"{stat} {k.split(':')[0]}": [f(daily[k][years == y]) for y in sorted(set(years))]
                       for k in versions for stat, f in (("sharpe", br.sharpe), ("P/L", np.sum))},
                      index=pd.Index([str(y) for y in sorted(set(years))], name="year"))
    s = panel[before].groupby("date")[fv.PL].sum()
    e = ev.reindex(s.index)
    removed = pd.DataFrame({k: {"sessions": int(e[k].sum()), "pl_removed": float(s[e[k]].sum()),
                                "avg_session_pl": float(s[e[k]].mean()) if e[k].any() else np.nan,
                                "winning_sessions_pct": float((s[e[k]] > 0).mean() * 100) if e[k].any() else np.nan}
                            for k in SAME_DAY + ["any_event"]}).T
    removed.loc["not an event day"] = {"sessions": int((~e["any_event"]).sum()), "pl_removed": float(s[~e["any_event"]].sum()),
                                       "avg_session_pl": float(s[~e["any_event"]].mean()),
                                       "winning_sessions_pct": float((s[~e["any_event"]] > 0).mean() * 100)}
    return {"stats": stats, "by_year": yr, "removed": removed, "daily": pd.DataFrame(daily)}


def fig_before_after(ba: dict[str, pd.DataFrame]):
    """Left: cumulative P/L before and after the blackout. Right: Sharpe by year, before and after."""
    d, yr = ba["daily"], ba["by_year"]
    with plt.rc_context(fv.STYLE):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
        for (k, col), color in zip(d.items(), (fv.GREY, fv.BLUE)):
            eq = col.cumsum()
            a1.plot(eq.index, eq, color=color, lw=1.8, label=k)
            a1.text(eq.index[-1], eq.iloc[-1], f" {fv._k(eq.iloc[-1])}".replace("$", r"\$"), va="center", fontsize=8.5, color=fv.INK)
        a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fv._k(v).replace("$", r"\$")))
        a1.legend(loc="upper left")
        a1.set_title("Cumulative P/L of the strategy")
        x = np.arange(len(yr))
        for i, (k, color) in enumerate((("before", fv.GREY), ("after", fv.BLUE))):
            v = yr[f"sharpe {k}"].to_numpy()
            a2.bar(x + (i - 0.5) * 0.38, v, width=0.36, color=color, label=k)
            for xi, val in zip(x, v):
                a2.text(xi + (i - 0.5) * 0.38, val, f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7.5,
                        color=fv.INK_2)
        a2.set_xticks(x, yr.index)
        a2.axhline(0, color=fv.INK_2, lw=0.7)
        a2.legend(loc="upper left")
        a2.set_title("Sharpe by year (2026: to 2026-05-07)")
        fig.suptitle("The live strategy before and after the event-day blackout", x=0.01, ha="left")
        fig.tight_layout()
    return fig


EVENT_TYPES = {"fomc": "FOMC", "cpi": "CPI", "nfp": "NFP", "opex": "monthly opex", "quad_witching": "quad witching",
               "vix_expiry": "VIX expiry", "around_holiday": "sessions either side of an NYSE holiday",
               "nyse_early_closes": "early closes", ("fomc", "cpi"): "FOMC + CPI",
               "any_event": "all listed (the live blackout)"}


def _log_daily() -> pd.Series:
    """The backtester's log (the previous 7, with 14:53, to 2026-09-28) as daily P/L on every business day."""
    from . import log_2026 as lg
    c = lg.condors(lg.load_log())
    d = c.groupby(pd.to_datetime(c["date"]))["pl"].sum()
    return d.reindex(pd.bdate_range(d.index.min(), d.index.max()), fill_value=0.0)


def _max_dd(x: np.ndarray) -> float:
    cum = np.cumsum(x)
    return float((cum - np.maximum.accumulate(cum)).min())


def blackout_by_event(panel: pd.DataFrame, cal: fv.Calendars) -> pd.DataFrame:
    """One event type blacked out at a time, on top of the live strategy (no blackout otherwise): the sessions and P/L it removes,
    Sharpe and drawdown on the export with a paired bootstrap and t-tests against no blackout, and the same blackout on the
    backtester's log (P/L, Sharpe, max drawdown, the 2026 drawdown)."""
    ev = event_days(pd.DatetimeIndex(cal.all))
    base = fv.version_mask(panel, **fv.STRATEGY)
    d0 = rs.daily_series(panel.loc[base, fv.PL], panel.loc[base, "date"], cal.all).to_numpy()
    log = _log_daily()
    lev = event_days(log.index)
    y26 = log.index.year == 2026
    rows = {"no blackout": {"sessions_removed": 0, "pl_removed": 0.0, "sharpe": br.sharpe(d0), "sharpe_change": 0.0,
                            "ci_low": np.nan, "ci_high": np.nan, "p_better": np.nan, "p_sharpe": np.nan, "total_pl": d0.sum(),
                            "max_drawdown": _max_dd(d0), "log_pl": log.sum(), "log_sharpe": br.sharpe(log.to_numpy()),
                            "log_max_drawdown": _max_dd(log.to_numpy()), "log_2026_drawdown": _max_dd(log[y26].to_numpy())}}
    for k, name in EVENT_TYPES.items():
        cols = list(k) if isinstance(k, tuple) else [k]
        keep = base & ~panel["date"].isin(ev.index[ev[cols].any(axis=1)])
        d = rs.daily_series(panel.loc[keep, fv.PL], panel.loc[keep, "date"], cal.all).to_numpy()
        lo, hi, pb = fv._paired_boot(d0, d)
        tt = cs.ab_t_tests(d0, d)
        lg_ = log.where(~lev[cols].any(axis=1), 0.0)
        rows[name] = {"sessions_removed": int(panel.loc[base & ~keep, "date"].nunique()), "pl_removed": d0.sum() - d.sum(),
                      "sharpe": br.sharpe(d), "sharpe_change": br.sharpe(d) - br.sharpe(d0), "ci_low": lo, "ci_high": hi,
                      "p_better": pb, "p_sharpe": tt["p_sharpe"], "total_pl": d.sum(), "max_drawdown": _max_dd(d),
                      "log_pl": lg_.sum(), "log_sharpe": br.sharpe(lg_.to_numpy()), "log_max_drawdown": _max_dd(lg_.to_numpy()),
                      "log_2026_drawdown": _max_dd(lg_[y26].to_numpy())}
    return pd.DataFrame(rows).T


def fig_blackout_by_event(t: pd.DataFrame):
    """Per event type blacked out: the Sharpe change on the export (95% interval) and the log's max drawdown."""
    dd0 = float(t.loc["no blackout", "log_max_drawdown"])
    t = t.drop(index="no blackout")
    y = np.arange(len(t))[::-1]
    live = [i == len(t) - 1 for i in range(len(t))]
    with plt.rc_context(fv.STYLE):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 4.8), sharey=True)
        v = t["sharpe_change"].astype(float)
        a1.barh(y, v, height=0.55, color=[fv.BLUE if l else fv.GREY for l in live])
        a1.errorbar(v, y, xerr=[v - t["ci_low"].astype(float), t["ci_high"].astype(float) - v], fmt="none", ecolor=fv.INK_2, lw=0.9,
                    capsize=3)
        for yi, val in zip(y, v):
            a1.text(val, yi + 0.3, f"{val:+.2f}", ha="center", fontsize=8, color=fv.INK_2)
        a1.axvline(0, color=fv.INK_2, lw=0.7)
        a1.set_yticks(y, [f"{n} ({int(s)})" for n, s in zip(t.index, t["sessions_removed"])], fontsize=8.5)
        a1.set_title("Export 2022-26: Sharpe change, 95% interval (sessions removed)")
        w = t["log_max_drawdown"].astype(float)
        a2.barh(y, w, height=0.55, color=[fv.BLUE if l else fv.GREY for l in live])
        a2.axvline(dd0, color=fv.RED, ls="--", lw=1.2)
        a2.text(dd0, 1.0, " no blackout", transform=a2.get_xaxis_transform(), fontsize=8, color=fv.INK_2, va="bottom")
        for yi, val in zip(y, w):
            a2.text(val, yi, f" {'-' if val < 0 else ''}${abs(val) / 1000:.1f}k ", ha="right", va="center", fontsize=8, color=fv.INK_2)
        a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{'-' if val < 0 else ''}${abs(val) / 1000:.0f}k"))
        a2.set_title("Backtester's log to 2026-09-28: max drawdown", loc="right")
        fig.suptitle("Which event days to black out: each type on its own, on top of the live strategy", x=0.01, ha="left")
        fig.tight_layout()
    return fig
