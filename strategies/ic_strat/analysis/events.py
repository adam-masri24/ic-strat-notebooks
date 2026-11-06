"""Event-day blackouts against the no-Thursday rule: is the Thursday loss just FOMC, CPI, NFP, expiries and holidays?

Every other setting of the strategy is on. The calendar (``event_calendar.json``) lists FOMC, CPI and NFP releases, monthly opex,
quad witching, VIX expiry, NYSE holidays and early closes.

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
# Holidays have no session; the day before one (and an early close) is the tradable effect.
SAME_DAY = ["fomc", "cpi", "nfp", "opex", "quad_witching", "vix_expiry", "nyse_early_closes"]


def load_calendar() -> dict[str, pd.DatetimeIndex]:
    return {k: pd.DatetimeIndex(pd.to_datetime(v)) for k, v in json.loads(CALENDAR_FILE.read_text()).items()}


def event_days(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Session x event flags: each event type on the day, the day after an FOMC (always a Thursday), the session before a
    holiday, and ``any_event`` (the blackout)."""
    cal = load_calendar()
    s = pd.DatetimeIndex(sorted(sessions))
    out = pd.DataFrame(index=s)
    for k in SAME_DAY:
        out[k] = s.isin(cal[k])
    prev = pd.Series(s, index=s).shift(1)
    out["day_after_fomc"] = prev.isin(cal["fomc"]).to_numpy()
    nxt = pd.Series(s, index=s).shift(-1)
    out["before_holiday"] = [any(h > d and (pd.isna(n) or h < n) for h in cal["nyse_holidays"]) for d, n in zip(s, nxt)]
    out["any_event"] = out[SAME_DAY + ["before_holiday"]].any(axis=1)
    out["weekday"] = s.dayofweek
    return out


def by_weekday(ev: pd.DataFrame) -> pd.DataFrame:
    """How many sessions of each event type fall on each weekday."""
    cols = SAME_DAY + ["before_holiday", "day_after_fomc", "any_event"]
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
