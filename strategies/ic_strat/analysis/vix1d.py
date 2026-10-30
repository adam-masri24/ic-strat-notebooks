"""Why the VIX1D features failed as filters: missing data before 2023-04-24, not a bad signal.

On the finished strategy (``final_version.STRATEGY``):

- ``coverage``: how many condors and how much P/L fall before VIX1D exists;
- ``scores``: each VIX1D feature's best single cut (the ranking's search) with the days it doesn't exist dropped (the old,
  wrong treatment) and kept (the fix), against the strategy;
- ``quintiles``: P/L per condor by quintile of each feature on the days it exists, with a session-bootstrap interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import backtester_rules as br
from . import final_version as fv
from . import regime_study as rs

FEATURES = {"vix1d_open": "VIX1D at the open", "ts_1d_30_open": "VIX1D / VIX30 at the open"}
MIN_KEEP = 0.30


def _book(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[fv.version_mask(panel, **fv.STRATEGY)]


def coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Condors, P/L and sessions before and after VIX1D starts."""
    b = _book(panel)
    start = panel.loc[panel["vix1d_open"].notna(), "date"].min()
    rows = {}
    for name, m in (("before VIX1D exists", b["date"] < start), ("VIX1D exists", b["date"] >= start)):
        rows[name] = {"from": b.loc[m, "date"].min(), "to": b.loc[m, "date"].max(), "sessions": b.loc[m, "date"].nunique(),
                      "condors": int(m.sum()), "share_of_condors_pct": m.mean() * 100, "total_pl": b.loc[m, fv.PL].sum()}
    out = pd.DataFrame(rows).T
    out.attrs["start"] = start
    return out


def _best_cut(b: pd.DataFrame, feat: str, keep_missing: bool, cal: fv.Calendars) -> dict:
    x, na = b[feat], b[feat].isna()
    best = None
    for q in np.arange(0.1, 1.0, 0.1):
        cut = x.quantile(q)
        for label, k in ((f">= {cut:.3g}", x >= cut), (f"<= {cut:.3g}", x <= cut)):
            k = k | na if keep_missing else k & ~na
            if k.mean() < MIN_KEEP or k.all():
                continue
            st = fv.score_pl(b.loc[k, fv.PL], b.loc[k, "date"], cal)
            if best is None or st["sharpe"] > best["sharpe"]:
                best = {"rule": label, "kept_pct": k.mean() * 100, **st}
    return best


def scores(panel: pd.DataFrame, cal: fv.Calendars) -> pd.DataFrame:
    """Each feature's best cut, with the days it doesn't exist dropped (old) and kept (fix): Sharpe gain and P/L change."""
    b = _book(panel)
    base = fv.score_pl(b[fv.PL], b["date"], cal)
    rows = []
    for feat, name in FEATURES.items():
        for keep, how in ((False, "missing days dropped (old)"), (True, "missing days kept (fix)")):
            r = _best_cut(b, feat, keep, cal)
            rows.append({"feature": name, "treatment": how, "rule": r["rule"], "kept_pct": r["kept_pct"],
                         "sharpe_gain": r["sharpe"] - base["sharpe"], "total_pl_change": r["total_pl"] - base["total_pl"]})
    return pd.DataFrame(rows).set_index(["feature", "treatment"])


def quintiles(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """P/L per condor by quintile of each feature, on the days it exists, with a 95% session-bootstrap interval."""
    b = _book(panel)
    out = {}
    for feat, name in FEATURES.items():
        s = b[b[feat].notna()]
        rows = []
        for iv, g in s.groupby(pd.qcut(s[feat], 5), observed=True):
            lo, hi = rs.session_bootstrap_mean(g[fv.PL], g["date"])
            rows.append({"bucket": f"{iv.left:.2f}-{iv.right:.2f}", "condors": len(g), "avg_pl": g[fv.PL].mean(),
                         "ci_low": lo, "ci_high": hi, "win_rate_pct": (g[fv.PL] > 0).mean() * 100})
        t = pd.DataFrame(rows).set_index("bucket")
        t.attrs["rank_corr"] = s[feat].corr(s[fv.PL], method="spearman")
        out[name] = t
    return out


def fig_coverage(panel: pd.DataFrame, cov: pd.DataFrame):
    """The strategy's cumulative P/L with the pre-VIX1D period shaded, and VIX1D at the open on the strategy's days."""
    b = _book(panel)
    start = cov.attrs["start"]
    daily = b.groupby("date")[fv.PL].sum()
    days = b.groupby("date")["vix1d_open"].first()
    with plt.rc_context(fv.STYLE):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
        eq = daily.cumsum()
        a1.plot(eq.index, eq, color=fv.BLUE, lw=1.6)
        a1.axvspan(eq.index.min(), start, color=fv.RED, alpha=0.10, lw=0)
        pre = cov.loc["before VIX1D exists"]
        a1.text(eq.index.min(), 0.95, f" no VIX1D data: {int(pre['condors'])} condors ({pre['share_of_condors_pct']:.0f}%), "
                f"{fv._k(pre['total_pl'])}".replace("$", r"\$"), transform=a1.get_xaxis_transform(), va="top", fontsize=8.5,
                color=fv.INK)
        a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fv._k(v)))
        a1.set_title("The strategy's cumulative P/L: a VIX1D rule that drops blank days throws out everything in the red zone")
        a2.scatter(days.index, days, s=10, color=fv.INK_2)
        a2.axvline(start, color=fv.RED, lw=1.2)
        a2.text(start, 0.95, f" VIX1D starts {start:%Y-%m-%d}", transform=a2.get_xaxis_transform(), va="top", fontsize=8.5,
                color=fv.INK)
        a2.set_title("VIX1D at the open, on the strategy's trading days")
        fig.suptitle("Why the VIX1D features failed: a fifth of the strategy happens before VIX1D exists", x=0.01, ha="left")
        fig.tight_layout()
    return fig


def fig_scores(sc: pd.DataFrame, noise: float, bar: float):
    """Best-cut Sharpe gain of each VIX1D feature, missing days dropped (old) vs kept (fix), against the noise control and bar."""
    with plt.rc_context(fv.STYLE):
        fig, ax = plt.subplots(figsize=(11, 4.2))
        feats = sc.index.get_level_values(0).unique()
        y = np.arange(len(feats))
        for k, (how, color) in enumerate((("missing days dropped (old)", fv.GREY), ("missing days kept (fix)", fv.BLUE))):
            v = [sc.loc[(f, how), "sharpe_gain"] for f in feats]
            ax.barh(y + (k - 0.5) * 0.36, v, height=0.34, color=color, label=how)
            for yi, val in zip(y, v):
                ax.text(val, yi + (k - 0.5) * 0.36, f" {val:+.2f} " , va="center", ha="left" if val >= 0 else "right", fontsize=8.5,
                        color=fv.INK_2)
        ax.axvline(0, color=fv.INK_2, lw=0.7)
        ax.axvline(noise, color=fv.GREY, ls=":", lw=1.4)
        ax.text(noise, 0.01, f" noise feature {noise:+.2f}", transform=ax.get_xaxis_transform(), fontsize=8, color=fv.INK_2, va="bottom")
        ax.axvline(bar, color=fv.RED, ls="--", lw=1.3)
        ax.text(bar, 0.99, f"noise bar {bar:+.2f} ", transform=ax.get_xaxis_transform(), fontsize=8, color=fv.INK_2, va="top", ha="right")
        ax.set_yticks(y, feats)
        ax.set_xlabel("Sharpe gain of the best single cut, vs the strategy")
        ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.5))
        ax.set_title("The same search, with the days VIX1D doesn't exist dropped (old) or kept (fix)")
        fig.tight_layout()
    return fig


def fig_quintiles(q: dict[str, pd.DataFrame]):
    """P/L per condor by quintile of each VIX1D feature (days it exists), with 95% intervals."""
    with plt.rc_context(fv.STYLE):
        fig, axes = plt.subplots(1, len(q), figsize=(15, 4.3), sharey=True)
        for ax, (name, t) in zip(axes, q.items()):
            x = np.arange(len(t))
            ax.bar(x, t["avg_pl"], width=0.6, color=fv.BLUE)
            ax.errorbar(x, t["avg_pl"], yerr=[t["avg_pl"] - t["ci_low"], t["ci_high"] - t["avg_pl"]], fmt="none", ecolor=fv.INK_2,
                        lw=0.9, capsize=3)
            ax.axhline(0, color=fv.INK_2, lw=0.7)
            ax.set_xticks(x, t.index, fontsize=8)
            ax.set_title(f"{name}: rank correlation with P/L {t.attrs['rank_corr']:+.2f}")
            ax.set_xlabel("quintile")
        axes[0].set_ylabel("avg P/L per condor ($), 95% interval")
        fig.suptitle("On the days VIX1D exists: P/L per condor by quintile (no clear direction, overlapping intervals)", x=0.01,
                     ha="left")
        fig.tight_layout()
    return fig


@dataclass
class VIX1DStudy:
    coverage: pd.DataFrame
    scores: pd.DataFrame
    quintiles: dict[str, pd.DataFrame]


def compute(panel: pd.DataFrame) -> VIX1DStudy:
    cal = fv.Calendars.of(panel)
    return VIX1DStudy(coverage=coverage(panel), scores=scores(panel, cal), quintiles=quintiles(panel))
