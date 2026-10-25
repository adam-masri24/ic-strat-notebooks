"""Inside the term-structure band only: which features move P/L, and at what threshold?

Right after the term structure is chosen, before any second filter: keep only
``BAND_LOW <= ts_9d_30_open <= BAND_HIGH`` (0.87-1.04, no SMA condition) and ask
every candidate feature **used as a filter on the band, how much does it move
P/L and Sharpe, and at what threshold?**

Candidates, all known before the tranche enters:

* SMA and EMA crossovers on SPX price (lagged one session), as flags and as
  spreads, plus SPX against its 20-day SMA;
* VVIX (prior close, from Cboe's daily history), its day-over-day change, its
  ratio to VIX, and its 252-day rank;
* IV rank of the VIX open over 30, 90 and 252 sessions;
* VIX1D/VIX9D/VIX levels, the 1D/9D ratio, VIX overnight move, SPX gap, SPX
  intraday movement, prior-day return, RSI(14), the fill (premium, short/long
  ratios), entry time and weekday;
* **one noise feature**, a random number per session. It carries nothing, so
  whatever it scores is the test's zero.

All on P/L with the strategy's 175% call stop at $20 slippage, and all on
every band session (no train/test split), so every gain is in-sample and the
noise feature is the bar:

1. **Filter scan** — each feature's best single cut (Sharpe and P/L).
2. **Tree model** — gradient-boosted trees on P/L, permutation importance.
3. **Quintiles** — avg P/L and Sharpe in each fifth of every feature.
4. **Years** — the highest-ranked features' rules, year by year.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from src.paths import VVIX_DAILY_FILE

from . import regime_study as rs
from .ic_analysis import require_file
from .pipeline import PRICE_CROSSOVER_COL
from .term_structure_features import TS_9D_30, load_term_structure_daily

STUDY_NAME = "SPX-IC-Band-TS087-104-Feature-Study"
# The band this study runs on: the strategy's lower edge, and the wider 1.04 cap.
BAND_LOW, BAND_HIGH = rs.TS_LOW, rs.ALT_TS_HIGH
VVIX_CSV = VVIX_DAILY_FILE
SEED = 0
STOP_PCT, SLIPPAGE = rs.RECOMMENDED_CALL_STOP, 20.0

# column -> (family, plain description). Flags are 0/1.
FEATURES: dict[str, tuple[str, str]] = {
    "sma5_gt_sma9": ("trend", "SPX SMA5 > SMA9, prior close (the strategy's filter)"),
    "sma_spread_pct": ("trend", "SPX SMA5 vs SMA9 spread, % (prior close)"),
    "ema5_gt_ema9": ("trend", "SPX EMA5 > EMA9, prior close"),
    "ema_spread_pct": ("trend", "SPX EMA5 vs EMA9 spread, % (prior close)"),
    "ema20m_gt_ema40m": ("trend", "SPX EMA20 > EMA40, minutes, at entry (approx.)"),
    "ema20m_40m_spread_pct": ("trend", "SPX EMA20 vs EMA40 spread, minutes, % (approx.)"),
    "spx_vs_sma20_pct": ("trend", "SPX prior close vs its 20-day SMA, %"),
    "ret_prev_pct": ("trend", "SPX prior-day return, %"),
    "rsi14_lagged": ("trend", "RSI(14) on SPX, prior close"),
    "vvix_prev": ("vvix", "VVIX, prior close"),
    "vvix_chg_pct": ("vvix", "VVIX prior-day change, %"),
    "vvix_vix_ratio": ("vvix", "VVIX / VIX, prior close"),
    "vvix_rank252": ("vvix", "VVIX 252-day rank, prior close"),
    "ivr30": ("iv_rank", "IV rank of the VIX open, 30 sessions"),
    "ivr90": ("iv_rank", "IV rank of the VIX open, 90 sessions"),
    "ivr252": ("iv_rank", "IV rank of the VIX open, 252 sessions"),
    "vix30_open": ("vix_level", "VIX at the open"),
    "vix9d_open": ("vix_level", "VIX9D at the open"),
    "vix1d_open": ("vix_level", "VIX1D at the open"),
    "ts_1d_9d_open": ("term_structure", "VIX1D / VIX9D at the open"),
    "ts_9d_30_open": ("term_structure", "VIX9D / VIX30 at the open (inside the band)"),
    "vix_overnight_pct": ("overnight", "VIX overnight move, %"),
    "gap_pct": ("overnight", "SPX opening gap, %"),
    "spx_move_pct": ("intraday", "SPX intraday movement since the 09:30 open, %"),
    "credit": ("fill", "premium collected, whole condor"),
    "slratio_call": ("fill", "short/long price ratio, call side"),
    "slratio_put": ("fill", "short/long price ratio, put side"),
    "mins_from_open": ("calendar", "entry time, minutes after 09:30"),
    "dow_code": ("calendar", "day of week (0 = Monday)"),
    "noise": ("noise", "random number per session (control)"),
}
FLAGS = ["sma5_gt_sma9", "ema5_gt_ema9", "ema20m_gt_ema40m"]
# Intraday EMA spans, in minutes.
FAST_MIN, SLOW_MIN = 20, 40
HEAD_TO_HEAD_PLOTTED = ["band only", "band + SMA5 > SMA9 (the strategy)", "band + EMA5 > EMA9",
                        "band + SMA + movement <= 0%", "band + EMA + movement <= 0%"]
FEATURE_COLS = list(FEATURES)


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #
def _rank(series: pd.Series, window: int) -> pd.Series:
    lo, hi = series.rolling(window, min_periods=window).min(), series.rolling(window, min_periods=window).max()
    return 100 * (series - lo) / (hi - lo)


def load_vvix(path: str | Path = VVIX_CSV) -> pd.DataFrame:
    """Cboe's daily VVIX history (closes only), with the prior-close features used at entry."""
    v = pd.read_csv(require_file(Path(path), "VVIX daily file"))
    v.columns = [c.strip().lower() for c in v.columns]
    v["date"] = pd.to_datetime(v["date"], format="%m/%d/%Y")
    v = v.sort_values("date").rename(columns={"vvix": "vvix_close"})
    # Everything is shifted one session: the open never sees that day's close.
    v["vvix_prev"] = v["vvix_close"].shift(1)
    v["vvix_chg_pct"] = (v["vvix_close"] / v["vvix_close"].shift(1) - 1).shift(1) * 100
    v["vvix_rank252"] = _rank(v["vvix_close"], 252).shift(1)
    return v[["date", "vvix_prev", "vvix_chg_pct", "vvix_rank252"]]


def _time_ema(prices: np.ndarray, minutes: np.ndarray, span: float) -> np.ndarray:
    """EMA of irregularly spaced prices, with a span in minutes.

    A per-minute EMA with span N has decay 2/(N+1) per minute; over a gap of dt
    minutes the weight on the new price is 1 - (1 - 2/(N+1)) ** dt.
    """
    keep = 1 - 2 / (span + 1)
    out = np.empty_like(prices, dtype=float)
    out[0] = prices[0]
    for i in range(1, len(prices)):
        w = 1 - keep ** (minutes[i] - minutes[i - 1])
        out[i] = out[i - 1] + w * (prices[i] - out[i - 1])
    return out


def add_minute_ema(panel: pd.DataFrame) -> pd.DataFrame:
    """Intraday EMA(FAST_MIN) vs EMA(SLOW_MIN) of SPX, in minutes, at each tranche's entry.

    **An approximation.** The backtester computes it on 1-minute bars; this file
    only has SPX at the 20-minute tranche entries plus the 09:30 open (spot minus
    the export's ``Movement``). Each day starts both EMAs at the open and updates
    them at every tranche with the time gap since the last one, so the value at
    a tranche uses only prices up to its entry.
    """
    out = panel.copy()
    fast, slow = pd.Series(np.nan, index=out.index), pd.Series(np.nan, index=out.index)
    for _, grp in out.groupby("date"):
        grp = grp.sort_values("mins_from_open")
        prices = np.r_[grp["spot"].iloc[0] - grp["movement"].iloc[0], grp["spot"].to_numpy()]
        minutes = np.r_[0.0, grp["mins_from_open"].to_numpy(float)]
        fast.loc[grp.index] = _time_ema(prices, minutes, FAST_MIN)[1:]
        slow.loc[grp.index] = _time_ema(prices, minutes, SLOW_MIN)[1:]
    out["ema20m_gt_ema40m"] = (fast > slow).astype(float)
    out["ema20m_40m_spread_pct"] = (fast / slow - 1) * 100
    return out


def build_band_panel(trades_csv: str | Path, term_structure_csv: str | Path | None = None,
                     vvix_csv: str | Path = VVIX_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every condor inside the band (no SMA filter), with every candidate feature and the stopped P/L."""
    panel, spreads = build_feature_panel(trades_csv, term_structure_csv, vvix_csv)
    band = panel[panel[TS_9D_30].between(BAND_LOW, BAND_HIGH)].copy()
    return band, spreads


def build_feature_panel(trades_csv: str | Path, term_structure_csv: str | Path | None = None,
                        vvix_csv: str | Path = VVIX_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every condor on every session, with every candidate feature and the stopped P/L (``pl_stop``)."""
    panel, spreads = rs.build_regime_panel(trades_csv, term_structure_csv)

    # SPX trend family, from the session closes, lagged one session like the SMA flag.
    closes = panel.groupby("date")["close"].first().sort_index()
    trend = pd.DataFrame(index=closes.index)
    sma5, sma9, sma20 = (closes.rolling(w, min_periods=w).mean() for w in (5, 9, 20))
    ema5, ema9 = (closes.ewm(span=w, min_periods=w).mean() for w in (5, 9))
    trend["sma_spread_pct"] = (sma5 / sma9 - 1) * 100
    trend["ema_spread_pct"] = (ema5 / ema9 - 1) * 100
    trend["ema5_gt_ema9"] = (ema5 > ema9).astype(float)
    trend["spx_vs_sma20_pct"] = (closes / sma20 - 1) * 100
    trend["ret_prev_pct"] = closes.pct_change() * 100
    trend = trend.shift(1)
    panel = panel.merge(trend, left_on="date", right_index=True, how="left")
    panel["sma5_gt_sma9"] = panel[PRICE_CROSSOVER_COL].astype(float)
    panel = add_minute_ema(panel)

    # IV rank of the VIX open, and VVIX against the VIX close it trades beside.
    ts_kwargs = {} if term_structure_csv is None else {"path": term_structure_csv}
    # (build_regime_panel already carries vix30_close_prev for the overnight move.)
    daily = load_term_structure_daily(**ts_kwargs)[["date", "vix30_open"]].set_index("date")
    ivr = pd.DataFrame({f"ivr{w}": _rank(daily["vix30_open"], w) for w in (30, 90, 252)})
    panel = panel.merge(ivr, left_on="date", right_index=True, how="left")
    panel = panel.merge(load_vvix(vvix_csv), on="date", how="left")
    panel["vvix_vix_ratio"] = panel["vvix_prev"] / panel["vix30_close_prev"]

    rng = np.random.default_rng(SEED)
    days = panel["date"].unique()
    panel["noise"] = panel["date"].map(dict(zip(days, rng.standard_normal(len(days)))))

    panel["pl_stop"], _ = rs.condor_pl_with_call_stop(panel, spreads, STOP_PCT, SLIPPAGE)
    return panel, spreads


# --------------------------------------------------------------------------- #
# 1. each feature as a filter
# --------------------------------------------------------------------------- #
def _sharpe(pl: pd.Series, dates: pd.Series, calendar: np.ndarray) -> float:
    return rs.book_stats(pl, dates, calendar)["sharpe"]


def filter_scan(band: pd.DataFrame, min_keep: float = 0.30) -> pd.DataFrame:
    """Each feature's best single keep-rule over every band session.

    A flag's candidates are "keep 1" and "keep 0"; a continuous feature's are a
    min or max at each decile. Sharpe is on daily summed P/L over the band's
    sessions, skipped days as zero, so every rule is scored on one calendar.
    Chosen and scored on the same sessions, so every gain is in-sample: the
    noise feature, and :func:`noise_distribution`, are the bar to clear.
    The kept mask of each feature's rule is in ``attrs["masks"]``.
    """
    pl = band["pl_stop"]
    calendar = np.sort(band["date"].unique())
    base = rs.book_stats(pl, band["date"], calendar)
    rows, masks = [], {}
    for c in FEATURE_COLS:
        if c in FLAGS:
            candidates = [(f"{c} = 1", band[c] == 1), (f"{c} = 0", band[c] == 0)]
        else:
            candidates = []
            for q in np.arange(0.1, 1.0, 0.1):
                cut = band[c].quantile(q)
                candidates += [(f"{c} >= {cut:.4g}", band[c] >= cut), (f"{c} <= {cut:.4g}", band[c] <= cut)]
        best = None
        for label, keep in candidates:
            if keep.mean() < min_keep:
                continue
            st = rs.book_stats(pl[keep], band.loc[keep, "date"], calendar)
            if best is None or st["sharpe"] > best[0]["sharpe"]:
                best = (st, label, keep)
        st, label, keep = best
        masks[c] = keep
        rows.append({
            "feature": c, "family": FEATURES[c][0], "description": FEATURES[c][1], "rule": label,
            "sharpe": st["sharpe"], "sharpe_gain": st["sharpe"] - base["sharpe"], "kept_pct": keep.mean() * 100,
            "kept_avg_pl": pl[keep].mean(), "dropped_avg_pl": pl[~keep].mean(),
            "total_pl": st["total_pl"], "max_drawdown": st["max_drawdown"],
        })
    out = pd.DataFrame(rows).set_index("feature").sort_values("sharpe_gain", ascending=False)
    out.attrs.update(noise_gain=float(out.loc["noise", "sharpe_gain"]), base_sharpe=base["sharpe"],
                     base_total_pl=base["total_pl"], base_max_drawdown=base["max_drawdown"], masks=masks)
    out["beats_noise"] = out["sharpe_gain"] > out.attrs["noise_gain"]
    return out


def noise_distribution(band: pd.DataFrame, n_draws: int = 200, min_keep: float = 0.30,
                       seed: int = 1) -> np.ndarray:
    """Sharpe gains of ``n_draws`` random session-level features put through the same search.

    One noise feature is one draw: it can land lucky or unlucky. The 95th
    percentile across many draws is the gain a threshold search can find in
    pure noise on these sessions, so it is the strict bar for a searched feature.
    """
    rng = np.random.default_rng(seed)
    days = band["date"].unique()
    calendar = np.sort(days)
    pl = band["pl_stop"]
    base = rs.book_stats(pl, band["date"], calendar)["sharpe"]
    gains = []
    for _ in range(n_draws):
        z = band["date"].map(dict(zip(days, rng.standard_normal(len(days))))).to_numpy()
        best = -np.inf
        for q in np.arange(0.1, 1.0, 0.1):
            cut = np.quantile(z, q)
            for keep in (z >= cut, z <= cut):
                if keep.mean() < min_keep:
                    continue
                best = max(best, rs.book_stats(pl[keep], band["date"][keep], calendar)["sharpe"])
        gains.append(best - base)
    return np.array(gains)


def second_filter_rules(band: pd.DataFrame) -> dict[str, pd.Series]:
    """The head-to-head: the band alone, with the SMA, and with the alternatives, at round thresholds."""
    sma, ema = band["sma5_gt_sma9"] == 1, band["ema5_gt_ema9"] == 1
    move = band["spx_move_pct"] <= 0
    return {
        "band only": pd.Series(True, index=band.index),
        "band + SMA5 > SMA9 (the strategy)": sma,
        "band + EMA5 > EMA9": ema,
        "band + SMA spread >= -0.25%": band["sma_spread_pct"] >= -0.25,
        "band + SMA5 > SMA9 or EMA5 > EMA9": sma | ema,
        "band + IV rank 90 <= 25": band["ivr90"] <= 25,
        "band + SPX >= 1% above its SMA20": band["spx_vs_sma20_pct"] >= 1,
        "band + SPX movement <= 0%": move,
        "band + SMA + movement <= 0%": sma & move,
        "band + EMA + movement <= 0%": ema & move,
    }


def disagreement(band: pd.DataFrame) -> pd.DataFrame:
    """The sessions where the SMA and EMA flags disagree, and what each side earned."""
    sma, ema = band["sma5_gt_sma9"] == 1, band["ema5_gt_ema9"] == 1
    rows = []
    for name, m in (("EMA says trade, SMA says skip", ema & ~sma), ("SMA says trade, EMA says skip", sma & ~ema)):
        d = band[m].groupby("date")["pl_stop"].sum()
        rows.append({"case": name, "sessions": len(d), "pl_per_session": d.mean(),
                     "winning_sessions_pct": (d > 0).mean() * 100, "total_pl": d.sum()})
    out = pd.DataFrame(rows).set_index("case")
    out.attrs["agreement_pct"] = float((sma == ema).mean() * 100)
    return out


# --------------------------------------------------------------------------- #
# 2. tree model
# --------------------------------------------------------------------------- #
@dataclass
class TreeResult:
    importance: pd.DataFrame
    fold_spearman: list[float]  # one entry: the in-sample skill
    null_spearman: list[float]


def _tree() -> HistGradientBoostingRegressor:
    # Shallow and regularised: ~1,000 effective samples.
    return HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                         min_samples_leaf=200, l2_regularization=1.0, random_state=SEED)


def tree_importance(band: pd.DataFrame, n_repeats: int = 5, n_null: int = 20) -> TreeResult:
    """Permutation importance of a gradient-boosted model of P/L, fitted and scored on every session.

    The target is the stopped P/L, winsorised at the 1st/99th percentile so a
    few blow-up days don't decide every split. Skill is the Spearman rank
    correlation between predicted and actual P/L; a feature's importance is how
    much it drops when the feature is shuffled between sessions. With no
    held-out data the skill is optimistic, so it is read against the same model
    fitted on P/L shuffled between whole sessions.
    """
    X = band[FEATURE_COLS].to_numpy(float)
    lo, hi = band["pl_stop"].quantile([0.01, 0.99])
    y = band["pl_stop"].clip(lo, hi).to_numpy()
    g = band["date"].to_numpy()
    rng = np.random.default_rng(SEED)
    model = _tree().fit(X, y)
    base = spearmanr(model.predict(X), y).statistic
    drops = {c: [] for c in FEATURE_COLS}
    for _ in range(n_repeats):
        for j, c in enumerate(FEATURE_COLS):
            Xp = X.copy()
            Xp[:, j] = X[rs._permute_by_session(np.arange(len(y)), g, rng), j]
            drops[c].append(base - spearmanr(model.predict(Xp), y).statistic)
    null = []
    for _ in range(n_null):
        yp = rs._permute_by_session(y, g, rng)
        null.append(float(spearmanr(_tree().fit(X, yp).predict(X), yp).statistic))

    imp = pd.DataFrame({c: {"mean": np.mean(v), "sd": np.std(v)} for c, v in drops.items()}).T
    imp["family"] = [FEATURES[c][0] for c in imp.index]
    imp["description"] = [FEATURES[c][1] for c in imp.index]
    imp = imp.sort_values("mean", ascending=False)
    imp.attrs["noise_bar"] = float(imp.loc["noise", "mean"] + 2 * imp.loc["noise", "sd"])
    imp["beats_noise"] = imp["mean"] > imp.attrs["noise_bar"]
    return TreeResult(imp, [float(base)], null)


# --------------------------------------------------------------------------- #
# 3. quintiles
# --------------------------------------------------------------------------- #
def quintiles(band: pd.DataFrame) -> pd.DataFrame:
    """Avg P/L, win rate and Sharpe in each fifth of every feature (two states for a flag)."""
    calendar = np.sort(band["date"].unique())
    rows = []
    for c in FEATURE_COLS:
        groups = band[c] if c in FLAGS else pd.qcut(band[c], 5, duplicates="drop")
        for i, (level, grp) in enumerate(band.groupby(groups, observed=True)):
            pl = grp["pl_stop"]
            rows.append({"feature": c, "bucket": i + 1, "range": str(level), "trades": len(grp),
                         "sessions": grp["date"].nunique(), "win_rate": (pl > 0).mean() * 100,
                         "avg_pl": pl.mean(), "sharpe": _sharpe(pl, grp["date"], calendar)})
    return pd.DataFrame(rows)


def threshold_table(q: pd.DataFrame) -> pd.DataFrame:
    """One row per feature: each quintile's range and avg P/L, low to high, for reading thresholds off."""
    rows = []
    for c in FEATURE_COLS:
        d = q[q["feature"] == c].sort_values("bucket")
        row = {"feature": c, "description": FEATURES[c][1]}
        for _, r in d.iterrows():
            b = int(r["bucket"])
            row[f"Q{b}_range"] = r["range"]
            row[f"Q{b}_avg_pl"] = round(r["avg_pl"], 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
BLUE, RED, GREY, GOLD = rs.BLUE, rs.RED, rs.GREY, "#e8a33d"


def _label(c: str) -> str:
    return FEATURES[c][1]


def _color(c: str, positive: bool = True) -> str:
    if c == "noise":
        return GREY
    if c == "sma5_gt_sma9":
        return GOLD
    return BLUE if positive else RED


def fig_filter_scan(scan: pd.DataFrame):
    r = scan.sort_values("sharpe_gain")
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.barh([_label(c) for c in r.index], r["sharpe_gain"],
            color=[_color(c, v > 0) for c, v in zip(r.index, r["sharpe_gain"])])
    ax.axvline(0, color="k", lw=0.6)
    ax.axvline(scan.attrs["noise_gain"], color=GREY, ls="--", lw=1.2, label="the one noise feature")
    if "noise_p95" in scan.attrs:
        ax.axvline(scan.attrs["noise_p95"], color=RED, ls="--", lw=1.2,
                   label="95th pct of 200 noise features (strict bar)")
    ax.set_xlabel("Sharpe change vs. the band alone, every session")
    ax.set_title("Each feature as a filter on the band (gold = SMA5 > SMA9, grey = noise)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_tree_importance(tree: TreeResult):
    imp = tree.importance.sort_values("mean")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.barh([_label(c) for c in imp.index], imp["mean"], xerr=imp["sd"], ecolor="#555", capsize=2,
            color=[_color(c, ok) if c in ("noise", "sma5_gt_sma9") else (BLUE if ok else "#9dbfe9")
                   for c, ok in zip(imp.index, imp["beats_noise"])])
    ax.axvline(imp.attrs["noise_bar"], color=RED, ls="--", lw=1.2, label="noise bar (noise mean + 2 SD)")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("drop in Spearman(predicted, actual P/L) when shuffled, every session")
    ax.set_title("Tree-model importance on the band (gold = SMA5 > SMA9, grey = noise)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def fig_quintile_grid(q: pd.DataFrame, features: list[str], metric: str = "avg_pl"):
    n = len(features)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.6 * rows), squeeze=False)
    for ax, c in zip(axes.flat, features):
        d = q[q["feature"] == c]
        ax.bar(d["bucket"], d[metric], color=[BLUE if v >= 0 else RED for v in d[metric]], width=0.7)
        ax.axhline(0, color="k", lw=0.6)
        ticks = ["0", "1"] if c in FLAGS else [f"Q{b}" for b in d["bucket"]]
        ax.set_xticks(d["bucket"], ticks)
        ax.set_title(_label(c), fontsize=8.5)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    ylabel = {"avg_pl": "avg P/L per condor ($)", "sharpe": "Sharpe (band calendar)"}[metric]
    fig.suptitle(f"{ylabel} by quintile, low -> high (flags: 0 / 1)", fontsize=12)
    fig.tight_layout()
    return fig


def fig_head_to_head(band: pd.DataFrame, rules: dict[str, pd.Series]):
    calendar = np.sort(band["date"].unique())
    fig, ax = plt.subplots(figsize=(11, 4.8))
    palette = [GREY, GOLD, BLUE, "#7b4fb3", "#2a9d8f"]
    for (name, keep), color in zip(rules.items(), palette):
        d = rs.daily_series(band.loc[keep, "pl_stop"], band.loc[keep, "date"], calendar).cumsum()
        ax.plot(calendar, d, lw=1.8, color=color, label=name)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("cumulative P/L ($), 175% call stop")
    ax.set_title("Band plus each second filter")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# the short/long ratio over time
# --------------------------------------------------------------------------- #
SL_SIDES = {"slratio_call": "call side", "slratio_put": "put side"}
ROLLING_SESSIONS = 60


def slratio_monthly(band: pd.DataFrame) -> pd.DataFrame:
    """Per month: median entry short/long ratio on each side, total P/L, and each ratio's P/L correlation."""
    m = band.assign(month=band["date"].dt.to_period("M"))
    out = m.groupby("month").agg(sessions=("date", "nunique"), total_pl=("pl_stop", "sum"),
                                 slratio_call=("slratio_call", "median"), slratio_put=("slratio_put", "median"))
    for c in SL_SIDES:
        out[f"corr_{c}"] = m.groupby("month").apply(
            lambda g, c=c: spearmanr(g[c], g["pl_stop"]).statistic if len(g) > 20 else np.nan)
    return out


def slratio_rolling_corr(band: pd.DataFrame, window: int = ROLLING_SESSIONS) -> pd.DataFrame:
    """Spearman correlation between each entry ratio and condor P/L over a trailing window of sessions."""
    days = np.sort(band["date"].unique())
    rows = []
    for i in range(window, len(days) + 1):
        w = band[band["date"].isin(days[i - window:i])]
        rows.append({"date": days[i - 1], **{c: spearmanr(w[c], w["pl_stop"]).statistic for c in SL_SIDES}})
    return pd.DataFrame(rows).set_index("date")


def slratio_by_year(band: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Avg P/L per condor by year and by quintile of each entry ratio (quintiles over all sessions)."""
    out = {}
    for c in SL_SIDES:
        q = pd.qcut(band[c], 5, labels=[f"Q{i}" for i in range(1, 6)])
        table = band.groupby([band["date"].dt.year, q], observed=True)["pl_stop"].mean().unstack()
        table.attrs["edges"] = [round(float(e), 2) for e in pd.qcut(band[c], 5, retbins=True)[1]]
        out[c] = table
    return out


def fig_slratio_over_time(band: pd.DataFrame):
    """Three stacked panels on one date axis: the ratios, the P/L, and how the two relate."""
    monthly = slratio_monthly(band)
    rolling = slratio_rolling_corr(band)
    x = monthly.index.to_timestamp()
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True, gridspec_kw={"height_ratios": [1.1, 1, 1]})
    ax = axes[0]
    for c, color in (("slratio_call", RED), ("slratio_put", BLUE)):
        ax.plot(x, monthly[c], marker="o", ms=3, lw=1.8, color=color, label=f"short/long ratio, {SL_SIDES[c]}")
    ax.set_ylabel("median ratio at entry")
    ax.set_title("Entry short/long ratio by month (median of every condor)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.bar(x, monthly["total_pl"] / 1000, width=20, color=np.where(monthly["total_pl"] >= 0, BLUE, RED))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("P/L per month ($k)")
    ax.set_title("Band P/L by month, 175% call stop")
    ax = axes[2]
    for c, color in (("slratio_call", RED), ("slratio_put", BLUE)):
        ax.plot(rolling.index, rolling[c], lw=1.6, color=color, label=SL_SIDES[c])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Spearman(ratio, P/L)")
    ax.set_title(f"How the ratio relates to P/L, trailing {ROLLING_SESSIONS} sessions "
                 "(below zero: a higher ratio meant a worse condor)")
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_slratio_by_year(band: pd.DataFrame):
    tables = slratio_by_year(band)
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
    lim = max(np.nanmax(np.abs(t.to_numpy())) for t in tables.values())
    for ax, (c, t) in zip(axes, tables.items()):
        im = ax.imshow(t.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        edges = t.attrs["edges"]
        ax.set_xticks(range(5), [f"Q{i + 1}\n{edges[i]}-{edges[i + 1]}" for i in range(5)], fontsize=8)
        ax.set_yticks(range(len(t)), t.index)
        for (i, j), v in np.ndenumerate(t.to_numpy()):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"Avg P/L per condor by year and short/long ratio ({SL_SIDES[c]})", fontsize=10)
    fig.colorbar(im, ax=axes, label="avg P/L per condor ($)")
    return fig


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def head_to_head(band: pd.DataFrame, rules: dict[str, pd.Series]) -> pd.DataFrame:
    """P/L and Sharpe of the band under each named keep-rule, over every session."""
    calendar = np.sort(band["date"].unique())
    rows = []
    for name, keep in rules.items():
        st = rs.book_stats(band.loc[keep, "pl_stop"], band.loc[keep, "date"], calendar)
        rows.append({"rule": name, "kept_pct": keep.mean() * 100, "win_rate": st["win_rate"],
                     "avg_pl": st["avg_pl"], "sharpe": st["sharpe"], "total_pl": st["total_pl"],
                     "max_drawdown": st["max_drawdown"]})
    return pd.DataFrame(rows).set_index("rule")


# How many of the highest-ranked features get the year-by-year check, and the ones always included.
N_TOP_FEATURES = 8
ALWAYS_CHECKED = ["sma5_gt_sma9", "ema5_gt_ema9", "ema20m_gt_ema40m"]


def top_feature_years(band: pd.DataFrame, scan: pd.DataFrame) -> pd.DataFrame:
    """Year-by-year Sharpe of the highest-ranked features' rules, SMA/EMA, and the noise feature."""
    ranked = [c for c in scan.index if c != "noise"][:N_TOP_FEATURES]
    chosen = ranked + [c for c in ALWAYS_CHECKED if c not in ranked] + ["noise"]
    rules = {"band only": pd.Series(True, index=band.index)}
    rules.update({f"{c}: {scan.loc[c, 'rule']}": scan.attrs["masks"][c] for c in chosen})
    return rs.year_by_year(band, band["pl_stop"], rules, baseline="band only")


@dataclass
class FeatureStudy:
    band: pd.DataFrame
    scan: pd.DataFrame
    noise_gains: np.ndarray
    tree: TreeResult
    quintiles: pd.DataFrame
    head_to_head: pd.DataFrame
    disagreement: pd.DataFrame
    years: pd.DataFrame


def compute_feature_study(trades_csv: str | Path, term_structure_csv: str | Path | None = None) -> FeatureStudy:
    band, _ = build_band_panel(trades_csv, term_structure_csv)
    scan = filter_scan(band)
    noise_gains = noise_distribution(band)
    scan.attrs["noise_p95"] = float(np.percentile(noise_gains, 95))
    scan["beats_noise_p95"] = scan["sharpe_gain"] > scan.attrs["noise_p95"]
    return FeatureStudy(band=band, scan=scan, noise_gains=noise_gains, tree=tree_importance(band),
                        quintiles=quintiles(band), head_to_head=head_to_head(band, second_filter_rules(band)),
                        disagreement=disagreement(band), years=top_feature_years(band, scan))


def run_feature_study(trades_csv: str | Path, reports_root: Path = rs.REPORTS_ROOT,
                      term_structure_csv: str | Path | None = None) -> FeatureStudy:
    out = reports_root / STUDY_NAME
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    s = compute_feature_study(trades_csv, term_structure_csv)
    s.scan.to_csv(out / "tables" / "filter_scan.csv")
    s.tree.importance.to_csv(out / "tables" / "tree_importance.csv")
    s.quintiles.to_csv(out / "tables" / "quintiles.csv", index=False)
    threshold_table(s.quintiles).to_csv(out / "tables" / "thresholds_by_quintile.csv")
    s.head_to_head.to_csv(out / "tables" / "head_to_head.csv")
    s.disagreement.to_csv(out / "tables" / "sma_vs_ema_disagreement.csv")
    s.years.to_csv(out / "tables" / "top_features_by_year.csv")
    scan_attrs = {k: v for k, v in s.scan.attrs.items() if k != "masks"}
    pd.Series({"tree_spearman": np.mean(s.tree.fold_spearman),
               "null_spearman_mean": np.mean(s.tree.null_spearman),
               "null_spearman_p95": np.percentile(s.tree.null_spearman, 95),
               "sma_ema_agreement_pct": s.disagreement.attrs["agreement_pct"],
               **scan_attrs}).to_csv(out / "tables" / "summary.csv", header=["value"])
    for fig, name in ((fig_filter_scan(s.scan), "filter_scan"), (fig_tree_importance(s.tree), "tree_importance"),
                      (fig_quintile_grid(s.quintiles, FEATURE_COLS), "quintiles_avg_pl"),
                      (fig_quintile_grid(s.quintiles, FEATURE_COLS, "sharpe"), "quintiles_sharpe"),
                      (fig_slratio_over_time(s.band), "slratio_over_time"),
                      (fig_slratio_by_year(s.band), "slratio_by_year"),
                      (fig_head_to_head(s.band, {k: v for k, v in second_filter_rules(s.band).items()
                                                 if k in HEAD_TO_HEAD_PLOTTED}), "head_to_head")):
        fig.savefig(out / "figures" / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    return s
