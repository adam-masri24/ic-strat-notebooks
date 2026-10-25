"""IC inside one regime: ``0.87 <= ts_9d_30_open <= 1.02`` AND SPX SMA5 > SMA9.

Three questions, all asked inside that regime:

1. **Tranches** — which of the 20 entry times earn their place, each scored as
   its own daily book (Sharpe, Sortino, drawdown, daily win rate).
2. **Call-side stop loss** — which stop, as a % of the call vertical's credit,
   does best. The baseline log has no stops (every spread expired), but each
   row carries ``Max Loss``: the spread's worst intraday P/L as a % of premium,
   before fees, bounded by the wing. A stop at X% fires exactly when
   ``max_loss <= -X``, whatever order the path took, so the stop can be replayed
   from the log without an intraday option path. What cannot be replayed is the
   fill: a stop that gaps through its level fills worse, so every scan is run
   at two slippage levels.
3. **Features** — which entry features still move P/L inside the regime. Only
   columns that can be set as an entry filter in the backtester are scored as
   candidates (``FEATURES``); everything else here is a control.

Every inference is at the **session** level. Twenty tranches share one SPX path
per day, so trades are not independent observations — see
:func:`effective_sample`. Everything is chosen and scored on every session (no
train/test split), so every gain is in-sample; each feature result is read
against four noise columns put through the identical procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.paths import strategy_reports_root

from .ic_analysis import load_ic_trades
from .pipeline import PRICE_CROSSOVER_COL, build_panel
from .term_structure_features import TS_9D_30, load_term_structure_daily

STUDY_NAME = "SPX-IC-Regime-TS087-102-SMA5gt9"
TS_LOW, TS_HIGH = 0.87, 1.02
# The wider cap tested against TS_HIGH (see cap_comparison).
ALT_TS_HIGH = 1.04
REPORTS_ROOT = strategy_reports_root("ic_strat")

# Stop levels, % of the call vertical's credit. None = no stop (the baseline).
STOP_GRID: list[float | None] = [None, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400]
# Closing a stopped spread pays commissions again; an expired one doesn't.
CLOSE_FEE = 1.94
# Extra $ lost per stopped spread for a fill through the stop level.
SLIPPAGE_LEVELS = (0.0, 20.0)
# Middle of the 125-200% plateau that holds in both periods and at both slippage levels.
RECOMMENDED_CALL_STOP = 175

TRADING_DAYS = 252
SEED = 0
N_BOOT = 1000

# Candidate features -> what each one is. All can be set as an entry filter.
FEATURES = {
    "vix30_open": "VIX level at the open",
    "vix9d_open": "VIX9D level at the open",
    "vix_overnight_pct": "VIX overnight move, %",
    "gap_pct": "SPX opening gap, %",
    "spx_move_pct": "SPX intraday movement since the 09:30 open, %",
    "credit": "premium collected, whole condor",
    "credit_call": "premium collected, call side",
    "credit_put": "premium collected, put side",
    "slratio_call": "short/long price ratio, call side",
    "slratio_put": "short/long price ratio, put side",
    "rsi14_lagged": "RSI(14) on SPX, prior close",
    "mins_from_open": "entry time",
    "dow_code": "day of week",
}
# Families for cluster importance: correlated columns are permuted together.
FAMILIES = {
    "vix_level": ["vix30_open", "vix9d_open"],
    "vix_overnight": ["vix_overnight_pct"],
    "spx_gap": ["gap_pct"],
    "spx_move": ["spx_move_pct"],
    "premium": ["credit", "credit_call", "credit_put"],
    "short_long_ratio": ["slratio_call", "slratio_put"],
    "rsi": ["rsi14_lagged"],
    "entry_time": ["mins_from_open"],
    "weekday": ["dow_code"],
    "noise": ["noise_row", "noise_row2", "noise_day", "noise_day2"],
}
NOISE = FAMILIES["noise"]
FEATURE_COLS = list(FEATURES) + NOISE
# The features whose quintiles the study shows: the one that beats noise, and the two runners-up.
QUINTILE_FEATURES = ["spx_move_pct", "vix9d_open", "slratio_call"]


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window).mean()
    return 100 - 100 / (1 + gain / loss)


def build_regime_panel(
    trades_csv: str | Path,
    term_structure_csv: str | Path | None = None,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(panel, spreads)``: every condor with regime flag and features, and the per-side rows.

    ``spreads`` keeps one row per vertical with ``max_loss`` so stops can be
    replayed per side; ``panel`` is :func:`pipeline.build_panel` plus the
    candidate feature columns and four noise controls.
    """
    panel = build_panel(trades_csv, term_structure_csv)
    spreads = load_ic_trades(trades_csv).rename(columns={"date_opened": "date", "time_opened": "entry_time"})
    spreads["date"] = pd.to_datetime(spreads["date"])

    ts_kwargs = {} if term_structure_csv is None else {"path": term_structure_csv}
    daily = load_term_structure_daily(**ts_kwargs)[["date", "vix30_close"]].copy()
    # Yesterday's close, taken from the daily file before any join so it's never same-day.
    daily["vix30_close_prev"] = daily["vix30_close"].shift(1)
    panel = panel.merge(daily[["date", "vix30_close_prev"]], on="date", how="left")
    panel["vix_overnight_pct"] = (panel["vix30_open"] / panel["vix30_close_prev"] - 1) * 100

    # Gap and Movement are in points in the export; spot less Movement is the 09:30 open,
    # and the open less the gap is the prior close.
    session_open = panel["spot"] - panel["movement"]
    panel["gap_pct"] = panel["gap"] / (session_open - panel["gap"]) * 100

    closes = panel.groupby("date")["close"].first()
    rsi = _rsi(closes).shift(1).rename("rsi14_lagged")
    panel = panel.merge(rsi, left_on="date", right_index=True, how="left")

    rng = np.random.default_rng(seed)
    days = panel["date"].unique()
    panel["noise_row"] = rng.standard_normal(len(panel))
    panel["noise_row2"] = rng.standard_normal(len(panel))
    panel["noise_day"] = panel["date"].map(dict(zip(days, rng.standard_normal(len(days)))))
    panel["noise_day2"] = panel["date"].map(dict(zip(days, rng.standard_normal(len(days)))))

    panel["in_regime"] = (
        panel[TS_9D_30].between(TS_LOW, TS_HIGH) & (panel[PRICE_CROSSOVER_COL] == True)  # noqa: E712
    )
    return panel, spreads


def side_pl(spreads: pd.DataFrame, side: str, stop_pct: float | None, slippage: float = 0.0) -> pd.Series:
    """Replay one side's P/L under a stop at ``stop_pct``% of its credit, indexed by (date, entry_time).

    A stopped spread loses ``stop_pct``% of premium plus opening and closing
    fees plus ``slippage``, capped at the wing's max loss. A stop wider than the
    wing never fires, because ``max_loss`` can't go past the wing.
    """
    legs = spreads[spreads["side"] == side].set_index(["date", "entry_time"])
    if stop_pct is None:
        return legs["p_l"]
    stopped = legs["max_loss"] <= -stop_pct
    wing_cap = legs["wing_width"] * 100 - legs["premium"]
    loss = np.minimum(stop_pct / 100 * legs["premium"] + slippage, wing_cap)
    stopped_pl = -loss - legs["opening_commissions_fees"] - CLOSE_FEE
    return legs["p_l"].where(~stopped, stopped_pl)


def condor_pl_with_call_stop(
    panel: pd.DataFrame, spreads: pd.DataFrame, stop_pct: float | None, slippage: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Condor P/L with a stop on the call side only, puts held to expiry. Also returns the stop-hit flag."""
    key = pd.MultiIndex.from_frame(panel[["date", "entry_time"]])
    call = side_pl(spreads, "call", stop_pct, slippage).reindex(key).to_numpy()
    put = side_pl(spreads, "put", None).reindex(key).to_numpy()
    calls = spreads[spreads["side"] == "call"].set_index(["date", "entry_time"])["max_loss"].reindex(key)
    hit = (calls <= -stop_pct).to_numpy() if stop_pct is not None else np.zeros(len(panel), bool)
    return pd.Series(call + put, index=panel.index), pd.Series(hit, index=panel.index)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def effective_sample(df: pd.DataFrame, value: str = "pl") -> dict:
    """Day-level ICC, design effect, and effective sample size of ``value``."""
    mu = df[value].mean()
    g = df.groupby("date")[value]
    icc = float(g.mean().sub(mu).pow(2).mul(g.size()).sum() / (df[value] - mu).pow(2).sum())
    k = float(g.size().mean())
    deff = 1 + (k - 1) * icc
    return {"icc_day": icc, "trades_per_day": k, "design_effect": deff,
            "effective_n": len(df) / deff, "nominal_n": len(df), "sessions": int(df["date"].nunique())}


def breakeven_win_rate(pl: pd.Series) -> float:
    wins, losses = pl[pl > 0].mean(), pl[pl <= 0].mean()
    return float(-losses / (wins - losses) * 100)


def daily_series(pl: pd.Series, dates: pd.Series, calendar: np.ndarray) -> pd.Series:
    """Summed P/L per session over ``calendar``, no-trade days as zero."""
    return pl.groupby(dates.to_numpy()).sum().reindex(calendar, fill_value=0.0)


def book_stats(pl: pd.Series, dates: pd.Series, calendar: np.ndarray) -> dict:
    """Per-trade and portfolio stats; no-trade days in ``calendar`` count as zero."""
    daily = daily_series(pl, dates, calendar)
    equity = daily.cumsum()
    sd = daily.std()
    return {
        "trades": int(len(pl)),
        "win_rate": float((pl > 0).mean() * 100),
        "avg_pl": float(pl.mean()),
        "total_pl": float(pl.sum()),
        "sharpe": float(daily.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
        "max_drawdown": float((equity - equity.cummax()).min()),
    }


def session_bootstrap_mean(values: pd.Series, dates: pd.Series, n_boot: int = N_BOOT,
                           seed: int = SEED) -> tuple[float, float]:
    """95% interval for the per-trade mean, resampling whole sessions."""
    rng = np.random.default_rng(seed)
    by_day = pd.DataFrame({"v": values.to_numpy(), "d": dates.to_numpy()}).groupby("d")["v"].agg(["sum", "size"])
    s, n = by_day["sum"].to_numpy(), by_day["size"].to_numpy()
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    means = s[idx].sum(1) / n[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def year_by_year(frame: pd.DataFrame, pl: pd.Series, rules: dict[str, pd.Series],
                 baseline: str) -> pd.DataFrame:
    """Sharpe of each keep-rule in each calendar year, and in how many years it beats ``baseline``.

    Not a train/test split: every rule was chosen on all sessions. It shows
    whether a rule's edge is spread across the years or sits in one stretch.
    Each year's Sharpe counts that year's sessions of ``frame``, skipped days as zero.
    """
    years = sorted(frame["date"].dt.year.unique())
    rows = []
    for name, keep in rules.items():
        row = {"rule": name}
        for y in years:
            days = np.sort(frame.loc[frame["date"].dt.year == y, "date"].unique())
            m = keep & (frame["date"].dt.year == y)
            row[y] = book_stats(pl[m], frame.loc[m, "date"], days)["sharpe"]
        rows.append(row)
    out = pd.DataFrame(rows).set_index("rule")
    base = out.loc[baseline, years]
    out["years_better_than_baseline"] = (out[years] > base).sum(axis=1)
    out.loc[baseline, "years_better_than_baseline"] = np.nan
    out.attrs["years"] = years
    return out


# --------------------------------------------------------------------------- #
# 1. tranches
# --------------------------------------------------------------------------- #
def tranche_daily_pivot(regime: pd.DataFrame, pl: pd.Series) -> pd.DataFrame:
    """Session × entry-time matrix of P/L; a tranche that didn't fill that day is zero."""
    return (pd.DataFrame({"date": regime["date"], "entry_time": regime["entry_time"], "pl": pl})
            .pivot_table(index="date", columns="entry_time", values="pl", aggfunc="sum", fill_value=0.0))


def _daily_ratios(daily: pd.Series) -> dict:
    equity = daily.cumsum()
    downside = daily[daily < 0].std()
    gains, losses = daily[daily > 0].sum(), -daily[daily < 0].sum()
    return {
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS)) if daily.std() > 0 else np.nan,
        "sortino": float(daily.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan,
        "max_drawdown": float((equity - equity.cummax()).min()),
        "daily_win_rate": float((daily > 0).mean() * 100),
        "profit_factor": float(gains / losses) if losses > 0 else np.nan,
        "total_pl": float(daily.sum()),
    }


def tranche_table(regime: pd.DataFrame, pl: pd.Series) -> pd.DataFrame:
    """Each entry time as its own daily book over every session, with a session-bootstrap CI."""
    pivot = tranche_daily_pivot(regime, pl)
    rows = []
    for t in pivot.columns:
        grp = regime[regime["entry_time"] == t]
        p = pl.loc[grp.index]
        lo, hi = session_bootstrap_mean(p, grp["date"])
        rows.append({
            "entry_time": t, "sessions": int(grp["date"].nunique()),
            "win_rate": float((p > 0).mean() * 100), "avg_pl": float(p.mean()),
            "ci_low": lo, "ci_high": hi, **_daily_ratios(pivot[t]),
        })
    return pd.DataFrame(rows).set_index("entry_time")


# --------------------------------------------------------------------------- #
# 2. call stop
# --------------------------------------------------------------------------- #
def call_stop_scan(regime: pd.DataFrame, spreads: pd.DataFrame, slippage: float) -> pd.DataFrame:
    """Every stop level over every strategy session."""
    rows = []
    calendar = np.sort(regime["date"].unique())
    base_pl, _ = condor_pl_with_call_stop(regime, spreads, None)
    for stop in STOP_GRID:
        pl, hit = condor_pl_with_call_stop(regime, spreads, stop, slippage)
        row = {"stop_pct": "none" if stop is None else stop, "call_stop_hit_rate": float(hit.mean() * 100)}
        row.update({f"all_{k}": v for k, v in book_stats(pl, regime["date"], calendar).items() if k != "trades"})
        diff = pl - base_pl
        lo, hi = session_bootstrap_mean(diff, regime["date"])
        row.update({"vs_no_stop_avg": float(diff.mean()), "vs_no_stop_ci_low": lo, "vs_no_stop_ci_high": hi})
        rows.append(row)
    return pd.DataFrame(rows).set_index("stop_pct")


def tranche_stop_grid(regime: pd.DataFrame, spreads: pd.DataFrame, slippage: float) -> pd.DataFrame:
    """Per tranche: avg P/L at every stop level, the best one, and the recommended shared stop."""
    by_stop = {s: condor_pl_with_call_stop(regime, spreads, s, slippage)[0] for s in STOP_GRID}
    rows = []
    for t, idx in regime.groupby("entry_time").groups.items():
        avg = {s: by_stop[s].loc[idx].mean() for s in STOP_GRID}
        best = max(avg, key=avg.get)
        rows.append({
            "entry_time": t, "best_stop": "none" if best is None else best, "avg_pl_best": avg[best],
            f"avg_pl_stop_{RECOMMENDED_CALL_STOP}": avg[RECOMMENDED_CALL_STOP], "avg_pl_no_stop": avg[None],
            **{f"avg_pl_stop_{'none' if s is None else s}": v for s, v in avg.items()},
        })
    return pd.DataFrame(rows).set_index("entry_time")


# --------------------------------------------------------------------------- #
# 3. features
# --------------------------------------------------------------------------- #
@dataclass
class ImportanceResult:
    auc_folds: list[float]
    null_auc: list[float]
    column: pd.DataFrame
    family: pd.DataFrame


def _model() -> HistGradientBoostingClassifier:
    # Shallow and heavily regularised: the effective sample is under a thousand.
    return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=SEED)


def _permute_by_session(values: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle whole sessions' values onto other sessions, keeping within-day structure."""
    days = np.unique(groups)
    donor = dict(zip(days, rng.permutation(days)))
    by_day = {d: values[groups == d] for d in days}
    out = np.empty_like(values)
    for d in days:
        rows = np.where(groups == d)[0]
        src = by_day[donor[d]]
        out[rows] = src[np.arange(len(rows)) % len(src)]
    return out


def feature_importance(regime: pd.DataFrame, n_repeats: int = 5, n_null: int = 20) -> ImportanceResult:
    """Permutation importance per column and per family, fitted and scored on every session.

    With no held-out data the model's own AUC is optimistic, so it is read
    against the same model fitted on labels shuffled between whole sessions,
    and every column against the noise columns fitted alongside it.
    """
    X = regime[FEATURE_COLS].to_numpy(float)
    y = regime["win"].to_numpy()
    g = regime["date"].to_numpy()
    rng = np.random.default_rng(SEED)
    model = _model().fit(X, y)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])

    def drop_for(cols: list[str]) -> float:
        # Permute by session so a day-level column moves as a unit.
        perm = _permute_by_session(np.arange(len(y)), g, rng)
        Xp = X.copy()
        for c in cols:
            j = FEATURE_COLS.index(c)
            Xp[:, j] = X[perm, j]
        return base - roc_auc_score(y, model.predict_proba(Xp)[:, 1])

    col_drop = {c: [drop_for([c]) for _ in range(n_repeats)] for c in FEATURE_COLS}
    fam_drop = {f: [drop_for(cols) for _ in range(n_repeats)] for f, cols in FAMILIES.items()}
    null = []
    for _ in range(n_null):
        yp = _permute_by_session(y, g, rng)
        null.append(float(roc_auc_score(yp, _model().fit(X, yp).predict_proba(X)[:, 1])))

    def frame(d):
        return pd.DataFrame({k: {"mean": np.mean(v), "sd": np.std(v)} for k, v in d.items()}).T

    col = frame(col_drop)
    col["description"] = [FEATURES.get(c, "noise control") for c in col.index]
    noise_bar = float((col.loc[NOISE, "mean"] + 2 * col.loc[NOISE, "sd"]).max())
    col["clears_noise_bar"] = col["mean"] > noise_bar
    fam = frame(fam_drop)
    fam_bar = float(fam.loc["noise", "mean"] + 2 * fam.loc["noise", "sd"])
    fam["clears_noise_bar"] = fam["mean"] > fam_bar
    col, fam = col.sort_values("mean", ascending=False), fam.sort_values("mean", ascending=False)
    col.attrs["noise_bar"], fam.attrs["noise_bar"] = noise_bar, fam_bar
    return ImportanceResult([float(base)], null, col, fam)


def quintile_table(regime: pd.DataFrame, pl: pd.Series) -> pd.DataFrame:
    """P/L by quintile of each continuous feature, with session counts per cell."""
    rows = []
    for c in [c for c in FEATURES if c not in ("dow_code", "mins_from_open")] + NOISE:
        q = pd.qcut(regime[c], 5, duplicates="drop")
        for i, (interval, grp) in enumerate(regime.groupby(q, observed=True)):
            p = pl.loc[grp.index]
            rows.append({"feature": c, "quintile": i + 1, "range": str(interval),
                         "trades": len(grp), "sessions": grp["date"].nunique(),
                         "win_rate": (p > 0).mean() * 100, "avg_pl": p.mean()})
    return pd.DataFrame(rows)


def threshold_rules(regime: pd.DataFrame, pl: pd.Series, min_keep: float = 0.30) -> pd.DataFrame:
    """For each feature, the best single min or max cut over every strategy session.

    Scored by portfolio Sharpe on daily summed P/L with skipped days as zero, the
    number a backtest report shows. Candidate cuts are the deciles; a cut must
    keep at least ``min_keep`` of trades. Chosen and scored on the same sessions,
    so every gain is in-sample: the noise columns go through the identical
    search, and their best gain is the bar a real feature has to clear.
    """
    calendar = np.sort(regime["date"].unique())
    base = book_stats(pl, regime["date"], calendar)
    rows, masks = [], {}
    for c in [c for c in FEATURES if c != "dow_code"] + NOISE:
        best = None
        for q in np.arange(0.1, 1.0, 0.1):
            cut = regime[c].quantile(q)
            for direction in ("min", "max"):
                keep = regime[c] >= cut if direction == "min" else regime[c] <= cut
                if keep.mean() < min_keep:
                    continue
                st = book_stats(pl[keep], regime.loc[keep, "date"], calendar)
                if best is None or st["sharpe"] > best[0]["sharpe"]:
                    best = (st, cut, direction, keep)
        st, cut, direction, keep = best
        masks[c] = keep
        rows.append({"feature": c, "description": FEATURES.get(c, "noise control"),
                     "rule": f"{c} {'>=' if direction == 'min' else '<='} {cut:.4g}",
                     "sharpe": st["sharpe"], "baseline_sharpe": base["sharpe"],
                     "sharpe_gain": st["sharpe"] - base["sharpe"], "trades_kept_pct": keep.mean() * 100,
                     "avg_pl": st["avg_pl"], "baseline_avg_pl": base["avg_pl"],
                     "total_pl": st["total_pl"], "max_drawdown": st["max_drawdown"]})
    out = pd.DataFrame(rows).set_index("feature").sort_values("sharpe_gain", ascending=False)
    bar = float(out.loc[NOISE, "sharpe_gain"].max())
    out["beats_best_noise"] = out["sharpe_gain"] > bar
    out.attrs["noise_bar"] = bar
    out.attrs["masks"] = masks
    return out


N_TOP_FEATURES = 5


def top_feature_years(regime: pd.DataFrame, pl: pd.Series, rules: pd.DataFrame) -> pd.DataFrame:
    """Year-by-year Sharpe of the highest-ranked features' rules, and of the best noise column's."""
    real = [c for c in rules.index if c not in NOISE][:N_TOP_FEATURES]
    noise = [c for c in rules.index if c in NOISE][:1]
    keep = {"strategy only": pd.Series(True, index=regime.index)}
    keep.update({f"{c}: {rules.loc[c, 'rule']}": rules.attrs["masks"][c] for c in real + noise})
    return year_by_year(regime, pl, keep, baseline="strategy only")


def cap_comparison(panel: pd.DataFrame, spreads: pd.DataFrame,
                   stop_pct: float = RECOMMENDED_CALL_STOP, slippage: float = 20.0) -> pd.DataFrame:
    """The strategy capped at ``TS_HIGH`` vs ``ALT_TS_HIGH``, and the slice between them.

    Every row uses the same SMA5 > SMA9 condition and the same call stop, over
    every session. Sharpe counts all 998 sessions, skipped days as zero, so the
    two caps are compared on one calendar, the way a backtest of either would be.
    """
    sma = panel[PRICE_CROSSOVER_COL] == True  # noqa: E712
    ts = panel[TS_9D_30]
    calendar = np.sort(panel["date"].unique())
    rows = []
    for label, mask in (
        (f"{TS_LOW}-{TS_HIGH} (the strategy)", ts.between(TS_LOW, TS_HIGH) & sma),
        (f"{TS_LOW}-{ALT_TS_HIGH} (wider cap)", ts.between(TS_LOW, ALT_TS_HIGH) & sma),
        (f"{TS_HIGH}-{ALT_TS_HIGH} slice only", (ts > TS_HIGH) & (ts <= ALT_TS_HIGH) & sma),
    ):
        book = panel[mask]
        stopped, _ = condor_pl_with_call_stop(book, spreads, stop_pct, slippage)
        st = book_stats(stopped, book["date"], calendar)
        sessions = int(book["date"].nunique())
        rows.append({"book": label, "sessions": sessions,
                     "total_pl_no_stop": float(book["pl"].sum()), "total_pl_with_stop": float(stopped.sum()),
                     "pl_per_session": st["total_pl"] / max(sessions, 1),
                     "win_rate": st["win_rate"], "sharpe": st["sharpe"], "max_drawdown": st["max_drawdown"]})
    return pd.DataFrame(rows).set_index("book")


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def weekday_table(regime: pd.DataFrame, pl: pd.Series) -> pd.DataFrame:
    """Session-level weekday comparison over every session: each session's total P/L, not trade-weighted."""
    daily = pd.DataFrame({"pl": pl, "date": regime["date"], "dow": regime["dow"]}).groupby(
        ["date", "dow"], as_index=False)["pl"].sum()
    out = daily.groupby("dow")["pl"].agg(avg_session_pl="mean", sessions="size", total_pl="sum")
    out["winning_sessions_pct"] = daily.assign(w=daily["pl"] > 0).groupby("dow")["w"].mean() * 100
    return out.reindex([d for d in WEEKDAYS if d in out.index])


def weekday_tranche_grid(regime: pd.DataFrame, pl: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avg P/L and session count for every weekday × entry-time cell."""
    frame = pd.DataFrame({"pl": pl, "dow": regime["dow"], "t": regime["entry_time"], "d": regime["date"]})
    avg = frame.pivot_table(index="dow", columns="t", values="pl", aggfunc="mean").reindex(WEEKDAYS)
    sessions = frame.pivot_table(index="dow", columns="t", values="d", aggfunc="nunique").reindex(WEEKDAYS)
    return avg, sessions


# --------------------------------------------------------------------------- #
# figures — each returns the figure so the notebook can show it inline
# --------------------------------------------------------------------------- #
BLUE, RED, GREY = "#2a78d6", "#e34948", "#8a8f98"
REGIME_LABEL = f"{TS_LOW} ≤ VIX9D/VIX ≤ {TS_HIGH}, SMA5 > SMA9"


def _hhmm(labels) -> list[str]:
    return [str(t)[:5] for t in labels]


TS_BINS = np.round(np.arange(0.75, 1.1501, 0.025), 3)


def ts_sma_band_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Avg daily P/L per ``ts_9d_30_open`` bin, split by the lagged SPX SMA5 > SMA9 state.

    Session level: each session's summed condor P/L is one observation. Fixed
    bins on a shared range, so the two states' curves read against each other.
    """
    daily = (panel.dropna(subset=[PRICE_CROSSOVER_COL])
             .groupby("date").agg(pl=("pl", "sum"), ts=(TS_9D_30, "first"), sma=(PRICE_CROSSOVER_COL, "first")))
    daily["bin"] = pd.cut(daily["ts"], TS_BINS)
    out = daily.groupby(["bin", "sma"], observed=True)["pl"].agg(["mean", "size"]).reset_index()
    out["bin_mid"] = out["bin"].map(lambda b: b.mid).astype(float)
    out["sma_state"] = np.where(out["sma"].astype(bool), "SMA5 > SMA9", "SMA5 <= SMA9")
    return out.drop(columns="sma").rename(columns={"mean": "avg_daily_pl", "size": "sessions"})


def fig_ts_sma_band(band: pd.DataFrame, min_sessions: int = 10):
    """The regime's two conditions on one chart: the ts_9d_30 band shaded, a line per SMA state."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    ax.axvspan(TS_LOW, TS_HIGH, color=BLUE, alpha=0.08, label=f"chosen band {TS_LOW}-{TS_HIGH}")
    for state, color in (("SMA5 > SMA9", BLUE), ("SMA5 <= SMA9", RED)):
        b = band[(band["sma_state"] == state) & (band["sessions"] >= min_sessions)]
        ax.plot(b["bin_mid"], b["avg_daily_pl"], marker="o", ms=5, lw=2, color=color, label=state)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("VIX9D / VIX30 at the open (ts_9d_30_open), 0.025-wide bins")
    ax.set_ylabel("avg daily P/L, all 20 tranches ($)")
    ax.set_title(f"Where the regime sits (bins under {min_sessions} sessions dropped)")
    ax.legend(frameon=False)
    ax = axes[1]
    counts = band.pivot_table(index="bin_mid", columns="sma_state", values="sessions", fill_value=0)
    ax.bar(counts.index, counts.get("SMA5 > SMA9", 0), width=0.02, color=BLUE, label="SMA5 > SMA9")
    ax.bar(counts.index, counts.get("SMA5 <= SMA9", 0), width=0.02, color=RED, alpha=0.6,
           bottom=counts.get("SMA5 > SMA9", 0), label="SMA5 <= SMA9")
    ax.axvline(TS_LOW, color="k", ls=":", lw=1)
    ax.axvline(TS_HIGH, color="k", ls=":", lw=1)
    ax.set_xlabel("ts_9d_30_open")
    ax.set_title("Sessions per bin")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def regime_state_table(panel: pd.DataFrame, calendar: np.ndarray) -> pd.DataFrame:
    """The four cells of (inside the ts band?) × (SMA5 > SMA9?), each as its own book."""
    p = panel.dropna(subset=[PRICE_CROSSOVER_COL])
    in_band = p[TS_9D_30].between(TS_LOW, TS_HIGH)
    sma = p[PRICE_CROSSOVER_COL] == True  # noqa: E712
    rows = []
    for band_ok in (True, False):
        for sma_ok in (True, False):
            m = (in_band == band_ok) & (sma == sma_ok)
            rows.append({"ts_band": f"{TS_LOW}-{TS_HIGH}" if band_ok else "outside",
                         "sma": "SMA5 > SMA9" if sma_ok else "SMA5 <= SMA9",
                         "sessions": int(p.loc[m, "date"].nunique()),
                         **book_stats(p.loc[m, "pl"], p.loc[m, "date"], calendar)})
    return pd.DataFrame(rows).set_index(["ts_band", "sma"])


def fig_regime_equity(panel: pd.DataFrame, regime_pl_stopped: pd.Series, regime: pd.DataFrame):
    """Cumulative P/L: whole book, the regime held to expiry, and the regime with the call stop."""
    calendar = np.sort(panel["date"].unique())
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for label, pl, dates, color, style in (
        ("every session, no stop", panel["pl"], panel["date"], GREY, "-"),
        ("regime only, no stop", regime["pl"], regime["date"], BLUE, "--"),
        (f"regime only, {RECOMMENDED_CALL_STOP}% call stop, $20 slippage", regime_pl_stopped, regime["date"], BLUE, "-"),
    ):
        ax.plot(calendar, daily_series(pl, dates, calendar).cumsum(), style, color=color, lw=1.8, label=label)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("cumulative P/L ($)")
    ax.set_title(f"Equity: whole book vs. the regime ({REGIME_LABEL})")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_tranches(tr: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    x = np.arange(len(tr))
    ax = axes[0]
    ax.bar(x, tr["avg_pl"], color=np.where(tr["avg_pl"] >= 0, BLUE, RED), width=0.7)
    ax.errorbar(x, tr["avg_pl"], yerr=[tr["avg_pl"] - tr["ci_low"], tr["ci_high"] - tr["avg_pl"]],
                fmt="none", ecolor="#333", lw=0.8, capsize=2)
    ax.set_title("Avg P/L per condor, 95% session-bootstrap CI")
    ax = axes[1]
    ax.bar(x, tr["sharpe"], color=np.where(tr["sharpe"] >= 0, BLUE, RED), width=0.7)
    ax.set_title("Sharpe per tranche, each as its own daily book")
    ax = axes[2]
    ax.bar(x, tr["max_drawdown"], color=RED, width=0.7)
    ax.set_title("Max drawdown per tranche ($)")
    for ax in axes:
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, _hhmm(tr.index), rotation=90, fontsize=8)
    fig.suptitle(f"Tranches inside the regime ({REGIME_LABEL})")
    fig.tight_layout()
    return fig


SLIP_COLORS = {0.0: BLUE, 20.0: "#e8a33d"}


def fig_stop(scans: dict):
    """Every stop level at both slippage levels: Sharpe and max drawdown over every session."""
    labels = [str(i) for i in next(iter(scans.values())).index]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, col, title, scale in ((axes[0], "all_sharpe", "Portfolio Sharpe by call stop", 1),
                                  (axes[1], "all_max_drawdown", "Max drawdown by call stop ($k)", 1000)):
        for slip, scan in scans.items():
            ax.plot(labels, scan[col] / scale, marker="o", ms=5, lw=2.2,
                    color=SLIP_COLORS.get(slip, GREY), label=f"${slip:.0f} slippage per stop")
        ax.set_title(title)
        ax.set_xlabel("call-side stop, % of call credit")
        ax.axvspan(labels.index("125") - 0.3, labels.index("200") + 0.3, color=GREY, alpha=0.1)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Call-side stop loss, every strategy session (shaded: 125-200%)")
    fig.tight_layout()
    return fig


def fig_weekdays(weekdays: pd.DataFrame):
    """Session-level weekday P/L over every session."""
    x = np.arange(len(weekdays))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x, weekdays["avg_session_pl"], color=np.where(weekdays["avg_session_pl"] >= 0, BLUE, RED), width=0.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, [f"{d}\n{n} sessions" for d, n in zip(weekdays.index, weekdays["sessions"])])
    ax.set_ylabel("avg P/L per session, all tranches ($)")
    ax.set_title(f"Day of week inside the strategy ({RECOMMENDED_CALL_STOP}% call stop)")
    fig.tight_layout()
    return fig


def fig_importance(col: pd.DataFrame):
    col = col.sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = [GREY if c in NOISE else (BLUE if ok else "#9dbfe9") for c, ok in zip(col.index, col["clears_noise_bar"])]
    ax.barh([FEATURES.get(c, c) for c in col.index], col["mean"], xerr=col["sd"], color=colors,
            ecolor="#555", capsize=2)
    ax.axvline(col.attrs["noise_bar"], color=RED, ls="--", lw=1.2, label="noise bar (best noise mean + 2 SD)")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("AUC drop when permuted (every session)")
    ax.set_title("Feature importance inside the strategy (grey = noise controls)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def fig_rules(rules: pd.DataFrame):
    r = rules.sort_values("sharpe_gain")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh([FEATURES.get(c, c) for c in r.index], r["sharpe_gain"],
            color=[GREY if c in NOISE else (BLUE if g > 0 else RED) for c, g in zip(r.index, r["sharpe_gain"])])
    ax.axvline(rules.attrs["noise_bar"], color=RED, ls="--", lw=1.2, label="best noise-column gain")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Sharpe change vs. the strategy alone, every session")
    ax.set_title("Best single min/max cut per feature")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def fig_quintiles(quintiles: pd.DataFrame, features: list[str]):
    fig, axes = plt.subplots(1, len(features), figsize=(5.8 * len(features), 4.6), squeeze=False)
    for ax, f in zip(axes[0], features):
        q = quintiles[quintiles["feature"] == f]
        ax.bar(q["quintile"], q["avg_pl"], color=np.where(q["avg_pl"] >= 0, BLUE, RED), width=0.7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(q["quintile"], [f"{r}\n{s} sess." for r, s in zip(q["range"], q["sessions"])], fontsize=7)
        ax.set_title(f"{FEATURES.get(f, f)}: avg P/L per condor by quintile", fontsize=10)
    fig.tight_layout()
    return fig


def fig_weekday_tranche(avg: pd.DataFrame, sessions: pd.DataFrame, min_sessions: int = 40):
    """Weekday × entry-time grid; cells under ``min_sessions`` are greyed rather than read."""
    data = avg.where(sessions >= min_sessions)
    lim = np.nanmax(np.abs(data.to_numpy()))
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_facecolor("#e6e6e6")
    im = ax.imshow(data, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(avg.shape[1]), _hhmm(avg.columns), rotation=90, fontsize=8)
    ax.set_yticks(range(len(avg)), avg.index)
    fig.colorbar(im, ax=ax, label="avg P/L per condor ($)")
    ax.set_title(f"Weekday × tranche, regime sessions (grey = under {min_sessions} sessions)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
@dataclass
class RegimeStudy:
    panel: pd.DataFrame
    spreads: pd.DataFrame
    regime: pd.DataFrame
    stopped_pl: pd.Series
    summary: dict
    tranches: pd.DataFrame
    stop_scans: dict
    tranche_stops: pd.DataFrame
    importance: ImportanceResult
    quintiles: pd.DataFrame
    rules: pd.DataFrame
    rules_stopped: pd.DataFrame
    rules_years: pd.DataFrame
    cap_comparison: pd.DataFrame
    weekdays: pd.DataFrame
    weekday_grid: tuple[pd.DataFrame, pd.DataFrame]
    ts_sma_band: pd.DataFrame
    regime_states: pd.DataFrame


def compute_regime_study(trades_csv: str | Path, term_structure_csv: str | Path | None = None) -> RegimeStudy:
    """Every table in the study, computed once; nothing is written."""
    panel, spreads = build_regime_panel(trades_csv, term_structure_csv)
    regime = panel[panel["in_regime"]].copy()
    all_days = np.sort(panel["date"].unique())
    stopped_pl, _ = condor_pl_with_call_stop(regime, spreads, RECOMMENDED_CALL_STOP, SLIPPAGE_LEVELS[-1])

    summary = {
        "regime_condors": len(regime), "regime_sessions": int(regime["date"].nunique()),
        "all_sessions": len(all_days),
        "breakeven_win_rate": breakeven_win_rate(regime["pl"]),
        "avg_win": float(regime.loc[regime.pl > 0, "pl"].mean()),
        "avg_loss": float(regime.loc[regime.pl <= 0, "pl"].mean()),
        **{f"regime_{k}": v for k, v in book_stats(regime["pl"], regime["date"], all_days).items()},
        **{f"full_book_{k}": v for k, v in book_stats(panel["pl"], panel["date"], all_days).items()},
        **effective_sample(regime),
    }

    importance = feature_importance(regime)
    summary.update({"model_auc": float(np.mean(importance.auc_folds)),
                    "null_auc_mean": float(np.mean(importance.null_auc)),
                    "null_auc_p95": float(np.percentile(importance.null_auc, 95)),
                    "column_noise_bar": importance.column.attrs["noise_bar"],
                    "family_noise_bar": importance.family.attrs["noise_bar"]})
    rules = threshold_rules(regime, regime["pl"])
    rules_stopped = threshold_rules(regime, stopped_pl)
    summary.update({"rule_noise_bar": rules.attrs["noise_bar"],
                    "rule_noise_bar_with_stop": rules_stopped.attrs["noise_bar"]})

    return RegimeStudy(
        panel=panel, spreads=spreads, regime=regime, stopped_pl=stopped_pl, summary=summary,
        tranches=tranche_table(regime, regime["pl"]),
        stop_scans={s: call_stop_scan(regime, spreads, s) for s in SLIPPAGE_LEVELS},
        tranche_stops=tranche_stop_grid(regime, spreads, SLIPPAGE_LEVELS[-1]),
        importance=importance,
        quintiles=quintile_table(regime, stopped_pl),
        rules=rules, rules_stopped=rules_stopped,
        rules_years=top_feature_years(regime, stopped_pl, rules_stopped),
        cap_comparison=cap_comparison(panel, spreads),
        weekdays=weekday_table(regime, stopped_pl),
        weekday_grid=weekday_tranche_grid(regime, stopped_pl),
        ts_sma_band=ts_sma_band_table(panel),
        regime_states=regime_state_table(panel, all_days),
    )


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_regime_study(trades_csv: str | Path, reports_root: Path = REPORTS_ROOT,
                     term_structure_csv: str | Path | None = None) -> RegimeStudy:
    """Compute the study and write every table and figure under ``reports_root/STUDY_NAME``."""
    out = reports_root / STUDY_NAME
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    s = compute_regime_study(trades_csv, term_structure_csv)
    pd.Series(s.summary).to_csv(tables / "summary.csv", header=["value"])
    s.ts_sma_band.drop(columns="bin").to_csv(tables / "ts_9d_30_band_by_sma_state.csv", index=False)
    s.regime_states.to_csv(tables / "regime_state_books.csv")
    s.tranches.to_csv(tables / "tranches_in_regime.csv")
    for slip, scan in s.stop_scans.items():
        scan.to_csv(tables / f"call_stop_scan_slippage_{slip:.0f}.csv")
    s.tranche_stops.to_csv(tables / "call_stop_by_tranche.csv")
    s.importance.column.to_csv(tables / "feature_importance_columns.csv")
    s.importance.family.to_csv(tables / "feature_importance_families.csv")
    s.quintiles.to_csv(tables / f"feature_quintiles_call_stop_{RECOMMENDED_CALL_STOP}.csv", index=False)
    s.rules.to_csv(tables / "filter_threshold_rules.csv")
    s.rules_stopped.to_csv(tables / f"filter_threshold_rules_call_stop_{RECOMMENDED_CALL_STOP}.csv")
    s.cap_comparison.to_csv(tables / "cap_comparison_102_vs_104.csv")
    s.rules_years.to_csv(tables / "top_features_by_year.csv")
    s.weekdays.to_csv(tables / "weekday_session_level.csv")
    s.weekday_grid[0].to_csv(tables / "weekday_by_tranche_avg_pl.csv")

    _save(fig_ts_sma_band(s.ts_sma_band), figures / "ts_9d_30_band_by_sma_state.png")
    _save(fig_regime_equity(s.panel, s.stopped_pl, s.regime), figures / "regime_equity.png")
    _save(fig_tranches(s.tranches), figures / "tranches_in_regime.png")
    _save(fig_stop(s.stop_scans), figures / "call_stop_scan.png")
    _save(fig_weekdays(s.weekdays), figures / "weekdays.png")
    _save(fig_importance(s.importance.column), figures / "feature_importance_columns.png")
    _save(fig_rules(s.rules_stopped), figures / "filter_threshold_rules.png")
    _save(fig_quintiles(s.quintiles, QUINTILE_FEATURES), figures / "feature_quintiles.png")
    _save(fig_weekday_tranche(*s.weekday_grid), figures / "weekday_by_tranche.png")
    s.summary["output_dir"] = str(out)
    return s
