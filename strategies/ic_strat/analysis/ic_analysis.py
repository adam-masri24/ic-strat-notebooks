"""Load the IC 20-tranche 10-spread baseline and reduce it to one row per condor.

The raw Options Omega export has one row per **vertical spread** (call side or
put side); two rows sharing a ``(date, entry time)`` key make one iron condor.
This module does the pairing and adds the entry-time-only features every later
analysis in this strategy builds on. Nothing here reaches past the moment of
entry — see :func:`build_iron_condors` for the leakage boundary.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.load_trades import clean_numeric_series, normalize_column_name
from src.paths import strategy_data_root

STRATEGY_NAME = "ic_strat"
BASELINE_FILENAME = "IC-20-Tranche-10-Baseline.csv"
# Trade logs live in data/ (git-ignored, shared through the Drive archives), not in the repo.
DEFAULT_DATA_DIR = strategy_data_root(STRATEGY_NAME) / "spx" / "raw_baseline"
DEFAULT_CSV = DEFAULT_DATA_DIR / BASELINE_FILENAME


def require_file(path: Path, what: str) -> Path:
    """Return ``path``, or raise with the fix when a data file isn't on disk (data/ is not in git)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}. data/ isn't in git: download the archives from the shared "
            "Drive folder into data_archives/ and run `python scripts/unpack_data.py` (see README, 'Data')."
        )
    return path

_STRIKE_PATTERN = re.compile(
    r"(?P<qty>\d+)\s+[A-Za-z]{3}\s+\d{1,2}\s+(?P<strike>[\d.]+)\s+(?P<right>[PC])\s+(?P<side>STO|BTO)",
    flags=re.IGNORECASE,
)

SESSION_OPEN_MINUTES = 9 * 60 + 30  # 09:30 ET, minutes from midnight


def _parse_leg_geometry(legs: str) -> tuple[str | None, float | None, float | None]:
    """Return (option_right, short_strike, long_strike) for one vertical spread.

    Each row is a single 10-wide vertical (one STO, one BTO, same right), so
    there is exactly one short and one long strike to recover.
    """
    if pd.isna(legs):
        return None, None, None

    right = None
    short_strike = None
    long_strike = None
    for match in _STRIKE_PATTERN.finditer(str(legs)):
        strike = float(match.group("strike"))
        side = match.group("side").upper()
        right = match.group("right").upper()
        if side == "STO":
            short_strike = strike
        elif side == "BTO":
            long_strike = strike
    return right, short_strike, long_strike


def load_ic_trades(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """Load the raw per-spread export and normalize columns to snake_case.

    Args:
        path: CSV export with one row per vertical spread.

    Returns:
        One row per spread, with ``side`` (call/put), ``short_strike``,
        ``long_strike`` and ``wing_width`` parsed from ``legs``.
    """
    df = pd.read_csv(require_file(Path(path), "IC trade log"), encoding="utf-8-sig")
    df.columns = [normalize_column_name(col) for col in df.columns]

    numeric_cols = [
        "opening_price", "premium", "closing_price", "days_in_trade",
        "avg_closing_cost", "p_l", "p_l_pct", "no_of_contracts", "funds_at_close",
        "margin_req", "opening_commissions_and_fees", "opening_short_long_ratio",
        "closing_short_long_ratio", "gap", "movement", "max_profit", "max_loss",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric_series(df[col])

    df["date_opened"] = pd.to_datetime(df["date_opened"], errors="coerce")
    df["date_closed"] = pd.to_datetime(df["date_closed"], errors="coerce")

    geometry = [_parse_leg_geometry(legs) for legs in df["legs"]]
    df["right"] = [g[0] for g in geometry]
    df["short_strike"] = pd.Series([g[1] for g in geometry], dtype="float")
    df["long_strike"] = pd.Series([g[2] for g in geometry], dtype="float")
    df["side"] = df["right"].map({"C": "call", "P": "put"})
    df["wing_width"] = (df["long_strike"] - df["short_strike"]).abs()
    # Distance of the short strike from spot, signed so "further OTM" is positive
    # on both sides: calls sell above spot, puts sell below it.
    df["otm_points"] = np.where(
        df["side"] == "call",
        df["short_strike"] - df["opening_price"],
        df["opening_price"] - df["short_strike"],
    )

    return df


def _entry_minutes(time_opened: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(time_opened, format="%H:%M:%S", errors="coerce")
    return parsed.dt.hour * 60 + parsed.dt.minute


def build_iron_condors(spreads: pd.DataFrame) -> pd.DataFrame:
    """Pair call and put verticals sharing (date, entry time) into one condor.

    Rows that do not have exactly one call and one put leg at their
    ``(date_opened, time_opened)`` key are dropped — an unmatched spread is not
    a condor and would otherwise silently double- or under-count P/L.

    Returns:
        One row per iron condor: combined P/L, combined credit, per-side
        credit/short-long-ratio/OTM-distance (so skew is visible), spot at
        entry, margin, gap, and the count of legs that formed the row.
    """
    working = spreads.copy()
    key = ["date_opened", "time_opened"]
    leg_counts = working.groupby(key)["side"].transform("count")
    sides_per_key = working.groupby(key)["side"].transform("nunique")
    well_formed = (leg_counts == 2) & (sides_per_key == 2)

    dropped = int((~well_formed).sum())
    working = working.loc[well_formed].copy()

    agg = working.groupby(key, as_index=False).agg(
        pl=("p_l", "sum"),
        credit=("premium", "sum"),
        spot=("opening_price", "first"),
        margin=("margin_req", "first"),
        gap=("gap", "first"),
        movement=("movement", "first"),
        underlying=("underlying", "first"),
    )

    pivoted = working.pivot_table(
        index=key,
        columns="side",
        values=["premium", "opening_short_long_ratio", "closing_short_long_ratio", "otm_points"],
        aggfunc="first",
    )
    pivoted.columns = [f"{value}_{side}" for value, side in pivoted.columns]
    pivoted = pivoted.reset_index().rename(
        columns={
            "premium_call": "credit_call",
            "premium_put": "credit_put",
            "opening_short_long_ratio_call": "slratio_call",
            "opening_short_long_ratio_put": "slratio_put",
            # Known only once the trade closes (expiry or stop-loss) — an
            # exit-side signal, never an entry feature. Keep it out of
            # CURATED_FEATS/cascading.FEATURE_GROUPS for that reason.
            "closing_short_long_ratio_call": "close_slratio_call",
            "closing_short_long_ratio_put": "close_slratio_put",
        }
    )

    ic = agg.merge(pivoted, on=key, how="left")
    ic["skew_credit"] = ic["credit_put"] - ic["credit_call"]
    ic["otm_asym"] = ic["otm_points_put"] - ic["otm_points_call"]
    ic["win"] = (ic["pl"] > 0).astype(int)

    ic = ic.rename(columns={"date_opened": "date", "time_opened": "entry_time"})
    ic["date"] = pd.to_datetime(ic["date"])
    ic["mins_from_open"] = _entry_minutes(ic["entry_time"]) - SESSION_OPEN_MINUTES
    ic["dow"] = ic["date"].dt.day_name()
    ic.attrs["dropped_unmatched_legs"] = dropped
    return ic.sort_values(["date", "mins_from_open"]).reset_index(drop=True)


def daily_close_series(ic: pd.DataFrame) -> pd.DataFrame:
    """Return one row per session: the last tranche's SPX spot as that day's close.

    Not a real closing print (the last tranche is 15:53, not 16:00), but close
    enough for a day-level price moving average — the gap is under a minute
    of a 0DTE index that's about to settle.
    """
    return (
        ic.sort_values(["date", "mins_from_open"])
        .groupby("date")["spot"].last()
        .reset_index()
        .rename(columns={"spot": "close"})
    )


def add_intraday_path_features(ic: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative-within-session SPX path features, computed causally.

    Every value at tranche *t* uses only tranches at or before *t* on the same
    day, so a 10:33 entry never sees the 15:53 close. This is what "known at
    entry" means for the SPX path, mirroring the DTE/leg discipline used
    elsewhere in this repo (src/legs.py) for what a trade could actually see.
    """
    enriched = ic.copy()
    parts = []
    for _, grp in enriched.groupby("date", sort=True):
        grp = grp.sort_values("mins_from_open").copy()
        spot_open = grp["spot"].iloc[0]
        running_max = grp["spot"].cummax()
        running_min = grp["spot"].cummin()
        grp["range_so_far"] = (running_max - running_min) / spot_open * 100
        grp["dd_from_hi"] = (grp["spot"] / running_max - 1) * 100
        grp["ret_last_tranche"] = grp["spot"].pct_change().fillna(0) * 100
        parts.append(grp)
    out = pd.concat(parts, ignore_index=True)
    # SPX move since the 09:30 open, as Option Omega measures it: the export's
    # `Movement` is entry spot minus the session open, in points. The first
    # tranche is 09:32, so this is not zero there the way a move measured from
    # the first tranche would be.
    out["spx_move_pct"] = out["movement"] / (out["spot"] - out["movement"]) * 100
    return out
