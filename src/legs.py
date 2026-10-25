"""Parse the Options Omega ``Legs`` string into per-trade expiry facts.

A double calendar's DTE pair only exists in the leg descriptions: the short
legs (``STO``) carry the near expiry and the long legs (``BTO``) carry the far
one. The filename says what the strategy *asked* for; these functions say what
the backtester actually filled, which is not always the same — when the
requested expiry is not listed, the engine rolls to the nearest one that is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


_MONTHS = {
    month: index + 1
    for index, month in enumerate(
        [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec",
        ]
    )
}

# "1222 Jul 15 7465 P STO 17.90" — quantity, expiry, strike, right, side, price.
_LEG_PATTERN = re.compile(
    r"(?P<quantity>\d+)\s+"
    r"(?P<month>[A-Za-z]{3})\s+(?P<day>\d{1,2})(?:\s+(?P<year>\d{4}))?\s+"
    r"(?P<strike>[\d.]+)\s+"
    r"(?P<right>[PC])\s+"
    r"(?P<side>STO|BTO|STC|BTC)",
    flags=re.IGNORECASE,
)

# An expiry more than this many days before the open date is read as next year;
# option chains never list an expiry in the past, so a "backwards" date is
# always a year that the export omitted.
_YEAR_ROLLOVER_DAYS = 180

OPENING_SIDES = ("STO", "BTO")


@dataclass(frozen=True)
class LegExpiries:
    """Distinct opening expiries found in one trade's leg description."""

    short_expiry: pd.Timestamp | None
    long_expiry: pd.Timestamp | None
    short_expiry_count: int
    long_expiry_count: int

    @property
    def is_well_formed(self) -> bool:
        """Whether the trade has exactly one short and one long expiry."""
        return self.short_expiry_count == 1 and self.long_expiry_count == 1


def _resolve_expiry(
    month: str,
    day: int,
    year: int | None,
    date_opened: pd.Timestamp,
) -> pd.Timestamp | None:
    """Build an expiry timestamp, inferring the year when the export omits it."""
    month_number = _MONTHS.get(month.lower())
    if month_number is None:
        return None

    for candidate_year in ([year] if year else [date_opened.year, date_opened.year + 1]):
        try:
            expiry = pd.Timestamp(candidate_year, month_number, day)
        except ValueError:
            continue
        if year or (expiry - date_opened).days >= -_YEAR_ROLLOVER_DAYS:
            return expiry
    return None


def parse_leg_expiries(legs: object, date_opened: object) -> LegExpiries:
    """Extract the opening short and long expiries from one ``Legs`` value.

    Args:
        legs: Raw ``Legs`` cell from an Options Omega export.
        date_opened: The trade's open date, used to infer omitted years.

    Returns:
        The distinct short and long expiries, with counts so callers can spot
        trades whose put and call legs did not land on the same expiry.
    """
    opened = pd.Timestamp(date_opened) if not pd.isna(date_opened) else None
    if opened is None or pd.isna(legs):
        return LegExpiries(None, None, 0, 0)

    by_side: dict[str, set[pd.Timestamp]] = {side: set() for side in OPENING_SIDES}
    for match in _LEG_PATTERN.finditer(str(legs)):
        side = match.group("side").upper()
        if side not in by_side:
            continue
        expiry = _resolve_expiry(
            match.group("month"),
            int(match.group("day")),
            int(match.group("year")) if match.group("year") else None,
            opened,
        )
        if expiry is not None:
            by_side[side].add(expiry)

    short_expiries, long_expiries = by_side["STO"], by_side["BTO"]
    return LegExpiries(
        short_expiry=min(short_expiries) if short_expiries else None,
        long_expiry=min(long_expiries) if long_expiries else None,
        short_expiry_count=len(short_expiries),
        long_expiry_count=len(long_expiries),
    )


def add_actual_dte_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the DTE actually filled, alongside the DTE the filename requested.

    Adds ``actual_short_dte``, ``actual_long_dte``, ``actual_dte_spread`` and
    ``dte_legs_parsed``. Rows whose ``Legs`` value could not be read leave the
    DTE columns null and ``dte_legs_parsed`` False, so a missing or unusual
    export is visible as unknown rather than silently scored as conforming.
    """
    enriched = df.copy()
    if "legs" not in enriched.columns or "date_opened" not in enriched.columns:
        enriched["actual_short_dte"] = np.nan
        enriched["actual_long_dte"] = np.nan
        enriched["actual_dte_spread"] = np.nan
        enriched["dte_legs_parsed"] = False
        return enriched

    opened = pd.to_datetime(enriched["date_opened"], errors="coerce")
    parsed = [
        parse_leg_expiries(legs, open_date)
        for legs, open_date in zip(enriched["legs"], opened)
    ]

    short_dte, long_dte, well_formed = [], [], []
    for expiries, open_date in zip(parsed, opened):
        usable = expiries.is_well_formed and not pd.isna(open_date)
        well_formed.append(bool(usable))
        short_dte.append((expiries.short_expiry - open_date).days if usable else np.nan)
        long_dte.append((expiries.long_expiry - open_date).days if usable else np.nan)

    enriched["actual_short_dte"] = pd.Series(short_dte, index=enriched.index, dtype="float")
    enriched["actual_long_dte"] = pd.Series(long_dte, index=enriched.index, dtype="float")
    enriched["actual_dte_spread"] = enriched["actual_long_dte"] - enriched["actual_short_dte"]
    enriched["dte_legs_parsed"] = pd.Series(well_formed, index=enriched.index, dtype="bool")
    return enriched


def mark_dte_conformance(
    df: pd.DataFrame,
    short_dte: int,
    long_dte: int,
    tolerance: int = 0,
) -> pd.DataFrame:
    """Flag whether each trade's filled expiries match the requested DTE pair.

    A trade is on-spec when both legs land within ``tolerance`` days of the
    requested DTE *and* the long-minus-short spread matches exactly. The spread
    is checked separately because a calendar's edge comes from the gap between
    the two expiries: 21/37 is not a wider version of 21/35, it is a different
    trade.

    Args:
        df: Trades carrying the ``actual_*`` columns.
        short_dte: Requested short-leg DTE from the filename.
        long_dte: Requested long-leg DTE from the filename.
        tolerance: Allowed absolute deviation per leg, in days.

    Returns:
        The frame with ``requested_dte_spread`` and ``dte_on_spec`` added.
    """
    enriched = df.copy()
    if "actual_short_dte" not in enriched.columns:
        enriched = add_actual_dte_columns(enriched)

    requested_spread = long_dte - short_dte
    enriched["requested_dte_spread"] = requested_spread

    short_ok = (enriched["actual_short_dte"] - short_dte).abs() <= tolerance
    long_ok = (enriched["actual_long_dte"] - long_dte).abs() <= tolerance
    spread_ok = enriched["actual_dte_spread"] == requested_spread

    enriched["dte_on_spec"] = (
        enriched["dte_legs_parsed"].fillna(False)
        & short_ok.fillna(False)
        & long_ok.fillna(False)
        & spread_ok.fillna(False)
    )
    return enriched


def summarize_dte_conformance(df: pd.DataFrame) -> pd.DataFrame:
    """Count on-spec and off-spec trades per source file and filled DTE pair."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "source_file", "delta_pair", "requested_dte_pair",
                "actual_dte_pair", "actual_dte_spread", "dte_on_spec", "trade_count",
            ]
        )

    working = df.copy()
    working["requested_dte_pair"] = working["dte_pair"]
    working["actual_dte_pair"] = [
        "unparsed" if not parsed else f"{int(short)}-{int(long)}"
        for parsed, short, long in zip(
            working["dte_legs_parsed"].fillna(False),
            working["actual_short_dte"],
            working["actual_long_dte"],
        )
    ]
    group_cols = [
        "source_file",
        "delta_pair",
        "requested_dte_pair",
        "actual_dte_pair",
        "actual_dte_spread",
        "dte_on_spec",
    ]
    return (
        working.groupby(group_cols, observed=True, dropna=False)
        .size()
        .reset_index(name="trade_count")
        .sort_values(["source_file", "dte_on_spec", "trade_count"], ascending=[True, False, False])
        .reset_index(drop=True)
    )


def parse_short_strikes(legs: object) -> tuple[float | None, float | None]:
    """Return the short put and short call strikes from one ``Legs`` value.

    A double calendar sells the near expiry and buys the far one at the same two
    strikes, so the sold (``STO``) legs carry the strikes that bracket the profit
    tent: the put strike below spot and the call strike above it. Those are the
    reference points a close is measured against.

    Args:
        legs: Raw ``Legs`` cell from an Options Omega export.

    Returns:
        ``(short_put_strike, short_call_strike)``, each ``None`` when the
        corresponding sold leg is absent or unparseable.
    """
    if pd.isna(legs):
        return None, None

    put_strike: float | None = None
    call_strike: float | None = None
    for match in _LEG_PATTERN.finditer(str(legs)):
        if match.group("side").upper() != "STO":
            continue
        try:
            strike = float(match.group("strike"))
        except (TypeError, ValueError):
            continue
        if match.group("right").upper() == "P" and put_strike is None:
            put_strike = strike
        elif match.group("right").upper() == "C" and call_strike is None:
            call_strike = strike
    return put_strike, call_strike


def add_short_strike_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``short_put_strike`` and ``short_call_strike`` parsed from ``legs``.

    Rows without a readable ``legs`` value leave the strike columns null rather
    than guessing, so a missing or unusual export drops out of a strike-distance
    view instead of landing at a fabricated distance.
    """
    enriched = df.copy()
    if "legs" not in enriched.columns:
        enriched["short_put_strike"] = np.nan
        enriched["short_call_strike"] = np.nan
        return enriched

    parsed = [parse_short_strikes(legs) for legs in enriched["legs"]]
    enriched["short_put_strike"] = pd.Series(
        [put for put, _ in parsed], index=enriched.index, dtype="float"
    )
    enriched["short_call_strike"] = pd.Series(
        [call for _, call in parsed], index=enriched.index, dtype="float"
    )
    return enriched
