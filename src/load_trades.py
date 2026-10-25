from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
import glob
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .legs import (
    add_actual_dte_columns,
    mark_dte_conformance,
    summarize_dte_conformance,
)


# Every dollar figure below is a position-level number: the backtest compounds
# size as the account grows, so one study mixes trades sized at 1 contract with
# trades sized at 1,400. Summing or averaging them across a study measures the
# sizing schedule, not the strategy. Each of these is divided by
# ``no_of_contracts`` at load time and its position-level value is preserved
# under a ``_position`` suffix for audit.
PER_CONTRACT_COLUMNS = (
    "p_l",
    "margin_req",
    "opening_commissions_fees",
    "closing_commissions_fees",
)

# Requested-vs-filled DTE handling. ``exclude`` keeps only trades whose legs
# landed on the DTE pair the filename asked for; ``flag`` keeps everything and
# leaves ``dte_on_spec`` for the caller to filter on.
DTE_POLICY_EXCLUDE = "exclude"
DTE_POLICY_FLAG = "flag"
DEFAULT_DTE_POLICY = DTE_POLICY_EXCLUDE
DEFAULT_DTE_TOLERANCE = 0

REQUIRED_COLUMNS = {
    "date_opened",
    "date_closed",
    "time_opened",
    "p_l",
    "p_l_pct",
    "opening_short_long_ratio",
    "closing_short_long_ratio",
    "opening_vix",
    "closing_vix",
    "margin_req",
    "premium",
}

NUMERIC_COLUMNS = [
    "opening_price",
    "premium",
    "closing_price",
    "avg_closing_cost",
    "p_l",
    "p_l_pct",
    "no_of_contracts",
    "funds_at_close",
    "margin_req",
    "opening_commissions_fees",
    "closing_commissions_fees",
    "opening_short_long_ratio",
    "closing_short_long_ratio",
    "opening_vix",
    "closing_vix",
    "gap",
    "movement",
    "max_profit",
    "max_loss",
]


@dataclass(frozen=True)
class StrategyMetadata:
    """Strategy metadata encoded in an Options Omega export filename."""

    short_dte: int
    long_dte: int
    strategy: str
    short_put_delta: int
    short_call_delta: int

    @property
    def dte_pair(self) -> str:
        return f"{self.short_dte}-{self.long_dte}"

    @property
    def delta_pair(self) -> str:
        return f"P{self.short_put_delta}/C{self.short_call_delta}"


def normalize_column_name(name: str) -> str:
    """Convert Options Omega headers into snake_case names."""
    cleaned = str(name).strip().lower()
    cleaned = cleaned.replace("%", " pct")
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def parse_strategy_metadata_from_filename(path: Path) -> StrategyMetadata:
    """Parse DTE, strategy type, and put/call deltas from a source filename.

    Supported filename stems include the paired-delta format used by this
    study, for example ``14-28-DC-D10-35``, and the older one-delta format
    ``14-28-DC-D20``. Older files are treated as symmetric put/call deltas.
    """
    match = re.fullmatch(
        r"(?P<short_dte>\d+)-(?P<long_dte>\d+)-(?P<strategy>[A-Za-z]+)"
        r"-D(?P<short_put_delta>\d+)(?:-(?P<short_call_delta>\d+))?",
        path.stem,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(
            "Could not parse strategy metadata from filename: "
            f"{path.name}. Expected a stem like 14-28-DC-D10-10."
        )

    short_put_delta = int(match.group("short_put_delta"))
    short_call_delta_text = match.group("short_call_delta")
    short_call_delta = (
        int(short_call_delta_text) if short_call_delta_text is not None else short_put_delta
    )
    return StrategyMetadata(
        short_dte=int(match.group("short_dte")),
        long_dte=int(match.group("long_dte")),
        strategy=match.group("strategy").upper(),
        short_put_delta=short_put_delta,
        short_call_delta=short_call_delta,
    )


def extract_delta_from_filename(path: Path) -> int:
    """Extract the short-put delta value encoded in a source filename.

    Args:
        path: CSV path whose filename contains a D-prefixed put delta.

    Returns:
        The parsed short-put delta value.
    """
    return parse_strategy_metadata_from_filename(path).short_put_delta


def _clean_numeric_value(value):
    """Parse numbers that may include $, %, commas, parentheses, or extra text."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    is_parenthetical_negative = text.startswith("(") and text.endswith(")")
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("%", "")
    text = text.replace("(", "")
    text = text.replace(")", "")

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return np.nan

    number = float(match.group(0))
    if is_parenthetical_negative and number > 0:
        number = -number
    return number


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Convert messy numeric values in a series to floats.

    Args:
        series: Values to clean and convert.

    Returns:
        A float series with invalid values set to NaN.
    """
    return series.apply(_clean_numeric_value).astype(float)


def _parse_time_label(value):
    """Return a stable HH:MM label for common Options Omega time formats."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if not text:
        return np.nan

    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p"):
        try:
            return datetime.strptime(text, fmt).strftime("%H:%M")
        except ValueError:
            pass

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return np.nan
    return parsed.strftime("%H:%M")


def _validate_required_columns(df: pd.DataFrame, path: Path) -> None:
    """Validate that a loaded trade file has the required columns.

    Args:
        df: Normalized trade data to check.
        path: Source CSV path used in the error message.
    """
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        available = ", ".join(df.columns)
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Missing required columns in {path.name}: {missing_text}. "
            f"Available normalized columns: {available}"
        )


def load_trade_file(
    path: Path,
    normalize_contracts: bool = True,
    dte_policy: str = DEFAULT_DTE_POLICY,
    dte_tolerance: int = DEFAULT_DTE_TOLERANCE,
) -> pd.DataFrame:
    """Load one Options Omega CSV and apply stable column parsing.

    Args:
        path: Strategy CSV to load.
        normalize_contracts: Rescale position-level dollars to one contract.
        dte_policy: ``"exclude"`` to keep only trades whose legs match the
            filename's DTE pair, or ``"flag"`` to keep every trade and rely on
            the ``dte_on_spec`` column.
        dte_tolerance: Allowed per-leg deviation, in days, when judging
            conformance.

    Returns:
        One normalized trade frame, with the conformance breakdown for the file
        available under ``df.attrs["dte_conformance"]``.
    """
    df = pd.read_csv(path)
    df.columns = [normalize_column_name(col) for col in df.columns]
    _validate_required_columns(df, path)

    # Add traceability fields before combining multiple delta files.
    metadata = parse_strategy_metadata_from_filename(path)
    df["source_file"] = path.name
    df["short_dte"] = metadata.short_dte
    df["long_dte"] = metadata.long_dte
    df["dte_pair"] = metadata.dte_pair
    df["strategy"] = metadata.strategy
    df["short_put_delta"] = metadata.short_put_delta
    df["short_call_delta"] = metadata.short_call_delta
    df["delta_pair"] = metadata.delta_pair
    # Backward-compatible alias for older one-dimensional reports.
    df["delta"] = metadata.short_put_delta

    for column in ("date_opened", "date_closed"):
        df[column] = pd.to_datetime(df[column], errors="coerce").dt.date

    for column in ("time_opened", "time_closed"):
        if column in df.columns:
            df[column] = df[column].apply(_parse_time_label)
            df[f"{column}_label"] = df[column]

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = clean_numeric_series(df[column])

    if normalize_contracts:
        df = normalize_to_one_contract(df)

    df = mark_dte_conformance(
        add_actual_dte_columns(df),
        short_dte=metadata.short_dte,
        long_dte=metadata.long_dte,
        tolerance=dte_tolerance,
    )
    conformance = summarize_dte_conformance(df)
    if dte_policy == DTE_POLICY_EXCLUDE:
        df = df.loc[df["dte_on_spec"]].reset_index(drop=True)
    elif dte_policy != DTE_POLICY_FLAG:
        raise ValueError(
            f"Unknown dte_policy {dte_policy!r}; expected "
            f"{DTE_POLICY_EXCLUDE!r} or {DTE_POLICY_FLAG!r}."
        )

    # Stored as records rather than a frame: pandas compares ``attrs`` values
    # with ``==`` whenever frames are concatenated, and a DataFrame there makes
    # every downstream concat raise.
    df.attrs["dte_conformance"] = conformance.to_dict("records")
    df.attrs["dte_policy"] = dte_policy
    return df


def normalize_to_one_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale position-level dollar columns to a single contract.

    Options Omega reports P/L, margin and fees for the whole position, and the
    backtest grows that position as the account compounds. Dividing by
    ``no_of_contracts`` makes a 2019 trade and a 2026 trade directly
    comparable; the original values stay available as ``<column>_position``.

    Trades with a missing or non-positive contract count cannot be rescaled, so
    their per-contract values are left null and drop out of every summary
    rather than being silently counted at position size.

    Args:
        df: Trades with cleaned numeric columns.

    Returns:
        The frame with per-contract dollar columns and ``_position`` originals.
    """
    normalized = df.copy()
    if "no_of_contracts" not in normalized.columns:
        normalized["contracts_usable"] = False
        return normalized

    contracts = pd.to_numeric(normalized["no_of_contracts"], errors="coerce")
    usable = contracts.notna() & (contracts > 0)
    divisor = contracts.where(usable)

    for column in PER_CONTRACT_COLUMNS:
        if column not in normalized.columns:
            continue
        normalized[f"{column}_position"] = normalized[column]
        normalized[column] = normalized[column] / divisor

    normalized["contracts_usable"] = usable
    return normalized


def load_trade_paths(
    paths: Iterable[str | Path],
    normalize_contracts: bool = True,
    dte_policy: str = DEFAULT_DTE_POLICY,
    dte_tolerance: int = DEFAULT_DTE_TOLERANCE,
) -> pd.DataFrame:
    """Load and combine an explicit list of strategy CSV files.

    Args:
        paths: Strategy CSV files to load.
        normalize_contracts: Rescale position-level dollars to one contract.
        dte_policy: How to treat trades whose filled DTE differs from the pair
            requested by the filename.
        dte_tolerance: Allowed per-leg deviation, in days.

    Returns:
        One combined normalized trade DataFrame, carrying the pooled DTE
        conformance breakdown under ``df.attrs["dte_conformance"]``.
    """
    resolved_paths = tuple(sorted(Path(path) for path in paths))
    if not resolved_paths:
        raise FileNotFoundError("No strategy CSV files were provided.")

    frames = [
        load_trade_file(
            path,
            normalize_contracts=normalize_contracts,
            dte_policy=dte_policy,
            dte_tolerance=dte_tolerance,
        )
        for path in resolved_paths
    ]
    conformance: list[dict] = []
    for frame in frames:
        conformance.extend(frame.attrs.get("dte_conformance", []))

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty and dte_policy == DTE_POLICY_EXCLUDE:
        raise ValueError(
            "Every trade was excluded because its filled expiries did not match "
            "the DTE pair in its filename. Widen --dte-tolerance or pass "
            "--dte-policy flag to keep off-spec trades."
        )
    combined.attrs["dte_conformance"] = conformance
    combined.attrs["dte_policy"] = dte_policy
    return combined


def load_trade_files(file_pattern: str | Path) -> pd.DataFrame:
    """Load and combine all CSVs matching the requested pattern."""
    file_pattern = Path(file_pattern)
    glob_pattern = file_pattern / "*.csv" if file_pattern.is_dir() else file_pattern
    paths = sorted(Path(path) for path in glob.glob(str(glob_pattern)))
    if not paths:
        search_dir = glob_pattern.parent
        nearby_files = sorted(search_dir.glob("*.csv")) if search_dir.exists() else []
        nearby_text = (
            " Nearby CSV files: "
            + ", ".join(path.name for path in nearby_files[:10])
            + ("..." if len(nearby_files) > 10 else "")
            if nearby_files
            else ""
        )
        raise FileNotFoundError(
            f"No CSV files found for pattern: {glob_pattern}."
            f"{nearby_text} Place the Options Omega files in data/raw/ "
            "or pass --data-dir / --data-pattern to run a scenario folder."
        )

    return load_trade_paths(paths)


def dte_conformance_frame(trades: pd.DataFrame) -> pd.DataFrame:
    """Return the DTE conformance breakdown carried by a loaded trade frame.

    Args:
        trades: A frame produced by :func:`load_trade_file` or
            :func:`load_trade_paths`.

    Returns:
        One row per source file and filled DTE pair, or an empty frame when the
        trades were not loaded through this module.
    """
    return pd.DataFrame(trades.attrs.get("dte_conformance", []))
