"""Every well-known location in the repo, resolved in one place.

Modules used to derive these by counting ``parents[...]`` from their own file,
which broke the moment a file moved. Import from here instead:

    from src.paths import VIX_DAILY_FILE, strategy_data_root

Layout::

    data/
      strategy/<name>/<UNDERLYING>/raw_<short>_<long>/*.csv   per-strategy trade logs
      market/vix/vix_open_daily.csv                           shared market reference
      market/vix/term_structure_daily.csv                     VIX1D/VIX9D/VIX30 per session
      market/vvix/vvix_daily_cboe.csv                         VVIX daily closes
    reports/<strategy>/                                       generated output
    config/                                                   shared configuration
"""

from __future__ import annotations

from pathlib import Path

# src/paths.py -> src/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_ROOT = PROJECT_ROOT / "config"
DATA_ROOT = PROJECT_ROOT / "data"
REPORTS_ROOT = PROJECT_ROOT / "reports"
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"

# Trade logs, partitioned by the strategy that produced them.
STRATEGY_DATA_ROOT = DATA_ROOT / "strategy"

# Market reference series that belong to no single strategy.
MARKET_DATA_ROOT = DATA_ROOT / "market"
VIX_DATA_ROOT = MARKET_DATA_ROOT / "vix"
VIX_DAILY_FILE = VIX_DATA_ROOT / "vix_open_daily.csv"
# Session-level VIX1D/VIX9D/VIX30 curve (scripts/vix_term_structure_daily.py writes this name).
VIX_TERM_STRUCTURE_DAILY_FILE = VIX_DATA_ROOT / "term_structure_daily.csv"
VVIX_DATA_ROOT = MARKET_DATA_ROOT / "vvix"
# Cboe's public VVIX daily history (closes only).
VVIX_DAILY_FILE = VVIX_DATA_ROOT / "vvix_daily_cboe.csv"
SPX_DATA_ROOT = MARKET_DATA_ROOT / "spx"
SPX_DAILY_RETURNS_FILE = SPX_DATA_ROOT / "spx_daily_returns.csv"

ARCHIVE_ROOT = PROJECT_ROOT / "data_archives"


def strategy_data_root(strategy: str) -> Path:
    """Return the trade-log root for one strategy, e.g. ``data/strategy/dc``."""
    return STRATEGY_DATA_ROOT / strategy


def strategy_reports_root(strategy: str) -> Path:
    """Return the report output root for one strategy, e.g. ``reports/dc``."""
    return REPORTS_ROOT / strategy
