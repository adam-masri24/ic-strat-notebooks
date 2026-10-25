"""Load the VIX-open decile edges that define this project's VIX regimes.

The regimes used to be six hand-picked bands (0-9, 9-14, ... 25+). They are now
the deciles of VIX open measured over a reference window, stored in
``config/vix_open_deciles.json`` so the cut points are data rather than code and
can be regenerated without touching the pipeline.

Two adjustments are made to the file's edges when they become bin edges:

* The outermost edges are replaced with -inf and +inf. The reference window is
  narrower than the trade history — a study opening at VIX 11.74 or 78.63 sits
  outside a file fitted on 11.81-55.68 — and a trade outside the range must land
  in the first or last regime rather than dropping out of every VIX table.
* The bins are right-closed, matching ``upper_inclusive`` in the file, so a VIX
  open of exactly 13.467 belongs to D1 and not D2.

Because the reference window is shorter than the trade history, the buckets are
decile-shaped for that window, not exact deciles of any given study's trades.
The tail regimes in particular hold more than a tenth of the trades whenever a
study reaches back into a higher-volatility period.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "vix_open_deciles.json"


@dataclass(frozen=True)
class VixDecileScheme:
    """Bin edges and labels for the VIX-open decile regimes."""

    bins: list[float]
    labels: list[str]
    source_path: Path
    reference_start: str
    reference_end: str
    reference_sessions: int
    observed_min: float
    observed_max: float

    @property
    def reference_window(self) -> str:
        """Human-readable description of the window the deciles were fitted on."""
        return (
            f"{self.reference_start} to {self.reference_end} "
            f"({self.reference_sessions:,} sessions, "
            f"VIX open {self.observed_min:.2f}-{self.observed_max:.2f})"
        )


def _format_edge(value: float) -> str:
    """Render a cut point compactly for a bucket label."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _build_labels(interior_edges: list[float], count: int) -> list[str]:
    """Label each decile with its VIX range, so an axis tick is self-describing.

    Both the S/L deciles and these VIX regimes are numbered D1-D10, and the VIX
    grid puts them on adjacent axes, so a bare ``D4`` would be ambiguous exactly
    where it matters most.
    """
    labels = []
    for index in range(count):
        if index == 0:
            labels.append(f"D1 ≤{_format_edge(interior_edges[0])}")
        elif index == count - 1:
            labels.append(f"D{count} >{_format_edge(interior_edges[-1])}")
        else:
            lower = _format_edge(interior_edges[index - 1])
            upper = _format_edge(interior_edges[index])
            labels.append(f"D{index + 1} {lower}-{upper}")
    return labels


@lru_cache(maxsize=1)
def load_vix_decile_scheme(config_path: Path | None = None) -> VixDecileScheme:
    """Read the decile configuration and turn it into bins and labels.

    Args:
        config_path: Optional override for the JSON configuration file.

    Returns:
        The bin edges and labels used for every VIX regime in the project.

    Raises:
        FileNotFoundError: If the configuration file is missing.
        ValueError: If the file has fewer than two breakpoints or its
            breakpoints are not strictly increasing.
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"VIX decile configuration not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    breakpoints = [float(value) for value in payload.get("breakpoints", [])]
    if len(breakpoints) < 2:
        raise ValueError(f"{path} needs at least two breakpoints to form one bucket.")
    if any(later <= earlier for earlier, later in zip(breakpoints, breakpoints[1:])):
        raise ValueError(f"{path} breakpoints must be strictly increasing.")

    interior_edges = breakpoints[1:-1]
    bucket_count = len(breakpoints) - 1
    return VixDecileScheme(
        # Open-ended tails: the trade history runs outside the reference window.
        bins=[-np.inf, *interior_edges, np.inf],
        labels=_build_labels(interior_edges, bucket_count),
        source_path=path,
        reference_start=str(payload.get("date_start", "unknown")),
        reference_end=str(payload.get("date_end", "unknown")),
        reference_sessions=int(payload.get("n_sessions", 0)),
        observed_min=float(payload.get("vix_open_min", breakpoints[0])),
        observed_max=float(payload.get("vix_open_max", breakpoints[-1])),
    )
