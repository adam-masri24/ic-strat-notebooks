"""Partial correlation among a curated feature set.

A plain (zero-order) correlation between two features can be almost entirely
explained by a third they both move with — this strategy's own term-structure
features are a textbook case, since VIX1D/VIX9D/VIX30 and every ratio built
from them share one underlying vol-level factor. Partial correlation controls
for every other feature in the set when reporting one pair, so what survives
is closer to "does X move with Y given everything else here," not "do X and Y
happen to share a common driver."

Deliberately **not** run on every numeric column in the panel — dumping in
credit, margin, wing geometry, and every _close/_close_prev variant alongside
the handful of signals under test would bury the term-structure/SMA story in
a wall of collinear, mostly-irrelevant pairs. Callers pass an explicit,
curated ``columns`` list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def partial_corr_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute the partial correlation matrix for a curated set of columns.

    Uses the standard precision-matrix identity: for the correlation matrix R
    of the given columns, the partial correlation of i and j controlling for
    every other column is ``-P_ij / sqrt(P_ii * P_jj)`` where ``P = R^-1``.
    Rows with a missing value in any listed column are dropped (listwise
    deletion) so every pairwise entry is computed on the same sample.

    Args:
        df: Source frame.
        columns: Columns to include. Keep this short and specific — this is
            not a substitute for choosing which relationships matter.

    Returns:
        A square DataFrame of partial correlations, columns and index both
        ``columns``, with 1.0 on the diagonal.
    """
    data = df[columns].dropna()
    if len(data) < len(columns) + 2:
        raise ValueError(
            f"Only {len(data)} complete rows for {len(columns)} columns; "
            "not enough to invert the correlation matrix."
        )

    corr = data.corr().to_numpy()
    precision = np.linalg.pinv(corr)
    diag = np.sqrt(np.diag(precision))
    with np.errstate(divide="ignore", invalid="ignore"):
        partial = -precision / np.outer(diag, diag)
    np.fill_diagonal(partial, 1.0)

    return pd.DataFrame(partial, index=columns, columns=columns)


def correlation_vs_partial(
    df: pd.DataFrame, columns: list[str], target: str
) -> pd.DataFrame:
    """Compare each feature's raw correlation with ``target`` to its partial correlation.

    A feature whose raw correlation is large but whose partial correlation
    collapses toward zero is riding on the other features in ``columns`` — a
    shared vol-level factor is the usual suspect here. A feature where the two
    numbers agree is telling you something more nearly its own.

    Args:
        df: Source frame.
        columns: Full feature set (should include ``target``).
        target: Column to read off from both matrices, e.g. ``"pl"``.

    Returns:
        One row per feature (excluding ``target``), sorted by the gap between
        the raw and partial correlation, largest first.
    """
    data = df[columns].dropna()
    raw = data.corr(method="spearman")[target]
    partial = partial_corr_matrix(data, columns)[target]

    out = pd.DataFrame(
        {
            "raw_corr": raw.drop(target),
            "partial_corr": partial.drop(target),
        }
    )
    out["abs_gap"] = (out["raw_corr"] - out["partial_corr"]).abs()
    return out.sort_values("abs_gap", ascending=False)
