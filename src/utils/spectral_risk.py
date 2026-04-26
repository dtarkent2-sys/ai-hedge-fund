"""ORCA AR₁-based market-concentration risk multiplier.

`risk_manager` already builds the correlation matrix across the run's
tickers for the per-ticker correlation multiplier. We piggyback on that
matrix and ask `hermes_research.orca.SpectralExtractor` to compute
spectral features. Only AR₁ (absorption ratio of the first eigenvalue
— the share of variance explained by the dominant principal component)
matters here:

  AR₁ low      → market dispersed, factors fragmented → normal sizing
  AR₁ high     → everything moving together (crisis / concentration regime) → shrink positions

The remaining 126 features are computed too but unused; if you want them
later, plug `SpectralExtractor.extract(corr)` directly. They're cheap.

Important caveat: AR₁ is a property of the *correlation matrix submitted
to risk_manager*, which is the run's basket — NOT the broader market.
With a 5-ticker run all in semis, AR₁ will always be high regardless of
whether the market itself is in a crisis. The multiplier is therefore
"how clustered is THIS request" more than "how scared should we be of
the market". Read it that way until we have a market-wide returns feed.
"""
from __future__ import annotations

import logging

from hermes_research.orca import SpectralExtractor

log = logging.getLogger(__name__)

_extractor = SpectralExtractor()


def compute_ar1_from_correlation(correlation_matrix) -> float | None:
    """Return AR₁ for a square correlation matrix (list[list[float]] or
    pandas DataFrame). Returns None if the matrix is too small or
    degenerate.
    """
    if correlation_matrix is None:
        return None
    # Coerce DataFrame → list of lists if needed.
    if hasattr(correlation_matrix, "values"):
        matrix = correlation_matrix.values.tolist()
    elif hasattr(correlation_matrix, "tolist"):
        matrix = correlation_matrix.tolist()
    else:
        matrix = correlation_matrix

    if not matrix or len(matrix) < 2 or len(matrix[0]) < 2:
        return None

    try:
        features = _extractor.extract(matrix)
        return float(features.get("AR1")) if features.get("AR1") is not None else None
    except Exception as exc:
        log.warning("ORCA AR1 extraction failed: %s", exc)
        return None


def calculate_concentration_multiplier(ar1: float | None) -> float:
    """Map AR₁ → cap multiplier alongside the existing volatility,
    correlation, and cluster multipliers in risk_manager. None / no data
    → 1.0 (no adjustment, fail open).

    Thresholds are conservative defaults — all uncalibrated against
    backtested portfolio outcomes. Tune after running paper trades.
    """
    if ar1 is None:
        return 1.00
    if ar1 < 0.50:
        return 1.00       # dispersed regime — normal sizing
    if ar1 < 0.70:
        return 0.85       # elevated correlation
    if ar1 < 0.85:
        return 0.65       # high concentration / crisis regime
    return 0.50            # severe crisis — half cap
