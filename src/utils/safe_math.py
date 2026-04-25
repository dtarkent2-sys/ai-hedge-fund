"""Numerical helpers used by agents — safe against the failure modes Python's
arithmetic operators have on financial data (negative bases under fractional
exponents → complex numbers; division by zero; missing values).
"""
from __future__ import annotations
import math
from typing import Optional


def safe_cagr(latest: Optional[float], oldest: Optional[float], years: float) -> Optional[float]:
    """Compound annual growth rate, returning None on undefined input.

    CAGR is mathematically only defined when both endpoints are strictly
    positive — `(b/a) ** (1/n)` returns a complex number when `b/a < 0`,
    which then crashes any later `<` / `>` comparison with:

        TypeError: '<' not supported between instances of 'float' and 'complex'

    This helper returns None for those cases so callers can guard with
    `if value is not None and value > threshold`.
    """
    if latest is None or oldest is None:
        return None
    if not isinstance(latest, (int, float)) or not isinstance(oldest, (int, float)):
        return None
    if oldest <= 0 or latest <= 0:
        return None
    if years is None or years <= 0:
        return None
    try:
        return (latest / oldest) ** (1.0 / years) - 1.0
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    """Plain division that returns None on missing/zero inputs."""
    if num is None or den is None:
        return None
    if den == 0:
        return None
    try:
        v = num / den
        if isinstance(v, complex) or math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError, ZeroDivisionError):
        return None
