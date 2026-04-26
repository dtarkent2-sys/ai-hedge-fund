"""Per-ticker concentration counts for the run.

For each ticker in `tickers`, this computes how many *other* tickers in
the same run share its sector + industry (similarity ≥ 0.5 in the
cross_stock graph builder). risk_manager.py reads the result and
shrinks the position cap when a ticker is sitting inside a tight
cluster — so an NVDA + AMD + INTC + AVGO run doesn't end up with the PM
sizing each at 25% of NLV just because every analyst is bullish on
chips.

This is the right home for cross-stock signal: it's basket-level by
nature, and risk_manager already does volatility + correlation cap
multipliers in the same shape. Keeping it out of the analyst-signals
stream means the PM doesn't see "cross_stock_agent: bearish" as if it
were a fundamental view of the company.
"""
from __future__ import annotations

import logging

from src.tools.alphavantage import _get_overview
from cross_stock import build_candidate_graph

log = logging.getLogger(__name__)

# similarity ≥ 0.5 from build_candidate_graph means same sector+industry.
# Below that the edges fire on market-cap proximity alone and produce
# false positives (JPM ↔ NVDA at sim=0.3). Threshold is conservative;
# back-test before relaxing.
NEIGHBOR_SIM_THRESHOLD = 0.5

# top_k controls how many candidate neighbors each ticker can have in the
# graph. With small runs (≤ 8 tickers) this is mostly cosmetic; with
# larger universes it caps the fan-out cost. 5 is what the paper used.
GRAPH_TOP_K = 5


def _safe_int(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _profile(ticker: str, api_key: str | None) -> dict | None:
    o = _get_overview(ticker, api_key) or {}
    if not o.get("Symbol"):
        return None
    return {
        "ticker": o["Symbol"].upper(),
        "sector": (o.get("Sector") or "").lower() or None,
        "industry": (o.get("Industry") or "").lower() or None,
        "market_cap": _safe_int(o.get("MarketCapitalization")),
    }


def compute_cluster_counts(tickers: list[str], api_key: str | None) -> dict[str, int]:
    """Return {ticker: number_of_in_run_neighbors_with_similarity_>=_threshold}.

    Tickers without AV OVERVIEW data (delisted, ETFs, micro-caps) get 0
    neighbors. Single-ticker runs always return {ticker: 0}.
    """
    profiles_by_ticker: dict[str, dict] = {}
    for t in tickers:
        prof = _profile(t, api_key)
        if prof:
            profiles_by_ticker[t.upper()] = prof

    if len(profiles_by_ticker) < 2:
        return {t.upper(): 0 for t in tickers}

    profiles = list(profiles_by_ticker.values())
    try:
        _edges, adjacency = build_candidate_graph(profiles, profiles, top_k=GRAPH_TOP_K)
    except Exception as exc:
        log.warning("cross_stock graph build failed: %s — returning zeros", exc)
        return {t.upper(): 0 for t in tickers}

    in_run = set(profiles_by_ticker)
    counts: dict[str, int] = {}
    for ticker_upper in (t.upper() for t in tickers):
        if ticker_upper not in profiles_by_ticker:
            counts[ticker_upper] = 0
            continue
        neighbors = adjacency.get(ticker_upper, [])
        n = sum(
            1 for n_ticker, sim in neighbors
            if n_ticker in in_run and n_ticker != ticker_upper and sim >= NEIGHBOR_SIM_THRESHOLD
        )
        counts[ticker_upper] = n
    return counts


# Cluster multiplier — shape matches calculate_correlation_multiplier in
# risk_manager.py: ≥1.0 means "give it more room", <1.0 means "shrink the
# cap". Tuned conservatively: 0 neighbors keeps the existing cap, 3+
# neighbors lops 30% off the position size, 5+ lops half.
def calculate_cluster_multiplier(neighbors_in_run: int) -> float:
    if neighbors_in_run <= 1:
        return 1.00       # diversifying or only one cluster mate
    if neighbors_in_run == 2:
        return 0.85       # mild concentration penalty
    if neighbors_in_run == 3:
        return 0.70       # 3-name cluster — meaningful penalty
    if neighbors_in_run == 4:
        return 0.55       # 4-name cluster — heavy penalty
    return 0.50            # 5+ — never give more than half the vol-adjusted cap
