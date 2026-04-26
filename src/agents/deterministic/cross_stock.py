"""Cross-Stock Concentration agent (Huang et al., 2026 — adapted).

Most agents in this graph are independent: they look at one ticker and
emit a signal without caring what else is being analyzed. That's a
problem when the user feeds in NVDA + AMD + INTC + AVGO together — every
agent says "bullish on chips" and the portfolio manager doubles down on
a single sector.

This agent fixes that by running across the full ticker list. For each
ticker it builds a candidate-edge graph (sector / industry / market-cap
similarity) against the rest of the run and counts how many of its
neighbors are also in the request. Tickers that sit in the middle of a
tight cluster get a bearish concentration signal; tickers that are
sector-isolated within the request get a bullish "diversifying" boost.

Pure-math (no LLM call). Underpinned by `cross_stock` from the
`sharkquant-research` package.
"""
from __future__ import annotations

from src.graph.state import AgentState
from src.tools.alphavantage import _get_overview
from src.utils.api_key import get_api_key_from_state
from src.agents.deterministic._helpers import (
    emit_signal,
    finalize,
    for_each_ticker,
)

# Public package — installed via `pip install -e D:/sharkquant-research`.
from cross_stock import build_candidate_graph

# Thresholds chosen so the signal triples cleanly:
#   neighbors_in_run = 0..1  → bullish (diversifying — confidence 50-65)
#   neighbors_in_run = 2     → neutral
#   neighbors_in_run = 3+    → bearish (concentration risk — conf 60-85)
_BULLISH_BELOW = 2
_BEARISH_AT = 3


def _safe_int(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _profile(ticker: str, api_key: str | None) -> dict | None:
    """Pull AV OVERVIEW into the {ticker, sector, industry, market_cap} shape
    that build_candidate_graph expects. Returns None if AV has no data
    (delisted, ETF, micro-cap, etc.)."""
    o = _get_overview(ticker, api_key) or {}
    if not o.get("Symbol"):
        return None
    return {
        "ticker": o["Symbol"].upper(),
        "sector": (o.get("Sector") or "").lower() or None,
        "industry": (o.get("Industry") or "").lower() or None,
        "market_cap": _safe_int(o.get("MarketCapitalization")),
    }


def cross_stock_agent(state: AgentState, agent_id: str = "cross_stock_agent"):
    api_key = get_api_key_from_state(state, "ALPHA_VANTAGE_API_KEY")
    tickers: list[str] = state["data"]["tickers"]

    # Build the universe of profiles ONCE up-front. Stashed in a closure so
    # for_each_ticker's per-ticker body can read it without N**2 AV calls.
    profiles_by_ticker: dict[str, dict] = {}
    for t in tickers:
        prof = _profile(t, api_key)
        if prof:
            profiles_by_ticker[t] = prof

    # Edge graph also computed once. for_each_ticker just reads adjacency.
    if len(profiles_by_ticker) >= 2:
        profiles = list(profiles_by_ticker.values())
        _edges, adjacency = build_candidate_graph(profiles, profiles, top_k=5)
    else:
        adjacency = {}

    def score_ticker(ticker: str) -> dict:
        prof = profiles_by_ticker.get(ticker)

        # Single-ticker run, or AV had nothing on this name.
        if len(profiles_by_ticker) < 2:
            return emit_signal(
                state, agent_id, ticker,
                signal="neutral",
                confidence=30,
                reasoning="Single-ticker run — cross-stock concentration is undefined.",
                extras={"neighbors_in_run": 0, "tickers_in_run": list(profiles_by_ticker)},
            )
        if prof is None:
            return emit_signal(
                state, agent_id, ticker,
                signal="neutral",
                confidence=30,
                reasoning=f"No AV OVERVIEW for {ticker}; cannot place in cross-stock graph.",
                extras={"neighbors_in_run": None},
            )

        # build_candidate_graph emits a 0.3 baseline edge between every
        # pair (driven by fund_similarity even when sector/industry don't
        # match). Only count neighbors with shared sector/industry —
        # similarity ≥ 0.5 — so JPM doesn't get flagged as a neighbor of
        # NVDA just because their market caps fit through the top-K filter.
        in_run = set(profiles_by_ticker)
        neighbors = adjacency.get(ticker, [])
        neighbor_pairs = [
            (n_ticker, sim) for n_ticker, sim in neighbors
            if n_ticker in in_run and n_ticker != ticker and sim >= 0.5
        ]
        n = len(neighbor_pairs)
        sector = prof["sector"] or "unknown sector"
        industry = prof["industry"] or "unknown industry"

        if n >= _BEARISH_AT:
            signal = "bearish"
            # 3 neighbors → 60, 4 → 72, 5+ → 85
            confidence = min(85, 60 + (n - _BEARISH_AT) * 12)
            reasoning = (
                f"Concentration risk: {ticker} sits inside a {n + 1}-name cluster within the run "
                f"({sector} / {industry}). Adding it doubles down on names you're already evaluating: "
                f"{', '.join(t for t, _ in neighbor_pairs[:5])}."
            )
        elif n < _BULLISH_BELOW:
            signal = "bullish"
            # 0 neighbors → 65, 1 → 50
            confidence = 65 if n == 0 else 50
            reasoning = (
                f"Diversifying name: {ticker} has {n} sector/industry neighbor(s) in the run "
                f"({sector}). Adding it broadens the analysis."
            )
        else:
            signal = "neutral"
            confidence = 45
            reasoning = (
                f"{ticker} has {n} sibling(s) in the run ({sector}). "
                f"Moderate overlap — neither diversifying nor concentrated."
            )

        return emit_signal(
            state, agent_id, ticker,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            extras={
                "neighbors_in_run": n,
                "neighbor_tickers": [t for t, _ in neighbor_pairs[:8]],
                "sector": prof["sector"],
                "industry": prof["industry"],
                "market_cap": prof["market_cap"],
                "tickers_in_run": sorted(in_run),
            },
        )

    signals = for_each_ticker(state, agent_id, score_ticker)
    return finalize(state, agent_id, "Cross-Stock Concentration", signals)
