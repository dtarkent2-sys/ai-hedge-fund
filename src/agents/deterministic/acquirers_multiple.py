"""Tobias Carlisle "Acquirer's Multiple" agent.

Single-factor deep-value test: EV / EBIT (sometimes EV / Operating Earnings).
Lower is cheaper from an acquirer's perspective. Carlisle's research argues
this single multiple matched or beat the Magic Formula in backtests.

Bullish if EV/EBIT < 8 (deep value).
Neutral if 8-15 (fair).
Bearish if > 15 (expensive on this measure).
Bearish if EBIT ≤ 0 (no earnings to price).
"""
from __future__ import annotations

from src.graph.state import AgentState
from src.tools.api import get_financial_metrics
from src.agents.deterministic._helpers import (
    emit_signal,
    finalize,
    for_each_ticker,
    get_api_key,
    safe_div,
)


def acquirers_multiple_agent(state: AgentState, agent_id: str = "acquirers_multiple_agent"):
    api_key = get_api_key(state)
    end_date = state["data"]["end_date"]

    def score_ticker(ticker: str) -> dict:
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=1, api_key=api_key)
        if not metrics:
            return {"signal": "neutral", "confidence": 30, "reasoning": "No financial metrics."}
        m = metrics[0]
        ebit = getattr(m, "ebit", None)
        ev = getattr(m, "enterprise_value", None)

        if ebit is None or ev is None:
            return emit_signal(
                state, agent_id, ticker,
                signal="neutral", confidence=30,
                reasoning="Acquirer's Multiple inputs missing (EBIT or EV).",
            )

        if ebit <= 0:
            return emit_signal(
                state, agent_id, ticker,
                signal="bearish", confidence=70,
                reasoning=f"EBIT non-positive (${ebit/1e9:.2f}B) — no operating earnings to value at any multiple.",
                extras={"ebit": ebit, "enterprise_value": ev},
            )

        am = safe_div(ev, ebit)
        if am is None:
            return emit_signal(
                state, agent_id, ticker,
                signal="neutral", confidence=30,
                reasoning="EV/EBIT undefined.",
            )

        if am < 5:
            signal, confidence = "bullish", 88
            label = "very cheap"
        elif am < 8:
            signal, confidence = "bullish", 75
            label = "cheap (deep value)"
        elif am < 12:
            signal, confidence = "neutral", 55
            label = "fair"
        elif am < 18:
            signal, confidence = "bearish", 60
            label = "expensive"
        else:
            signal, confidence = "bearish", 80
            label = "very expensive"

        reasoning = (
            f"Acquirer's Multiple (EV/EBIT) = {am:.1f}x ({label}).\n"
            f"  EV = ${ev/1e9:.1f}B   EBIT = ${ebit/1e9:.1f}B\n"
            f"  Carlisle thresholds: <8 cheap, 8-15 fair, >15 expensive."
        )
        return emit_signal(
            state, agent_id, ticker,
            signal=signal, confidence=confidence,
            reasoning=reasoning,
            extras={"acquirers_multiple": am, "ebit": ebit, "enterprise_value": ev},
        )

    signals = for_each_ticker(state, agent_id, score_ticker)
    return finalize(state, agent_id, "Acquirer's Multiple (Carlisle)", signals)
