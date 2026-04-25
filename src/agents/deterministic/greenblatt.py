"""Greenblatt Magic Formula agent.

Two factors only, both from "The Little Book That Beats the Market":
  Earnings Yield  = EBIT / Enterprise Value
  Return on Capital = EBIT / (Net Working Capital + Net Fixed Assets)

High of both = bullish (a "wonderful business at a bargain price"). Low of both
= bearish.

No LLM call. Reasoning is the two factor numbers and where they fall vs.
sensible thresholds.
"""
from __future__ import annotations

from src.graph.state import AgentState
from src.tools.api import get_financial_metrics, search_line_items
from src.agents.deterministic._helpers import (
    emit_signal,
    finalize,
    fmt_pct,
    for_each_ticker,
    get_api_key,
    safe_div,
)


def greenblatt_agent(state: AgentState, agent_id: str = "greenblatt_agent"):
    api_key = get_api_key(state)
    end_date = state["data"]["end_date"]

    def score_ticker(ticker: str) -> dict:
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=1, api_key=api_key)
        if not metrics:
            return {"signal": "neutral", "confidence": 30, "reasoning": "No financial metrics."}
        m = metrics[0]

        ebit = getattr(m, "ebit", None)
        ev = getattr(m, "enterprise_value", None)

        items = search_line_items(
            ticker,
            ["current_assets", "current_liabilities", "total_assets", "intangible_assets",
             "goodwill_and_intangible_assets", "cash_and_equivalents"],
            end_date, period="ttm", limit=1, api_key=api_key,
        )
        li = items[0] if items else None

        def gv(o, a):
            return getattr(o, a, None) if o is not None else None

        ca = gv(li, "current_assets")
        cl = gv(li, "current_liabilities")
        ta = gv(li, "total_assets")
        intangibles = gv(li, "intangible_assets") or 0.0
        cash = gv(li, "cash_and_equivalents") or 0.0

        nwc = (ca - cl) if (ca is not None and cl is not None) else None
        # Net Fixed Assets ≈ Total Assets − Current Assets − Intangibles − Cash
        nfa = None
        if ta is not None and ca is not None:
            nfa = ta - ca - (intangibles or 0.0)

        invested_capital = None
        if nwc is not None and nfa is not None:
            invested_capital = max(nwc + nfa, 1.0)  # guard against absurdly small denominators

        earnings_yield = safe_div(ebit, ev)
        return_on_capital = safe_div(ebit, invested_capital)

        if earnings_yield is None or return_on_capital is None:
            return emit_signal(
                state, agent_id, ticker,
                signal="neutral",
                confidence=30,
                reasoning="Magic Formula inputs missing (EBIT, EV, NWC, or NFA).",
                extras={"earnings_yield": earnings_yield, "return_on_capital": return_on_capital},
            )

        # Thresholds calibrated to roughly mean Greenblatt-friendly territory.
        # Earnings yield: > 12% cheap, 6-12% fair, < 6% expensive.
        # Return on capital: > 25% high-quality, 10-25% mid, < 10% low.
        ey_score = 1 if earnings_yield > 0.12 else (0 if earnings_yield > 0.06 else -1)
        roc_score = 1 if return_on_capital > 0.25 else (0 if return_on_capital > 0.10 else -1)
        composite = ey_score + roc_score  # range −2 … +2

        if composite >= 2:
            signal, confidence = "bullish", 80
        elif composite == 1:
            signal, confidence = "bullish", 65
        elif composite == 0:
            signal, confidence = "neutral", 50
        elif composite == -1:
            signal, confidence = "bearish", 60
        else:
            signal, confidence = "bearish", 75

        reasoning = (
            f"Magic Formula composite {composite:+d}/+2:\n"
            f"  Earnings yield (EBIT/EV) = {fmt_pct(earnings_yield)}  "
            f"({'cheap' if earnings_yield > 0.12 else 'fair' if earnings_yield > 0.06 else 'expensive'})\n"
            f"  Return on capital (EBIT/IC) = {fmt_pct(return_on_capital)}  "
            f"({'high quality' if return_on_capital > 0.25 else 'mid' if return_on_capital > 0.10 else 'low'})"
        )

        return emit_signal(
            state, agent_id, ticker,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            extras={"earnings_yield": earnings_yield, "return_on_capital": return_on_capital,
                    "composite": composite},
        )

    signals = for_each_ticker(state, agent_id, score_ticker)
    return finalize(state, agent_id, "Greenblatt Magic Formula", signals)
