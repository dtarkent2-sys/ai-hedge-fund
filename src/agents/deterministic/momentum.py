"""Cross-sectional momentum agent (12-1 academic factor).

Definition: 12-month total return EXCLUDING the most recent month, computed
on adjusted close prices. The "skip-1-month" form is the standard academic
implementation that avoids short-term reversal contamination.

Strong positive momentum → bullish; flat → neutral; sustained negative → bearish.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from src.graph.state import AgentState
from src.tools.api import get_prices
from src.agents.deterministic._helpers import (
    emit_signal,
    finalize,
    for_each_ticker,
    get_api_key,
)


def momentum_agent(state: AgentState, agent_id: str = "momentum_agent"):
    api_key = get_api_key(state)
    end_date_str = state["data"]["end_date"]
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    # Pull 14 months back so we have enough room around the 12-1 window
    start_date = (end_date - timedelta(days=440)).strftime("%Y-%m-%d")

    def score_ticker(ticker: str) -> dict:
        prices = get_prices(ticker, start_date, end_date_str, api_key=api_key)
        if len(prices) < 200:
            return emit_signal(
                state, agent_id, ticker,
                signal="neutral",
                confidence=30,
                reasoning=f"Need ≥200 daily bars; got {len(prices)}. Momentum undefined.",
            )

        # prices are sorted oldest-first by our adapter
        end_idx = len(prices) - 1                              # most recent close
        # ~21 trading days = 1 month. Skip 1 month for 12-1 form.
        skip_idx = max(0, end_idx - 21)
        # 12 months back from skip_idx = 12 * 21 = 252 trading days back
        start_idx = max(0, skip_idx - 252)

        if start_idx >= skip_idx:
            return emit_signal(
                state, agent_id, ticker,
                signal="neutral",
                confidence=30,
                reasoning="Insufficient history for the 12-1 momentum window.",
            )

        p_start = prices[start_idx].close
        p_skip = prices[skip_idx].close
        p_end = prices[end_idx].close

        if p_start <= 0 or p_skip <= 0 or p_end <= 0:
            return emit_signal(
                state, agent_id, ticker,
                signal="neutral",
                confidence=30,
                reasoning="Non-positive price in window.",
            )

        mom_12_1 = (p_skip / p_start) - 1.0
        mom_1m = (p_end / p_skip) - 1.0
        mom_3m_skip_idx = max(0, end_idx - 63)
        mom_3m = (p_end / prices[mom_3m_skip_idx].close) - 1.0 if prices[mom_3m_skip_idx].close > 0 else 0.0

        # Map 12-1 return to a signal. Calibrated to broad-equity history:
        #   > 15% → strong bullish, 5–15% → bullish, −5–5% → neutral,
        #   −15 to −5% → bearish, < −15% → strong bearish.
        if mom_12_1 > 0.15:
            signal, confidence = "bullish", 85
        elif mom_12_1 > 0.05:
            signal, confidence = "bullish", 70
        elif mom_12_1 > -0.05:
            signal, confidence = "neutral", 50
        elif mom_12_1 > -0.15:
            signal, confidence = "bearish", 65
        else:
            signal, confidence = "bearish", 80

        # If recent month diverges sharply from the trailing 12, soften confidence.
        if (mom_12_1 > 0 and mom_1m < -0.05) or (mom_12_1 < 0 and mom_1m > 0.05):
            confidence = max(40, confidence - 15)

        reasoning = (
            f"12-1 momentum (skip-month): {mom_12_1*100:+.1f}%\n"
            f"  trailing 1m: {mom_1m*100:+.1f}%   trailing 3m: {mom_3m*100:+.1f}%\n"
            f"  window: {prices[start_idx].time} → {prices[skip_idx].time}"
        )
        return emit_signal(
            state, agent_id, ticker,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            extras={"mom_12_1": mom_12_1, "mom_1m": mom_1m, "mom_3m": mom_3m},
        )

    signals = for_each_ticker(state, agent_id, score_ticker)
    return finalize(state, agent_id, "Momentum (12-1)", signals)
