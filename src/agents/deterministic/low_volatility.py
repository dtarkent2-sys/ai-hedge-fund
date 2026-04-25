"""Low-volatility / defensive-equity agent.

Computes annualized realized vol of the trailing 60 trading days. Stocks with
low realized vol historically deliver higher risk-adjusted returns ("low-vol
anomaly", Pim van Vliet). The Sharpe-like reading combines the trailing return
with realized vol.

Bullish: low vol + non-negative trailing return.
Bearish: high vol + negative trailing return.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from src.graph.state import AgentState
from src.tools.api import get_prices
from src.agents.deterministic._helpers import (
    emit_signal,
    finalize,
    for_each_ticker,
    get_api_key,
)

WINDOW_DAYS = 60  # ~3 months


def low_volatility_agent(state: AgentState, agent_id: str = "low_volatility_agent"):
    api_key = get_api_key(state)
    end_date_str = state["data"]["end_date"]
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    start_date = (end_date - timedelta(days=180)).strftime("%Y-%m-%d")  # safety margin

    def score_ticker(ticker: str) -> dict:
        prices = get_prices(ticker, start_date, end_date_str, api_key=api_key)
        if len(prices) < WINDOW_DAYS + 1:
            return emit_signal(
                state, agent_id, ticker,
                signal="neutral", confidence=30,
                reasoning=f"Need ≥{WINDOW_DAYS+1} bars for vol; got {len(prices)}.",
            )

        # prices are oldest-first
        window = prices[-(WINDOW_DAYS + 1):]
        rets: list[float] = []
        for i in range(1, len(window)):
            p1 = window[i - 1].close
            p2 = window[i].close
            if p1 > 0 and p2 > 0:
                rets.append(math.log(p2 / p1))
        if len(rets) < WINDOW_DAYS - 5:
            return emit_signal(
                state, agent_id, ticker,
                signal="neutral", confidence=30,
                reasoning="Returns series too sparse to compute vol.",
            )

        mean_r = sum(rets) / len(rets)
        var = sum((r - mean_r) ** 2 for r in rets) / max(1, len(rets) - 1)
        daily_vol = math.sqrt(var)
        annualized_vol = daily_vol * math.sqrt(252)

        trailing_return = (window[-1].close / window[0].close) - 1.0 if window[0].close > 0 else 0.0
        # Annualized realised return for the window
        ann_return = (1.0 + trailing_return) ** (252 / max(1, WINDOW_DAYS)) - 1.0
        sharpe_like = (ann_return - 0.04) / annualized_vol if annualized_vol > 0 else 0.0  # rf ≈ 4%

        # Calibration: broad-equity ann vol typically 15-25%. Below 18% = "low vol".
        if annualized_vol < 0.18 and trailing_return >= 0:
            signal, confidence = "bullish", 75
        elif annualized_vol < 0.25 and trailing_return >= 0:
            signal, confidence = "bullish", 60
        elif annualized_vol > 0.50 or (annualized_vol > 0.30 and trailing_return < -0.05):
            signal, confidence = "bearish", 70
        else:
            signal, confidence = "neutral", 50

        # Sharpe override: anything north of 1.0 is unambiguously good
        if sharpe_like > 1.0 and signal != "bearish":
            signal = "bullish"
            confidence = max(confidence, 75)

        reasoning = (
            f"Annualized vol (60d): {annualized_vol*100:.1f}%\n"
            f"  Trailing 60d return: {trailing_return*100:+.1f}%   "
            f"annualized: {ann_return*100:+.1f}%\n"
            f"  Sharpe-like (rf=4%): {sharpe_like:+.2f}"
        )
        return emit_signal(
            state, agent_id, ticker,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            extras={
                "ann_vol": annualized_vol,
                "trailing_return_60d": trailing_return,
                "sharpe_like": sharpe_like,
            },
        )

    signals = for_each_ticker(state, agent_id, score_ticker)
    return finalize(state, agent_id, "Low Volatility", signals)
