"""Dividend Aristocrat agent.

Uses AV's per-share DIVIDENDS history to score income quality:
  • TTM dividend yield
  • 5-year dividend CAGR (ought to be positive — declines flag a cut)
  • Streak of years with non-decreasing TTM dividend (Aristocrat ≥ 25y, but
    we'll surface anything with ≥ 5y as bullish here since most names won't
    qualify under the strict Aristocrat / Index rule).

Bullish: positive yield + 5-yr CAGR > 0 + streak ≥ 5 yrs
Neutral: pays dividends but stagnant or short streak
Bearish: no dividends OR cut history (5-yr CAGR < 0)
"""
from __future__ import annotations

from src.graph.state import AgentState
from src.tools.alphavantage import (
    get_dividend_history,
    trailing_12m_dividend,
    dividend_growth_cagr,
)
from src.tools.api import get_financial_metrics
from src.agents.deterministic._helpers import (
    emit_signal,
    finalize,
    fmt_pct,
    for_each_ticker,
    get_api_key,
)


def _years_of_non_decreasing_ttm(ticker: str, end_date: str, api_key) -> int:
    """How many consecutive years (working backward from end_date) the trailing-12m
    dividend was ≥ the prior year's TTM."""
    # Walk back up to 30 years
    year = int(end_date[:4])
    suffix = end_date[4:]
    streak = 0
    prev_ttm: float | None = None
    for offset in range(0, 30):
        anchor = f"{year - offset}{suffix}"
        ttm = trailing_12m_dividend(ticker, anchor, api_key=api_key)
        if ttm <= 0:
            break
        if prev_ttm is None:
            prev_ttm = ttm
            continue
        # Walking backward: ttm = older year. We want non-decreasing FORWARD,
        # so older ttm should be ≤ newer ttm = prev_ttm.
        if ttm <= prev_ttm * 1.001:
            streak += 1
            prev_ttm = ttm
        else:
            break
    return streak


def dividend_aristocrat_agent(state: AgentState, agent_id: str = "dividend_aristocrat_agent"):
    api_key = get_api_key(state)
    end_date = state["data"]["end_date"]

    def score_ticker(ticker: str) -> dict:
        rows = get_dividend_history(ticker, api_key=api_key)
        if not rows:
            return emit_signal(
                state, agent_id, ticker,
                signal="bearish", confidence=55,
                reasoning="Pays no dividends. Bearish from an income perspective; neutral from a growth perspective.",
                extras={"dividend_yield": 0, "dividend_growth_5y": None, "streak_years": 0},
            )

        ttm_dps = trailing_12m_dividend(ticker, end_date, api_key=api_key)
        cagr_5y = dividend_growth_cagr(ticker, end_date, years=5, api_key=api_key)
        streak = _years_of_non_decreasing_ttm(ticker, end_date, api_key)

        # Pull yield from FinancialMetrics so we use the same period-price as everyone else
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=1, api_key=api_key)
        yld = getattr(metrics[0], "dividend_yield", None) if metrics else None

        if ttm_dps == 0:
            return emit_signal(
                state, agent_id, ticker,
                signal="bearish", confidence=55,
                reasoning="No dividends in trailing 12 months.",
                extras={"dividend_yield": 0, "dividend_growth_5y": None, "streak_years": streak},
            )

        if cagr_5y is not None and cagr_5y < 0:
            signal, confidence = "bearish", 75
            verdict = "5-yr CAGR negative — likely dividend cut history."
        elif streak >= 25 and (cagr_5y or 0) > 0:
            signal, confidence = "bullish", 90
            verdict = "True Aristocrat (25+y of non-decreasing dividends)."
        elif streak >= 10 and (cagr_5y or 0) > 0.03:
            signal, confidence = "bullish", 80
            verdict = "Strong dividend grower."
        elif streak >= 5 and (cagr_5y or 0) > 0:
            signal, confidence = "bullish", 65
            verdict = "Reliable dividend grower."
        elif streak >= 2:
            signal, confidence = "neutral", 55
            verdict = "Pays dividends but limited growth track record."
        else:
            signal, confidence = "neutral", 45
            verdict = "Recent / inconsistent dividend payer."

        reasoning = (
            f"Dividend Aristocrat verdict: {verdict}\n"
            f"  TTM yield: {fmt_pct(yld)}   TTM dividend per share: ${ttm_dps:.2f}\n"
            f"  5-yr CAGR: {fmt_pct(cagr_5y)}\n"
            f"  Non-decreasing streak: {streak} year(s)"
        )
        return emit_signal(
            state, agent_id, ticker,
            signal=signal, confidence=confidence,
            reasoning=reasoning,
            extras={
                "dividend_yield": yld,
                "ttm_dps": ttm_dps,
                "dividend_growth_5y": cagr_5y,
                "streak_years": streak,
            },
        )

    signals = for_each_ticker(state, agent_id, score_ticker)
    return finalize(state, agent_id, "Dividend Aristocrat", signals)
