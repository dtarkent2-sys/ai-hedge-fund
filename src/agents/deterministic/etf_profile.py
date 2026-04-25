"""ETF Profile agent.

For ETFs (anything AV's ETF_PROFILE endpoint returns data for), this agent
surfaces the structural fundamentals an ETF investor cares about:
  • Net assets (size / liquidity proxy)
  • Expense ratio (cost drag)
  • Portfolio turnover (tax / friction proxy)
  • Top sector concentration
  • Top-10 holding concentration

For non-ETFs, returns a low-confidence neutral.

Bullish: cheap (expense < 0.20%), large (> $10B AUM), reasonable turnover.
Bearish: expensive (> 0.75%) or tiny (< $100M AUM, illiquid).
"""
from __future__ import annotations

from src.graph.state import AgentState
from src.tools.alphavantage import get_etf_profile
from src.agents.deterministic._helpers import (
    emit_signal,
    finalize,
    fmt_pct,
    for_each_ticker,
    get_api_key,
)


def _f(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def etf_profile_agent(state: AgentState, agent_id: str = "etf_profile_agent"):
    api_key = get_api_key(state)

    def score_ticker(ticker: str) -> dict:
        prof = get_etf_profile(ticker, api_key=api_key)
        if not prof:
            return emit_signal(
                state, agent_id, ticker,
                signal="neutral", confidence=20,
                reasoning=f"{ticker} is not an ETF (no profile returned). This agent only applies to ETFs.",
            )

        net_assets = _f(prof.get("net_assets"))
        expense = _f(prof.get("net_expense_ratio"))
        turnover = _f(prof.get("portfolio_turnover"))
        div_yld = _f(prof.get("dividend_yield"))
        leveraged = (prof.get("leveraged") or "").upper() in ("TRUE", "YES", "1")

        sectors = prof.get("sectors") or []
        sectors_sorted = sorted(sectors, key=lambda s: _f(s.get("weight")) or 0, reverse=True)
        top_sector = sectors_sorted[0] if sectors_sorted else None

        holdings = prof.get("holdings") or []
        top10_concentration = sum(_f(h.get("weight")) or 0 for h in holdings[:10])
        top1 = holdings[0] if holdings else None

        score = 0
        if expense is not None:
            if expense < 0.0020: score += 2     # < 0.20% cheap
            elif expense < 0.0050: score += 1   # < 0.50% acceptable
            elif expense > 0.0075: score -= 2   # > 0.75% expensive
        if net_assets is not None:
            if net_assets >= 10e9: score += 2
            elif net_assets >= 1e9: score += 1
            elif net_assets < 100e6: score -= 2
        if turnover is not None:
            if turnover > 1.0: score -= 1   # > 100% turnover = high friction
        if leveraged: score -= 2

        if score >= 3:
            signal, confidence = "bullish", 80
        elif score >= 1:
            signal, confidence = "bullish", 60
        elif score >= -1:
            signal, confidence = "neutral", 55
        elif score >= -3:
            signal, confidence = "bearish", 65
        else:
            signal, confidence = "bearish", 80

        lines = [f"ETF profile composite: {score:+d}"]
        if net_assets is not None:
            label = "huge" if net_assets >= 10e9 else "large" if net_assets >= 1e9 else "small" if net_assets >= 100e6 else "tiny/illiquid"
            lines.append(f"  Net assets: ${net_assets/1e9:.1f}B ({label})")
        if expense is not None:
            lines.append(f"  Expense ratio: {expense*100:.2f}%  ({'cheap' if expense < 0.002 else 'fair' if expense < 0.005 else 'expensive'})")
        if turnover is not None:
            lines.append(f"  Portfolio turnover: {fmt_pct(turnover)}")
        if div_yld is not None:
            lines.append(f"  Dividend yield: {fmt_pct(div_yld)}")
        if top_sector:
            lines.append(f"  Top sector: {top_sector.get('sector')} ({fmt_pct(_f(top_sector.get('weight')))})")
        if top1:
            lines.append(f"  Top holding: {top1.get('symbol')} {top1.get('description')} ({fmt_pct(_f(top1.get('weight')))})")
        lines.append(f"  Top-10 concentration: {fmt_pct(top10_concentration)}")
        if leveraged:
            lines.append("  ⚠ Leveraged ETF — decay risk on long holds")

        return emit_signal(
            state, agent_id, ticker,
            signal=signal, confidence=confidence,
            reasoning="\n".join(lines),
            extras={
                "net_assets": net_assets, "expense_ratio": expense, "turnover": turnover,
                "dividend_yield": div_yld, "leveraged": leveraged,
                "top_sector": top_sector, "top_holding": top1,
                "top_10_concentration": top10_concentration,
            },
        )

    signals = for_each_ticker(state, agent_id, score_ticker)
    return finalize(state, agent_id, "ETF Profile", signals)
