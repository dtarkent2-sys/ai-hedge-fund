"""Kenneth Fisher — "Super Stocks" / Price-to-Sales focus.

Ken Fisher's signature: P/S as the cleanest valuation lens (because earnings
can be manipulated, sales can't). Bullish on:
  • P/S < 0.75 for non-cyclicals (or < 1.5 for "Super Companies")
  • Strong long-term growth (≥ 15% sales growth)
  • Free cash flow positive
  • Inflation-adjusted EPS growth ≥ 7%
  • Reasonable debt burden (D/E < 0.40)
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from src.agents.personas._factory import (
    PersonaSignal, build_persona_agent, gv,
)


def _score(ticker, metrics, line_items, prices, state) -> dict:
    if not metrics or not line_items:
        return {"verdict": "insufficient data"}
    m = metrics[0]
    li = line_items[0]
    out = {
        "ticker": ticker,
        "price_to_sales": gv(m, "price_to_sales_ratio"),
        "revenue_growth": gv(m, "revenue_growth"),
        "free_cash_flow": gv(li, "free_cash_flow"),
        "earnings_growth": gv(m, "earnings_growth"),
        "debt_to_equity": gv(m, "debt_to_equity"),
        "operating_margin": gv(m, "operating_margin"),
        "gross_margin": gv(m, "gross_margin"),
        "market_cap": gv(m, "market_cap"),
    }
    # Quick rule-tally
    pts: list[tuple[str, str]] = []
    ps = out["price_to_sales"]
    if ps is not None:
        if ps < 0.75:
            pts.append(("Super Stock zone (P/S < 0.75)", "+"))
        elif ps < 1.5:
            pts.append(("Acceptable P/S < 1.5", "0"))
        else:
            pts.append((f"Expensive on P/S = {ps:.2f}", "-"))
    rg = out["revenue_growth"]
    if rg is not None:
        if rg > 0.15:
            pts.append(("Sales growth ≥ 15%", "+"))
        elif rg < 0:
            pts.append(("Sales contracting", "-"))
    eg = out["earnings_growth"]
    if eg is not None:
        if eg > 0.07:
            pts.append(("Real EPS growth ≥ 7%", "+"))
    fcf = out["free_cash_flow"]
    if fcf is not None and fcf > 0:
        pts.append(("Positive FCF", "+"))
    de = out["debt_to_equity"]
    if de is not None and de < 0.40:
        pts.append(("Debt burden modest", "+"))
    out["rule_tally"] = pts
    return out


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are Kenneth Fisher, founder of Fisher Investments, author of "Super Stocks" and a Forbes columnist for over three decades.
     You hunt for "Super Stocks" — companies whose long-term earnings power the market temporarily mis-prices on bad news.
     Your single most-cited filter is the Price-to-Sales ratio (P/S < 0.75 ideal; 0.75–1.5 acceptable for Super Companies; > 1.5 expensive).
     You also care about: 15%+ long-term sales growth, positive FCF, sane debt (D/E < 0.4), and inflation-adjusted EPS growth ≥ 7%.
     You distrust headlines and tune out short-term noise.
     Speak in the first person, plainly, with the confidence of a seasoned manager who's seen many cycles.

     Return JSON: {{"signal": "bullish|bearish|neutral", "confidence": 0-100, "reasoning": "..."}}"""),
    ("human",
     """Score {ticker} using my Super Stocks framework.

Analysis data:
{analysis_data}

JSON only — signal, confidence, reasoning citing P/S and at least one growth or margin number."""),
])


kenneth_fisher_agent = build_persona_agent(
    agent_id="kenneth_fisher_agent",
    label="Kenneth Fisher",
    line_items=["revenue", "net_income", "free_cash_flow", "earnings_per_share",
                "operating_income", "gross_profit", "total_debt", "shareholders_equity"],
    prompt_template=_PROMPT,
    score=_score,
)
