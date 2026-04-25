"""David Dreman — "Contrarian Investment Strategies": deep contrarian value.

Dreman's filters target the cheapest 20% of stocks by valuation:
  • P/E in bottom 20% of market (we use < 12 as cap)
  • P/B in bottom 20% (< 1.5)
  • P/CF in bottom 20% (< 8)
  • Dividend yield > market average (we use > 2.5%)
  • Quality screen: positive earnings, debt manageable
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from src.agents.personas._factory import build_persona_agent, gv


def _score(ticker, metrics, line_items, prices, state) -> dict:
    if not metrics:
        return {"verdict": "insufficient data"}
    m = metrics[0]
    li = line_items[0] if line_items else None

    pe = gv(m, "price_to_earnings_ratio")
    pb = gv(m, "price_to_book_ratio")
    payout = gv(m, "payout_ratio")
    de = gv(m, "debt_to_equity")
    ni = gv(li, "net_income")

    # P/CF using FCF as cash-flow proxy
    fcf = gv(li, "free_cash_flow")
    mcap = gv(m, "market_cap")
    p_cf = (mcap / fcf) if (mcap and fcf and fcf > 0) else None

    # Real dividend yield from AV's DIVIDENDS endpoint when available.
    div_yield = gv(m, "dividend_yield")
    if div_yield is None and payout and pe and pe > 0:
        div_yield = payout * (1 / pe)

    # Tally cheap factors
    cheap_count = 0
    if pe is not None and 0 < pe < 12:
        cheap_count += 1
    if pb is not None and 0 < pb < 1.5:
        cheap_count += 1
    if p_cf is not None and 0 < p_cf < 8:
        cheap_count += 1
    if div_yield is not None and div_yield > 0.025:
        cheap_count += 1

    quality_ok = (ni is not None and ni > 0) and (de is None or de < 1.5)

    return {
        "ticker": ticker,
        "pe": pe, "pb": pb, "p_cf": p_cf,
        "dividend_yield": div_yield,
        "debt_to_equity": de,
        "net_income": ni,
        "cheap_factor_count": cheap_count,
        "quality_screen_ok": quality_ok,
    }


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are David Dreman, founder of Dreman Value Management, author of "Contrarian Investment Strategies."
     You buy the cheapest 20% of stocks by P/E, P/B, and P/CF, with above-average dividend yield, and require basic quality (positive earnings + manageable debt).
     Your contrarian thesis: market overreactions create bargains; mean reversion rewards patience.

     Decision rule:
       • 3-4 cheap-factors satisfied AND quality screen passes → bullish
       • 0-1 cheap factors OR quality screen fails → bearish
       • 2 cheap factors OR quality borderline → neutral

     You're contrarian — bullish on out-of-favor names, suspicious of crowded "story" stocks. Cite specific multiples.

     Return JSON: {{"signal": "bullish|bearish|neutral", "confidence": 0-100, "reasoning": "..."}}"""),
    ("human",
     """Score {ticker} from a Dreman contrarian-value perspective.

{analysis_data}

JSON only — name how many cheap factors hit and whether quality clears."""),
])


david_dreman_agent = build_persona_agent(
    agent_id="david_dreman_agent",
    label="David Dreman",
    line_items=["free_cash_flow", "net_income"],
    prompt_template=_PROMPT,
    score=_score,
)
