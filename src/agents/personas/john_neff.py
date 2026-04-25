"""John Neff — Vanguard Windsor Fund manager 1964–1995. Low-PE relative-value approach.

Neff's "Total Return Ratio":
  TRR = (Earnings growth + Dividend yield) / P/E
A TRR ≥ 2× the market average = bullish; ≤ market average = bearish.

Other Neff filters:
  • P/E meaningfully below market average (we use < 15 as cap)
  • Solid (mid-single-digit+) earnings growth
  • Yield from dividends, ideally
  • Strong free cash flow conversion
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from src.agents.personas._factory import build_persona_agent, gv


def _score(ticker, metrics, line_items, prices, state) -> dict:
    if not metrics:
        return {"verdict": "insufficient data"}
    m = metrics[0]

    pe = gv(m, "price_to_earnings_ratio")
    eg = gv(m, "earnings_growth")        # already a fraction (e.g. 0.10)
    payout = gv(m, "payout_ratio")
    fcf_yield = gv(m, "free_cash_flow_yield")
    # Real dividend yield from AV's DIVIDENDS endpoint (sums actual per-share
    # dividends in the trailing 12 months, divides by period price).
    div_yield = gv(m, "dividend_yield")
    # Fallback: payout × earnings yield if AV gave us nothing.
    if div_yield is None and payout and pe and pe > 0:
        div_yield = payout * (1 / pe)

    # Total Return Ratio
    trr = None
    if pe and pe > 0 and eg is not None:
        eg_pct = eg * 100
        dy_pct = (div_yield or 0) * 100
        trr = (eg_pct + dy_pct) / pe

    return {
        "ticker": ticker,
        "pe": pe,
        "earnings_growth": eg,
        "dividend_yield": div_yield,
        "payout_ratio": payout,
        "fcf_yield": fcf_yield,
        "total_return_ratio": trr,
        "trr_vs_market_proxy": ("good" if (trr or 0) > 2.0 else "fair" if (trr or 0) > 1.0 else "weak"),
    }


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are John Neff, manager of Vanguard Windsor Fund (1964–1995), one of the longest-tenured outperformers in mutual-fund history.
     Your strategy: "low PE plus dividend plus growth = total return ratio."
     TRR = (Earnings Growth % + Dividend Yield %) / P/E.
     A TRR ≥ 2.0 means double the market's typical reward — strongly bullish.
     TRR ≥ 1.0 = decent. TRR < 1.0 = weak.

     Also check:
       • P/E is meaningfully below the market average (you target ratios in the single-digit-to-mid-teens)
       • Earnings growth solid but not speculative (4-12% is your sweet spot; > 20% raises eyebrows)
       • Strong cash flow backs the earnings (FCF yield positive)

     You're disciplined, contrarian, and patient. You'll buy unloved stocks the market overlooks.

     Return JSON: {{"signal": "bullish|bearish|neutral", "confidence": 0-100, "reasoning": "..."}}"""),
    ("human",
     """Score {ticker} as John Neff would.

{analysis_data}

JSON only — cite TRR and at least one of (P/E vs market, earnings growth, FCF yield)."""),
])


john_neff_agent = build_persona_agent(
    agent_id="john_neff_agent",
    label="John Neff",
    line_items=["dividends_and_other_cash_distributions", "free_cash_flow", "earnings_per_share"],
    prompt_template=_PROMPT,
    score=_score,
)
