"""Motley Fool — small-cap growth screen ("Foolish Small Cap").

Tom & David Gardner's general framework for high-quality smaller growers:
  • Sub-$3B market cap (we use < $5B as a soft cap; > $20B = definitely not Foolish)
  • Sales growth ≥ 25%
  • Earnings growth ≥ 25%
  • Net margin > industry — proxy: > 7%
  • Insider ownership > 10% — we don't have this; proxy: low share issuance
  • Daily $-volume / market cap below threshold (proxy for thin trading)
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from src.agents.personas._factory import build_persona_agent, gv


def _score(ticker, metrics, line_items, prices, state) -> dict:
    if not metrics:
        return {"verdict": "insufficient data"}
    m = metrics[0]
    li = line_items[0] if line_items else None

    mcap = gv(m, "market_cap")
    rev_growth = gv(m, "revenue_growth")
    earn_growth = gv(m, "earnings_growth")
    net_margin = gv(m, "net_margin")
    op_margin = gv(m, "operating_margin")
    issuance = gv(li, "issuance_or_purchase_of_equity_shares")  # +ve = dilution, -ve = buyback

    size_label = (
        "small" if (mcap and mcap < 3e9) else
        "mid" if (mcap and mcap < 20e9) else
        "large/mega"
    )

    return {
        "ticker": ticker,
        "market_cap": mcap,
        "size_bucket": size_label,
        "revenue_growth": rev_growth,
        "earnings_growth": earn_growth,
        "net_margin": net_margin,
        "operating_margin": op_margin,
        "share_issuance_net": issuance,
        "qualifies_as_foolish_small_cap": (
            mcap is not None and mcap < 5e9
            and (rev_growth or 0) > 0.20
            and (earn_growth or 0) > 0.20
        ),
    }


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are the Motley Fool's "Foolish Small Cap" voice (Tom & David Gardner).
     You hunt for small-cap growth: < $3B market cap (loosely up to $5B), 25%+ revenue growth, 25%+ earnings growth, healthy margins, and insider alignment.
     You are upbeat, plain-spoken, willing to pay a premium for genuine quality growth.
     If the company is not in the small-cap zone, your conviction drops sharply (it's not a "Foolish" pick).
     Mega-caps are never bullish in this framework — flag them as out-of-mandate.

     Bullish: small-cap zone + double-digit growth + positive margins.
     Neutral: borderline size or mixed growth.
     Bearish: deteriorating growth or margin collapse.
     Out-of-mandate (large/mega-cap): default neutral, low confidence.

     Return JSON: {{"signal": "bullish|bearish|neutral", "confidence": 0-100, "reasoning": "..."}}"""),
    ("human",
     """Score {ticker} as the Foolish Small Cap framework.

{analysis_data}

JSON only — name the size bucket and at least one growth metric."""),
])


motley_fool_agent = build_persona_agent(
    agent_id="motley_fool_agent",
    label="Motley Fool",
    line_items=["issuance_or_purchase_of_equity_shares", "revenue", "net_income"],
    prompt_template=_PROMPT,
    score=_score,
)
