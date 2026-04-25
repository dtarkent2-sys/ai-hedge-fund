"""Wesley Gray — "Quantitative Value": deep-value with a quality moat.

Gray (Alpha Architect) refines Greenblatt by:
  • EV/EBIT in the cheapest 10% (we use < 8 as a strong cut; < 12 acceptable)
  • Quality screen: high & stable F-Score-like profitability
  • Margin defense (operating margin stable & positive)
  • Avoidance of distress: no negative net income, low D/E
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from src.agents.personas._factory import build_persona_agent, gv


def _score(ticker, metrics, line_items, prices, state) -> dict:
    if not metrics or not line_items:
        return {"verdict": "insufficient data"}
    m = metrics[0]
    li = line_items[0]

    ebit = gv(m, "ebit")
    ev = gv(m, "enterprise_value")
    ev_ebit = (ev / ebit) if (ev and ebit and ebit > 0) else None
    op_margin = gv(m, "operating_margin")
    de = gv(m, "debt_to_equity")
    ni = gv(li, "net_income")

    # Margin stability across reported periods
    op_incomes = [gv(it, "operating_income") for it in line_items]
    revs = [gv(it, "revenue") for it in line_items]
    margins = [
        (oi / rv) for oi, rv in zip(op_incomes, revs)
        if oi is not None and rv is not None and rv > 0
    ]
    margin_floor = min(margins) if margins else None
    margin_negative_ever = (margin_floor is not None and margin_floor < 0)

    return {
        "ticker": ticker,
        "ev_to_ebit": ev_ebit,
        "operating_margin": op_margin,
        "min_op_margin_history": margin_floor,
        "margin_negative_ever": margin_negative_ever,
        "debt_to_equity": de,
        "net_income": ni,
    }


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are Wesley Gray of Alpha Architect, author of "Quantitative Value."
     Your method refines Greenblatt:
       • EV/EBIT < 8 = strong value zone; 8-12 = acceptable; > 12 = expensive
       • Margins stable and positive across history (no losses)
       • Net income positive
       • Manageable debt (D/E < 1.0)

     A stock that's deeply cheap on EV/EBIT AND has stable, positive margins = bullish.
     Cheap but margin instability = neutral or bearish (likely a quality trap).
     Expensive on EV/EBIT = bearish in this framework.

     Be quantitatively rigorous. Cite EV/EBIT and at least one quality check.

     Return JSON: {{"signal": "bullish|bearish|neutral", "confidence": 0-100, "reasoning": "..."}}"""),
    ("human",
     """Score {ticker} via Quantitative Value.

{analysis_data}

JSON only."""),
])


wesley_gray_agent = build_persona_agent(
    agent_id="wesley_gray_agent",
    label="Wesley Gray",
    line_items=["operating_income", "revenue", "net_income", "ebit"],
    prompt_template=_PROMPT,
    score=_score,
    line_items_limit=8,
)
