"""Joel Greenblatt — "The Little Book That Beats the Market" (Magic Formula persona).

The deterministic version (greenblatt.py) computes earnings yield + ROC and
emits a numeric verdict. This persona wraps the same factors in Greenblatt's
narrative voice — useful as a heuristic check that complements the quant.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from src.agents.personas._factory import build_persona_agent, gv


def _score(ticker, metrics, line_items, prices, state) -> dict:
    if not metrics:
        return {"verdict": "insufficient data"}
    m = metrics[0]
    li = line_items[0] if line_items else None

    ebit = gv(m, "ebit")
    ev = gv(m, "enterprise_value")
    ca = gv(li, "current_assets")
    cl = gv(li, "current_liabilities")
    ta = gv(li, "total_assets")
    intang = gv(li, "intangible_assets") or 0

    nwc = (ca - cl) if (ca is not None and cl is not None) else None
    nfa = (ta - ca - intang) if (ta and ca is not None) else None
    invested_capital = (nwc + nfa) if (nwc is not None and nfa is not None) else None

    earnings_yield = (ebit / ev) if (ebit and ev) else None
    return_on_capital = (ebit / invested_capital) if (ebit and invested_capital and invested_capital > 0) else None

    return {
        "ticker": ticker,
        "earnings_yield": earnings_yield,
        "return_on_capital": return_on_capital,
        "ebit": ebit,
        "enterprise_value": ev,
        "invested_capital": invested_capital,
    }


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are Joel Greenblatt, founder of Gotham Capital and author of "The Little Book That Beats the Market."
     Your "Magic Formula" combines two factors:
       1. Earnings Yield = EBIT / Enterprise Value (cheapness)
       2. Return on Capital = EBIT / (Net Working Capital + Net Fixed Assets) (business quality)

     A stock that's cheap (high earnings yield) AND good (high ROC) = bullish.
     Cheap but low ROC = neutral (might be a value trap).
     Expensive but high ROC = neutral (great business, paying up).
     Expensive AND low ROC = bearish.

     Speak plainly. Use thresholds: earnings yield > 12% is cheap, > 6% fair, < 6% expensive.
     Return on capital > 25% is high quality, 10-25% is mid, < 10% is low.

     Return JSON: {{"signal": "bullish|bearish|neutral", "confidence": 0-100, "reasoning": "..."}}"""),
    ("human",
     """Score {ticker} using the Magic Formula.

Factors:
{analysis_data}

JSON only — cite both earnings yield and return on capital with their thresholds."""),
])


joel_greenblatt_agent = build_persona_agent(
    agent_id="joel_greenblatt_agent",
    label="Joel Greenblatt",
    line_items=["current_assets", "current_liabilities", "total_assets",
                "intangible_assets", "ebit"],
    prompt_template=_PROMPT,
    score=_score,
)
