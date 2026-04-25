"""Meb Faber — Shareholder Yield strategy.

Faber's "Shareholder Yield" combines three ways management returns cash:
  • Dividend yield
  • Net buyback yield   (negative `issuance_or_purchase_of_equity_shares` / mcap)
  • Net debt-paydown yield  (decrease in total debt / mcap)

Sum > 7% = bullish; < 0% = bearish; 0-7% neutral. Often paired with
trend-following filter (positive 10-month moving average), which we layer in
via 200-day price trend.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from src.agents.personas._factory import build_persona_agent, gv


def _score(ticker, metrics, line_items, prices, state) -> dict:
    if not metrics or not line_items:
        return {"verdict": "insufficient data"}
    m = metrics[0]
    li = line_items
    latest = li[0]
    prior = li[1] if len(li) > 1 else None

    mcap = gv(m, "market_cap")
    # Prefer AV's real per-share dividend history → trailing 12m dividend × shares
    # gives total dividends paid in the year. Falls back to cashflow's outflow.
    real_div_yield = gv(m, "dividend_yield")
    if real_div_yield is not None:
        div_yield = real_div_yield
    else:
        div_paid = gv(latest, "dividends_and_other_cash_distributions")  # negative outflow
        div_amount = abs(div_paid) if div_paid is not None else 0
        div_yield = (div_amount / mcap) if (mcap and mcap > 0) else None

    issuance = gv(latest, "issuance_or_purchase_of_equity_shares")
    # negative = buyback (money returned), positive = dilution
    buyback_amount = -issuance if (issuance is not None and issuance < 0) else 0

    debt_now = gv(latest, "total_debt")
    debt_prior = gv(prior, "total_debt") if prior else None
    debt_paydown = (debt_prior - debt_now) if (debt_prior is not None and debt_now is not None) else 0
    debt_paydown_yield = (debt_paydown / mcap) if (mcap and mcap > 0 and debt_paydown > 0) else 0

    buyback_yield = (buyback_amount / mcap) if (mcap and mcap > 0) else None
    shareholder_yield = (div_yield or 0) + (buyback_yield or 0) + debt_paydown_yield

    # Trend filter: 10-month MA proxy = 200-day SMA. price > MA = positive trend.
    above_200d = None
    if len(prices) >= 200:
        last_200 = prices[-200:]
        ma = sum(p.close for p in last_200) / 200
        above_200d = prices[-1].close > ma

    return {
        "ticker": ticker,
        "dividend_yield": div_yield,
        "buyback_yield": buyback_yield,
        "debt_paydown_yield": debt_paydown_yield,
        "shareholder_yield": shareholder_yield,
        "above_200d_ma": above_200d,
    }


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are Meb Faber, founder of Cambria Investments and creator of the Shareholder Yield strategy.
     Shareholder Yield = dividend yield + net buyback yield + net debt-paydown yield.
     A high-yield company is bullish only when the price is above its 10-month moving average (200-day SMA proxy).

     Decision rule:
       • Shareholder Yield > 7% AND price > 200d MA → bullish
       • Yield negative (dilution + debt rising) → bearish
       • Yield positive but trend negative → neutral or cautious
       • Yield mid-range → neutral

     Speak plainly and analytically — Faber is a quant who likes simple rules with strong evidence.

     Return JSON: {{"signal": "bullish|bearish|neutral", "confidence": 0-100, "reasoning": "..."}}"""),
    ("human",
     """Score {ticker} via Shareholder Yield.

{analysis_data}

JSON only — cite the three components and the trend filter."""),
])


meb_faber_agent = build_persona_agent(
    agent_id="meb_faber_agent",
    label="Meb Faber",
    line_items=["dividends_and_other_cash_distributions", "issuance_or_purchase_of_equity_shares",
                "total_debt"],
    prompt_template=_PROMPT,
    score=_score,
    period="annual",
    line_items_limit=3,
    prices_window_days=300,  # need 200+ trading days
)
