"""James O'Shaughnessy — "What Works on Wall Street" / Trending Value strategy.

The Trending Value model ranks stocks on a 6-factor value composite + then
filters for 6-month price strength. Composite value factors:
  • Low P/E
  • Low P/B
  • Low P/S
  • Low P/CF (price-to-cashflow)
  • Low EV/EBITDA
  • High shareholder yield (div + buybacks)

Then keep only those with positive 6-month momentum.
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

    # Value sub-factors (lower is better for first 5; higher is better for shareholder yield)
    pe = gv(m, "price_to_earnings_ratio")
    pb = gv(m, "price_to_book_ratio")
    ps = gv(m, "price_to_sales_ratio")
    ev_ebitda = gv(m, "enterprise_value_to_ebitda_ratio")

    # P/CF: Market Cap / Operating Cash Flow (use FCF as a workable proxy)
    fcf = gv(li, "free_cash_flow")
    mcap = gv(m, "market_cap")
    p_cf = (mcap / fcf) if (mcap and fcf and fcf > 0) else None

    # Shareholder yield = (dividends paid + net buybacks) / market cap
    div = gv(li, "dividends_and_other_cash_distributions")  # negative outflow in our adapter
    issuance = gv(li, "issuance_or_purchase_of_equity_shares")  # +issuance / -buyback
    div_paid = abs(div) if div is not None else 0
    buyback = -issuance if (issuance is not None and issuance < 0) else 0
    sh_yield = ((div_paid + buyback) / mcap) if mcap and mcap > 0 else None

    # 6-month price momentum
    mom_6m = None
    if len(prices) >= 130:
        try:
            p_now = prices[-1].close
            p_then = prices[-126].close
            if p_then > 0:
                mom_6m = p_now / p_then - 1
        except Exception:
            pass

    factors = {
        "pe": pe, "pb": pb, "ps": ps, "p_cf": p_cf, "ev_ebitda": ev_ebitda,
        "shareholder_yield": sh_yield, "momentum_6m": mom_6m,
    }
    # Cheap-or-not heuristic per factor
    pts: list[tuple[str, str]] = []
    if pe is not None: pts.append((f"P/E {pe:.1f}", "+" if 0 < pe < 15 else ("0" if pe < 25 else "-")))
    if pb is not None: pts.append((f"P/B {pb:.1f}", "+" if 0 < pb < 1.5 else ("0" if pb < 3 else "-")))
    if ps is not None: pts.append((f"P/S {ps:.1f}", "+" if 0 < ps < 1.0 else ("0" if ps < 2.5 else "-")))
    if p_cf is not None: pts.append((f"P/CF {p_cf:.1f}", "+" if p_cf < 10 else ("0" if p_cf < 20 else "-")))
    if ev_ebitda is not None: pts.append((f"EV/EBITDA {ev_ebitda:.1f}", "+" if ev_ebitda < 8 else ("0" if ev_ebitda < 15 else "-")))
    if sh_yield is not None: pts.append((f"Shareholder yield {sh_yield*100:.1f}%", "+" if sh_yield > 0.05 else ("0" if sh_yield > 0.02 else "-")))
    if mom_6m is not None: pts.append((f"6m momentum {mom_6m*100:+.1f}%", "+" if mom_6m > 0.05 else ("0" if mom_6m > -0.05 else "-")))
    factors["rule_tally"] = pts
    return factors


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are James O'Shaughnessy, author of "What Works on Wall Street" and creator of the Trending Value strategy.
     You evaluate stocks via a 6-factor value composite (low P/E, P/B, P/S, P/CF, EV/EBITDA + high shareholder yield), then keep only those with positive 6-month price momentum.
     A stock that scores cheap on most factors AND has positive 6-month price strength = bullish.
     Cheap but weakening price = bearish (value trap).
     Expensive but strong momentum = neutral (not your strategy).
     Expensive AND weak = bearish.
     Be data-driven and quantitative; the strategy is the strategy.

     Return JSON: {{"signal": "bullish|bearish|neutral", "confidence": 0-100, "reasoning": "..."}}"""),
    ("human",
     """Apply the Trending Value framework to {ticker}.

Factor data:
{analysis_data}

JSON only — cite at least 2 of the 6 value factors plus the momentum check."""),
])


oshaughnessy_agent = build_persona_agent(
    agent_id="oshaughnessy_agent",
    label="James O'Shaughnessy",
    line_items=["free_cash_flow", "dividends_and_other_cash_distributions",
                "issuance_or_purchase_of_equity_shares"],
    prompt_template=_PROMPT,
    score=_score,
    prices_window_days=240,  # need 6mo+ for momentum
)
