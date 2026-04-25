"""Martin Zweig — "Winning on Wall Street": growth at a reasonable price + earnings persistence.

Zweig's checklist (paraphrased):
  • Earnings persistence: 4+ years of rising EPS
  • Sales growth tracks earnings growth (not just margin tricks)
  • Reasonable P/E (≤ market average × 1.4 generally; we use ≤ 30 as cap)
  • Strong recent earnings acceleration (current quarter > prior)
  • Insider/institutional buying (we don't have insider here)
  • Reasonable debt (D/E < industry; we use < 0.5)
  • Relative strength vs market positive
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from src.agents.personas._factory import build_persona_agent, gv


def _score(ticker, metrics, line_items, prices, state) -> dict:
    if not metrics or not line_items:
        return {"verdict": "insufficient data"}
    m = metrics[0]
    li = line_items

    # EPS persistence — count consecutive YoY rises in net income (oldest→newest)
    nis = list(reversed([gv(it, "net_income") for it in li if gv(it, "net_income") is not None]))
    streak = 0
    for i in range(1, len(nis)):
        if nis[i] > nis[i - 1]:
            streak += 1
        else:
            break
    persistence_years = streak

    # Sales-vs-earnings parallel growth
    revs = list(reversed([gv(it, "revenue") for it in li if gv(it, "revenue") is not None]))
    rev_growth = (revs[-1] / revs[0] - 1) if (len(revs) >= 2 and revs[0] > 0) else None
    eps_growth = gv(m, "earnings_growth")
    sales_eps_aligned = (
        rev_growth is not None and eps_growth is not None
        and rev_growth > 0 and eps_growth > 0
        and abs(rev_growth - eps_growth) < max(0.5, abs(rev_growth) * 0.5)
    )

    # Acceleration: latest YoY net income growth > median of prior 3
    accel = None
    if len(nis) >= 4 and nis[-2] > 0:
        latest_yoy = (nis[-1] - nis[-2]) / nis[-2]
        prior_yoys = [(nis[i] - nis[i - 1]) / nis[i - 1] for i in range(1, len(nis) - 1) if nis[i - 1] > 0]
        if prior_yoys:
            median_prior = sorted(prior_yoys)[len(prior_yoys) // 2]
            accel = latest_yoy > median_prior

    pe = gv(m, "price_to_earnings_ratio")
    de = gv(m, "debt_to_equity")

    return {
        "ticker": ticker,
        "earnings_persistence_years": persistence_years,
        "sales_growth_total": rev_growth,
        "earnings_growth_yoy": eps_growth,
        "sales_eps_aligned": sales_eps_aligned,
        "earnings_accelerating": accel,
        "pe": pe,
        "debt_to_equity": de,
    }


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are Martin Zweig, author of "Winning on Wall Street" and creator of the Zweig Forecast.
     Your strategy: growth at a reasonable price with strong earnings persistence.
     Key checks:
       • EPS rising 4+ consecutive years
       • Sales growth ≈ earnings growth (no margin tricks)
       • Earnings accelerating (latest QoQ growth > prior trend)
       • P/E reasonable — not above market multiple × 1.4 (we cap at 30)
       • Modest debt (D/E < 0.5)
     If 3+ checks pass strongly → bullish. If most fail → bearish. Mixed → neutral.
     You're disciplined, data-driven, and skeptical of high P/E "story" stocks.

     Return JSON: {{"signal": "bullish|bearish|neutral", "confidence": 0-100, "reasoning": "..."}}"""),
    ("human",
     """Score {ticker} using the Zweig framework.

Analysis:
{analysis_data}

JSON only — cite EPS persistence years and at least one of (sales-EPS alignment, acceleration, P/E)."""),
])


martin_zweig_agent = build_persona_agent(
    agent_id="martin_zweig_agent",
    label="Martin Zweig",
    line_items=["revenue", "net_income", "earnings_per_share", "total_debt", "shareholders_equity"],
    prompt_template=_PROMPT,
    score=_score,
    period="annual",
    line_items_limit=6,
)
