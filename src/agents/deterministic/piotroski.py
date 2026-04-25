"""Piotroski F-Score agent.

Scores a stock on nine binary financial-strength tests from Piotroski (2000),
"Value Investing: The Use of Historical Financial Statement Information…".
8-9 → strong bullish, 7 → bullish, 4-6 → neutral, ≤ 3 → bearish.

No LLM call. Reasoning is a plain-text breakdown of which tests passed.
"""
from __future__ import annotations

from src.graph.state import AgentState
from src.tools.api import get_financial_metrics, search_line_items
from src.agents.deterministic._helpers import (
    emit_signal,
    finalize,
    for_each_ticker,
    get_api_key,
    safe_div,
)

LINE_ITEMS = [
    "net_income",
    "operating_cashflow" if False else "free_cash_flow",  # AV exposes OCF via line items
    "total_assets",
    "current_assets",
    "current_liabilities",
    "total_debt",
    "outstanding_shares",
    "revenue",
    "gross_profit",
]


def piotroski_agent(state: AgentState, agent_id: str = "piotroski_agent"):
    api_key = get_api_key(state)
    end_date = state["data"]["end_date"]

    def score_ticker(ticker: str) -> dict:
        # Need at least 2 annual periods for the trend-based tests.
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=3, api_key=api_key)
        if len(metrics) < 2:
            return {"signal": "neutral", "confidence": 30, "reasoning": "Insufficient annual history (need ≥2 yrs)."}
        cur, prev = metrics[0], metrics[1]

        line_items = search_line_items(
            ticker,
            ["net_income", "free_cash_flow", "total_assets", "current_assets", "current_liabilities",
             "total_debt", "outstanding_shares", "revenue", "gross_profit"],
            end_date,
            period="annual",
            limit=3,
            api_key=api_key,
        )
        li_cur = line_items[0] if len(line_items) > 0 else None
        li_prev = line_items[1] if len(line_items) > 1 else None

        # Pull values defensively
        def gv(obj, attr):
            return getattr(obj, attr, None) if obj is not None else None

        ni_cur = gv(li_cur, "net_income")
        ni_prev = gv(li_prev, "net_income")
        ocf_cur = gv(li_cur, "free_cash_flow")  # FCFF proxy from our adapter — close enough for the OCF>NI test
        ta_cur = gv(li_cur, "total_assets")
        ta_prev = gv(li_prev, "total_assets")
        debt_cur = gv(li_cur, "total_debt")
        debt_prev = gv(li_prev, "total_debt")
        ca_cur = gv(li_cur, "current_assets")
        ca_prev = gv(li_prev, "current_assets")
        cl_cur = gv(li_cur, "current_liabilities")
        cl_prev = gv(li_prev, "current_liabilities")
        sh_cur = gv(li_cur, "outstanding_shares")
        sh_prev = gv(li_prev, "outstanding_shares")
        rev_cur = gv(li_cur, "revenue")
        rev_prev = gv(li_prev, "revenue")
        gp_cur = gv(li_cur, "gross_profit")
        gp_prev = gv(li_prev, "gross_profit")

        roa_cur = safe_div(ni_cur, ta_cur)
        roa_prev = safe_div(ni_prev, ta_prev)
        cur_ratio_cur = safe_div(ca_cur, cl_cur)
        cur_ratio_prev = safe_div(ca_prev, cl_prev)
        gross_margin_cur = safe_div(gp_cur, rev_cur)
        gross_margin_prev = safe_div(gp_prev, rev_prev)
        asset_turnover_cur = safe_div(rev_cur, ta_cur)
        asset_turnover_prev = safe_div(rev_prev, ta_prev)

        tests: list[tuple[str, bool | None]] = []
        # Profitability (4)
        tests.append(("Net income > 0", (ni_cur is not None) and ni_cur > 0))
        tests.append(("Operating cash flow > 0", (ocf_cur is not None) and ocf_cur > 0))
        tests.append((
            "ΔROA > 0",
            None if (roa_cur is None or roa_prev is None) else roa_cur > roa_prev,
        ))
        tests.append((
            "OCF > Net income (low accruals)",
            None if (ocf_cur is None or ni_cur is None) else ocf_cur > ni_cur,
        ))
        # Leverage / Liquidity (3)
        tests.append((
            "Long-term debt down YoY",
            None if (debt_cur is None or debt_prev is None) else debt_cur < debt_prev,
        ))
        tests.append((
            "Current ratio up YoY",
            None if (cur_ratio_cur is None or cur_ratio_prev is None) else cur_ratio_cur > cur_ratio_prev,
        ))
        tests.append((
            "No new shares issued",
            None if (sh_cur is None or sh_prev is None) else sh_cur <= sh_prev * 1.001,
        ))
        # Operating efficiency (2)
        tests.append((
            "Gross margin up YoY",
            None if (gross_margin_cur is None or gross_margin_prev is None) else gross_margin_cur > gross_margin_prev,
        ))
        tests.append((
            "Asset turnover up YoY",
            None if (asset_turnover_cur is None or asset_turnover_prev is None) else asset_turnover_cur > asset_turnover_prev,
        ))

        passed = sum(1 for _, ok in tests if ok is True)
        skipped = sum(1 for _, ok in tests if ok is None)
        max_score = 9 - skipped

        # Map score → signal; skipped tests count as neither pass nor fail
        if max_score == 0:
            signal, confidence = "neutral", 30
        else:
            # 7-9 → bullish, 4-6 → neutral, 0-3 → bearish on the 9-point scale
            if passed >= 7:
                signal = "bullish"
                confidence = min(90, 60 + (passed - 7) * 10)  # 60/70/80/90
            elif passed >= 4:
                signal = "neutral"
                confidence = 50
            else:
                signal = "bearish"
                confidence = 60 if passed <= 1 else 50

        # Plain-text reasoning, one line per test
        lines = [f"Piotroski F-Score: {passed}/{max_score}"]
        for name, ok in tests:
            mark = "✔" if ok is True else ("✘" if ok is False else "—")
            lines.append(f"  {mark} {name}")
        if skipped:
            lines.append(f"  ({skipped} test(s) skipped due to missing data)")
        reasoning = "\n".join(lines)

        return emit_signal(
            state,
            agent_id,
            ticker,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            extras={"f_score": passed, "max_score": max_score},
        )

    signals = for_each_ticker(state, agent_id, score_ticker)
    return finalize(state, agent_id, "Piotroski F-Score", signals)
