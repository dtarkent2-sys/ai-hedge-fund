"""Mohanram G-Score agent.

8-point growth-quality score from Mohanram (2005), "Separating Winners from
Losers among Low Book-to-Market Stocks Using Financial Statement Analysis."
Designed for growth (low B/M) stocks, complementing Piotroski's value (high
B/M) screen.

Tests (1 point each):
  G1  ROA above peer median (proxy: > 5%)
  G2  CFO/Total Assets above peer median (proxy: > 5%)
  G3  CFO > Net Income (low accruals)
  G4  Earnings variance below peer median (proxy: low EPS std/mean)
  G5  Sales-growth variance below peer median (proxy: low rev growth std/mean)
  G6  R&D / Assets above peer median (proxy: > 3%)
  G7  CapEx / Assets above peer median (proxy: > 4%)
  G8  Advertising / Assets above peer median  ← AV doesn't surface ad spend
        separately; we skip this one.

So the scoring scale is 0-7 here. 6-7 → bullish; 3-5 → neutral; 0-2 → bearish.
"""
from __future__ import annotations

import statistics

from src.graph.state import AgentState
from src.tools.api import get_financial_metrics, search_line_items
from src.agents.deterministic._helpers import (
    emit_signal,
    finalize,
    for_each_ticker,
    get_api_key,
    safe_div,
)


def mohanram_agent(state: AgentState, agent_id: str = "mohanram_agent"):
    api_key = get_api_key(state)
    end_date = state["data"]["end_date"]

    def score_ticker(ticker: str) -> dict:
        # Need 4-5 yrs of annual data for variance calcs.
        items = search_line_items(
            ticker,
            ["net_income", "total_assets", "free_cash_flow", "research_and_development",
             "capital_expenditure", "revenue"],
            end_date, period="annual", limit=5, api_key=api_key,
        )
        if len(items) < 3:
            return emit_signal(
                state, agent_id, ticker,
                signal="neutral", confidence=30,
                reasoning="Need ≥3 annual periods for G-Score (variance calcs).",
            )

        latest = items[0]
        ni = getattr(latest, "net_income", None)
        ta = getattr(latest, "total_assets", None)
        ocf = getattr(latest, "free_cash_flow", None)  # adapter's FCFF approximates OCF for ratio
        rd = getattr(latest, "research_and_development", None)
        cx = getattr(latest, "capital_expenditure", None)
        rev = getattr(latest, "revenue", None)

        roa = safe_div(ni, ta)
        cfo_to_assets = safe_div(ocf, ta)
        rd_to_assets = safe_div(rd or 0, ta) if rd else None
        capex_to_assets = safe_div(abs(cx) if cx is not None else None, ta)

        # Variance proxies (coefficient of variation = std / |mean|)
        eps_proxy = [getattr(it, "net_income", None) for it in items]
        eps_proxy = [v for v in eps_proxy if v is not None]
        revs = [getattr(it, "revenue", None) for it in items]
        revs = [v for v in revs if v is not None]

        def cv(seq: list[float]) -> float | None:
            if len(seq) < 3:
                return None
            mean = sum(seq) / len(seq)
            if abs(mean) < 1:
                return None
            try:
                std = statistics.pstdev(seq)
            except statistics.StatisticsError:
                return None
            return std / abs(mean)

        eps_cv = cv(eps_proxy)
        # For sales we want growth variance not level variance — compute YoY growths first
        rev_growths: list[float] = []
        for i in range(len(revs) - 1):
            if revs[i + 1] != 0:
                rev_growths.append((revs[i] - revs[i + 1]) / abs(revs[i + 1]))
        rev_growth_cv = cv(rev_growths) if len(rev_growths) >= 3 else None

        tests: list[tuple[str, bool | None]] = [
            ("ROA > 5%", None if roa is None else roa > 0.05),
            ("CFO/Assets > 5%", None if cfo_to_assets is None else cfo_to_assets > 0.05),
            ("CFO > Net income", None if (ocf is None or ni is None) else ocf > ni),
            ("Earnings variance low (CV < 0.5)", None if eps_cv is None else eps_cv < 0.5),
            ("Revenue growth variance low (CV < 1.0)",
                None if rev_growth_cv is None else rev_growth_cv < 1.0),
            ("R&D/Assets > 3%", None if rd_to_assets is None else rd_to_assets > 0.03),
            ("CapEx/Assets > 4%", None if capex_to_assets is None else capex_to_assets > 0.04),
        ]
        passed = sum(1 for _, ok in tests if ok is True)
        skipped = sum(1 for _, ok in tests if ok is None)
        max_score = 7 - skipped

        if max_score == 0:
            signal, confidence = "neutral", 30
        elif passed >= 6:
            signal, confidence = "bullish", 85
        elif passed >= 4:
            signal, confidence = "bullish", 60
        elif passed >= 2:
            signal, confidence = "neutral", 50
        else:
            signal, confidence = "bearish", 65

        lines = [f"Mohanram G-Score: {passed}/{max_score}"]
        for name, ok in tests:
            mark = "✔" if ok is True else ("✘" if ok is False else "—")
            lines.append(f"  {mark} {name}")
        if skipped:
            lines.append(f"  ({skipped} test(s) skipped due to missing data)")
        reasoning = "\n".join(lines)

        return emit_signal(
            state, agent_id, ticker,
            signal=signal, confidence=confidence,
            reasoning=reasoning,
            extras={"g_score": passed, "max_score": max_score},
        )

    signals = for_each_ticker(state, agent_id, score_ticker)
    return finalize(state, agent_id, "Mohanram G-Score", signals)
