"""Dashan Huang Twin Momentum agent.

From Huang, Zhang & Zhong (2018), "Twin Momentum" — combining seven
fundamental variables into a single fundamental-momentum composite, then
overlaying it on classical 12-1 price momentum. Stocks ranking in the top
quintile of the combined measure outperformed; the combination roughly
doubled vanilla price-momentum's alpha.

The seven fundamental variables:
  1. Earnings (TTM net income, growth direction)
  2. Return on equity (ROE)
  3. Return on assets (ROA)
  4. Accrual operating profitability to equity (AOP/E)
       ≈ (operating income − accruals) / equity, where accruals ≈ NI − OCF
  5. Cash operating profitability to assets (COP/A) ≈ OCF / assets
  6. Gross profit to assets (GP/A)
  7. Net payout ratio    = (dividends + buybacks) / net income

Single-stock approximation: absolute thresholds + period-over-period direction
(true Twin Momentum cross-sectionally ranks the universe; we don't have it).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from src.graph.state import AgentState
from src.tools.api import get_financial_metrics, get_prices, search_line_items
from src.agents.deterministic._helpers import (
    emit_signal,
    finalize,
    fmt_pct,
    for_each_ticker,
    get_api_key,
    safe_div,
)


def _fund_score(metrics_now, metrics_prior, line_now, line_prior, mcap):
    """Score the seven fundamentals: each +1 if 'good', -1 if 'bad', 0 if missing.

    Plus a momentum/direction bonus when the current value improves over prior.
    """
    def gv(o, a):
        return getattr(o, a, None) if o is not None else None

    pts: list[tuple[str, str, float]] = []
    score = 0

    # 1. Earnings (TTM net income, growth)
    ni_now = gv(line_now, "net_income")
    ni_prior = gv(line_prior, "net_income")
    if ni_now is not None:
        if ni_now > 0:
            score += 1; mark = "+"
        else:
            score -= 1; mark = "-"
        pts.append((f"Net income: ${ni_now/1e9:.1f}B", mark, ni_now))
        if ni_prior is not None and ni_prior > 0 and ni_now > ni_prior * 1.05:
            score += 1
            pts.append(("Earnings improving YoY", "+", ni_now / ni_prior - 1))

    # 2. ROE
    roe = gv(metrics_now, "return_on_equity")
    if roe is not None:
        if roe > 0.15: score += 1; pts.append((f"ROE {roe*100:.1f}%", "+", roe))
        elif roe < 0.05: score -= 1; pts.append((f"ROE {roe*100:.1f}%", "-", roe))

    # 3. ROA
    roa = gv(metrics_now, "return_on_assets")
    if roa is not None:
        if roa > 0.07: score += 1; pts.append((f"ROA {roa*100:.1f}%", "+", roa))
        elif roa < 0.02: score -= 1; pts.append((f"ROA {roa*100:.1f}%", "-", roa))

    # 4. Accrual operating profitability to equity (AOP/E)
    op_inc = gv(metrics_now, "ebit")  # adapter sets ebit ≈ operating income
    fcf = gv(line_now, "free_cash_flow")
    equity = gv(metrics_now, "debt_to_equity")  # placeholder; we need raw equity
    # Compute equity from total_debt and D/E or from line_items
    total_debt = gv(metrics_now, "total_debt")
    raw_equity = (total_debt / equity) if (total_debt and equity) else None
    accruals = (ni_now - fcf) if (ni_now is not None and fcf is not None) else None
    aop_to_equity = None
    if op_inc is not None and accruals is not None and raw_equity:
        aop_to_equity = (op_inc - accruals) / raw_equity
    if aop_to_equity is not None:
        if aop_to_equity > 0.20: score += 1; pts.append((f"AOP/E {aop_to_equity*100:.1f}%", "+", aop_to_equity))
        elif aop_to_equity < 0: score -= 1; pts.append((f"AOP/E {aop_to_equity*100:.1f}%", "-", aop_to_equity))

    # 5. Cash operating profitability to assets (COP/A)
    ta_now = gv(line_now, "total_assets")
    cop_to_assets = safe_div(fcf, ta_now)
    if cop_to_assets is not None:
        if cop_to_assets > 0.10: score += 1; pts.append((f"COP/A {cop_to_assets*100:.1f}%", "+", cop_to_assets))
        elif cop_to_assets < 0.02: score -= 1; pts.append((f"COP/A {cop_to_assets*100:.1f}%", "-", cop_to_assets))

    # 6. Gross profit to assets (GP/A)
    gp = gv(line_now, "gross_profit")
    gp_to_assets = safe_div(gp, ta_now)
    if gp_to_assets is not None:
        if gp_to_assets > 0.30: score += 1; pts.append((f"GP/A {gp_to_assets*100:.1f}%", "+", gp_to_assets))
        elif gp_to_assets < 0.10: score -= 1; pts.append((f"GP/A {gp_to_assets*100:.1f}%", "-", gp_to_assets))

    # 7. Net payout ratio = (dividends + buybacks) / net income
    div = gv(line_now, "dividends_and_other_cash_distributions")
    issuance = gv(line_now, "issuance_or_purchase_of_equity_shares")
    div_paid = abs(div) if div is not None and div < 0 else 0
    buyback = -issuance if (issuance is not None and issuance < 0) else 0
    net_payout = div_paid + buyback
    payout_ratio = safe_div(net_payout, ni_now) if (ni_now is not None and ni_now > 0) else None
    if payout_ratio is not None:
        if payout_ratio > 0.50: score += 1; pts.append((f"Net payout / NI: {payout_ratio*100:.0f}%", "+", payout_ratio))
        elif payout_ratio < 0: score -= 1; pts.append((f"Net dilution (negative payout)", "-", payout_ratio))

    return score, pts


def _price_mom_12_1(prices) -> float | None:
    if len(prices) < 280:
        return None
    end_idx = len(prices) - 1
    skip_idx = max(0, end_idx - 21)
    start_idx = max(0, skip_idx - 252)
    if prices[start_idx].close <= 0 or prices[skip_idx].close <= 0:
        return None
    return prices[skip_idx].close / prices[start_idx].close - 1.0


def dashan_huang_agent(state: AgentState, agent_id: str = "dashan_huang_agent"):
    api_key = get_api_key(state)
    end_date_str = state["data"]["end_date"]
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    price_start = (end_date - timedelta(days=440)).strftime("%Y-%m-%d")

    def score_ticker(ticker: str) -> dict:
        metrics = get_financial_metrics(ticker, end_date_str, period="annual", limit=2, api_key=api_key)
        items = search_line_items(
            ticker,
            ["net_income", "free_cash_flow", "total_assets", "gross_profit",
             "dividends_and_other_cash_distributions", "issuance_or_purchase_of_equity_shares"],
            end_date_str, period="annual", limit=2, api_key=api_key,
        )
        if not metrics:
            return emit_signal(state, agent_id, ticker, signal="neutral", confidence=30,
                               reasoning="Insufficient annual fundamentals.")
        m_now = metrics[0]
        m_prior = metrics[1] if len(metrics) > 1 else None
        li_now = items[0] if items else None
        li_prior = items[1] if len(items) > 1 else None
        mcap = getattr(m_now, "market_cap", None)

        fund_score, fund_pts = _fund_score(m_now, m_prior, li_now, li_prior, mcap)

        prices = get_prices(ticker, price_start, end_date_str, api_key=api_key)
        price_mom = _price_mom_12_1(prices)

        # Combine: both positive = strong bullish; both negative = strong bearish; mixed = neutral.
        fund_dir = 1 if fund_score >= 3 else -1 if fund_score <= -2 else 0
        price_dir = 1 if (price_mom or 0) > 0.05 else -1 if (price_mom or 0) < -0.05 else 0
        combined = fund_dir + price_dir

        if combined >= 2:
            signal, confidence = "bullish", 90
        elif combined == 1:
            signal, confidence = "bullish", 70
        elif combined == 0:
            signal, confidence = "neutral", 50
        elif combined == -1:
            signal, confidence = "bearish", 70
        else:
            signal, confidence = "bearish", 90

        lines = [
            f"Twin Momentum composite — fundamental {fund_score:+d}, price 12-1 {fmt_pct(price_mom)}, combined dir {combined:+d}",
            "  Fundamental factor checks:",
        ]
        for label, mark, _ in fund_pts:
            lines.append(f"    {mark} {label}")
        lines.append(f"  Price momentum (12-1): {fmt_pct(price_mom)}")

        return emit_signal(
            state, agent_id, ticker,
            signal=signal, confidence=confidence,
            reasoning="\n".join(lines),
            extras={
                "fundamental_score": fund_score,
                "price_momentum_12_1": price_mom,
                "combined_direction": combined,
                "n_factors_scored": len(fund_pts),
            },
        )

    signals = for_each_ticker(state, agent_id, score_ticker)
    return finalize(state, agent_id, "Dashan Huang Twin Momentum", signals)
