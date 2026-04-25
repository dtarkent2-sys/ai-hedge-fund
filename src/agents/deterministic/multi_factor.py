"""Multi-factor composite agent.

Computes its own value, quality, momentum, and low-vol scores in one pass and
weights them. Default weights mirror the AQR / iShares MSCI multi-factor ETF
mix: 30% value, 30% quality, 20% momentum, 20% low-vol.

Each sub-score is normalized to [-1, +1] before weighting, so the composite
range is also [-1, +1]. Mapping:
  composite >  0.40  → bullish (high confidence)
  composite >  0.10  → bullish (low/medium confidence)
  composite > -0.10  → neutral
  composite > -0.40  → bearish (low/medium confidence)
  composite ≤ -0.40  → bearish (high confidence)
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from src.graph.state import AgentState
from src.tools.api import get_financial_metrics, get_prices
from src.agents.deterministic._helpers import (
    emit_signal,
    finalize,
    fmt_pct,
    for_each_ticker,
    get_api_key,
    safe_div,
)

WEIGHTS = {"value": 0.30, "quality": 0.30, "momentum": 0.20, "low_vol": 0.20}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def multi_factor_agent(state: AgentState, agent_id: str = "multi_factor_agent"):
    api_key = get_api_key(state)
    end_date_str = state["data"]["end_date"]
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    price_start = (end_date - timedelta(days=440)).strftime("%Y-%m-%d")

    def score_ticker(ticker: str) -> dict:
        metrics = get_financial_metrics(ticker, end_date_str, period="ttm", limit=1, api_key=api_key)
        m = metrics[0] if metrics else None

        # ─── Value sub-score: combine PE and FCF yield. ────────────────────────
        pe = getattr(m, "price_to_earnings_ratio", None)
        fcf_yield = getattr(m, "free_cash_flow_yield", None)
        # PE: 5 → +1, 30 → -1 (linear)
        if pe is not None and pe > 0:
            value_pe = _clip((20 - pe) / 15, -1, 1)
        else:
            value_pe = 0.0
        # FCF yield: 8% → +1, 0% → -1
        if fcf_yield is not None:
            value_fcf = _clip((fcf_yield - 0.02) / 0.06, -1, 1)
        else:
            value_fcf = 0.0
        value_score = (value_pe + value_fcf) / 2

        # ─── Quality sub-score: ROE + gross margin. ────────────────────────────
        roe = getattr(m, "return_on_equity", None)
        gross_margin = getattr(m, "gross_margin", None)
        # ROE 25% → +1, 5% → -1
        q_roe = _clip((roe - 0.10) / 0.15, -1, 1) if roe is not None else 0.0
        # Gross margin 50% → +1, 15% → -1
        q_gm = _clip((gross_margin - 0.30) / 0.20, -1, 1) if gross_margin is not None else 0.0
        quality_score = (q_roe + q_gm) / 2

        # ─── Momentum sub-score: 12-1 return. ──────────────────────────────────
        prices = get_prices(ticker, price_start, end_date_str, api_key=api_key)
        mom_score = 0.0
        mom_12_1 = None
        if len(prices) >= 280:
            end_idx = len(prices) - 1
            skip_idx = max(0, end_idx - 21)
            start_idx = max(0, skip_idx - 252)
            if prices[start_idx].close > 0 and prices[skip_idx].close > 0:
                mom_12_1 = prices[skip_idx].close / prices[start_idx].close - 1.0
                # 25% → +1, -25% → -1
                mom_score = _clip(mom_12_1 / 0.25, -1, 1)

        # ─── Low-vol sub-score: lower vol = higher score. ──────────────────────
        lv_score = 0.0
        ann_vol = None
        if len(prices) >= 61:
            window = prices[-61:]
            rets: list[float] = []
            for i in range(1, len(window)):
                p1, p2 = window[i - 1].close, window[i].close
                if p1 > 0 and p2 > 0:
                    rets.append(math.log(p2 / p1))
            if rets:
                mean_r = sum(rets) / len(rets)
                var = sum((r - mean_r) ** 2 for r in rets) / max(1, len(rets) - 1)
                ann_vol = math.sqrt(var) * math.sqrt(252)
                # Vol 15% → +1, 50% → -1 (lower vol better)
                lv_score = _clip((0.30 - ann_vol) / 0.15, -1, 1)

        composite = (
            WEIGHTS["value"] * value_score
            + WEIGHTS["quality"] * quality_score
            + WEIGHTS["momentum"] * mom_score
            + WEIGHTS["low_vol"] * lv_score
        )

        if composite > 0.40:
            signal, confidence = "bullish", 85
        elif composite > 0.10:
            signal, confidence = "bullish", 65
        elif composite > -0.10:
            signal, confidence = "neutral", 50
        elif composite > -0.40:
            signal, confidence = "bearish", 65
        else:
            signal, confidence = "bearish", 85

        reasoning = (
            f"Multi-factor composite: {composite:+.2f}\n"
            f"  Value    {value_score:+.2f} (P/E {pe if pe is not None else 'n/a'},"
            f" FCF yld {fmt_pct(fcf_yield)})\n"
            f"  Quality  {quality_score:+.2f} (ROE {fmt_pct(roe)},"
            f" GM {fmt_pct(gross_margin)})\n"
            f"  Momentum {mom_score:+.2f}"
            f" (12-1: {fmt_pct(mom_12_1)})\n"
            f"  Low-vol  {lv_score:+.2f} (ann vol: {fmt_pct(ann_vol)})\n"
            f"  Weights: value 30% / quality 30% / momentum 20% / low-vol 20%"
        )

        return emit_signal(
            state, agent_id, ticker,
            signal=signal, confidence=confidence,
            reasoning=reasoning,
            extras={
                "composite": composite,
                "value": value_score,
                "quality": quality_score,
                "momentum": mom_score,
                "low_vol": lv_score,
            },
        )

    signals = for_each_ticker(state, agent_id, score_ticker)
    return finalize(state, agent_id, "Multi-Factor Composite", signals)
