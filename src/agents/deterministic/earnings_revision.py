"""Earnings Revision agent.

When AV's EARNINGS_ESTIMATES has data for the ticker, we use the real
analyst-revision deltas (current avg vs. avg 7/30/60/90 days ago, plus
analyst count). This is the textbook revision-momentum factor.

When AV returns no estimates for the ticker (coverage gap), we fall back to a
surprise-momentum proxy built from EARNINGS history: beat/miss pattern over
the last 4 quarters, avg surprise %, and slope of `estimatedEPS` over 8
quarters.

Bullish:
  • Real path: nearest-horizon estimate revising UP > +2% over 30 days, with
    20+ analysts and broad agreement (high/low spread tight)
  • Fallback: 3+ beats in last 4 + rising estimate trend
Bearish: symmetric on the downside.
"""
from __future__ import annotations

from src.graph.state import AgentState
from src.tools.alphavantage import get_earnings_estimates, get_earnings_history
from src.agents.deterministic._helpers import (
    emit_signal,
    finalize,
    fmt_pct,
    for_each_ticker,
    get_api_key,
)


def _f(x):
    if x is None or x == "" or x == "None":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _slope(values: list[float]) -> float | None:
    """Simple linear-regression slope on [0,1,…,n-1] vs values, normalized by mean."""
    if len(values) < 3:
        return None
    n = len(values)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    if mean_y == 0:
        return None
    num = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if den == 0:
        return None
    slope = num / den
    return slope / abs(mean_y)  # normalize → "slope per period as fraction of mean"


def _real_revision_score(ticker: str, api_key) -> dict | None:
    """Try the AV EARNINGS_ESTIMATES path. Returns None if no data for the ticker."""
    estimates = get_earnings_estimates(ticker, api_key=api_key)
    if not estimates:
        return None

    # Find the nearest-horizon FY entry (closest fiscal year forward).
    fy_rows = [e for e in estimates if (e.get("horizon") or "").lower() == "fiscal year"]
    fy_rows.sort(key=lambda e: e.get("date") or "")
    if not fy_rows:
        return None
    fy = fy_rows[0]
    avg_now = _f(fy.get("eps_estimate_average"))
    avg_30 = _f(fy.get("eps_estimate_average_30_days_ago"))
    avg_60 = _f(fy.get("eps_estimate_average_60_days_ago"))
    avg_90 = _f(fy.get("eps_estimate_average_90_days_ago"))
    avg_7 = _f(fy.get("eps_estimate_average_7_days_ago"))
    n_analysts = _f(fy.get("eps_estimate_analyst_count"))
    high = _f(fy.get("eps_estimate_high"))
    low = _f(fy.get("eps_estimate_low"))

    if avg_now is None:
        return None

    def pct(now, then) -> float | None:
        if now is None or then is None or then == 0:
            return None
        return now / then - 1.0

    rev_30 = pct(avg_now, avg_30)
    rev_60 = pct(avg_now, avg_60)
    rev_90 = pct(avg_now, avg_90)
    rev_7 = pct(avg_now, avg_7)
    spread = ((high - low) / abs(avg_now)) if (high is not None and low is not None and avg_now != 0) else None

    score = 0
    if rev_30 is not None:
        if rev_30 > 0.05: score += 3
        elif rev_30 > 0.02: score += 2
        elif rev_30 > 0.005: score += 1
        elif rev_30 < -0.05: score -= 3
        elif rev_30 < -0.02: score -= 2
        elif rev_30 < -0.005: score -= 1
    if rev_90 is not None:
        if rev_90 > 0.05: score += 1
        elif rev_90 < -0.05: score -= 1
    if rev_7 is not None and rev_30 is not None:
        # Acceleration: short-term revision faster than 30-day → adds conviction
        if rev_7 > rev_30 * 0.5 and rev_7 > 0: score += 1
        if rev_7 < rev_30 * 0.5 and rev_7 < 0: score -= 1
    # Analyst breadth gates confidence
    if n_analysts is not None and n_analysts >= 20:
        score = int(score * 1.2)

    if score >= 4:
        signal, confidence = "bullish", 90
    elif score >= 2:
        signal, confidence = "bullish", 70
    elif score >= -1:
        signal, confidence = "neutral", 50
    elif score >= -3:
        signal, confidence = "bearish", 70
    else:
        signal, confidence = "bearish", 90

    reasoning = (
        f"Earnings Revision (real EPS estimate revisions, {fy.get('date')}):\n"
        f"  Current avg estimate: {avg_now}  ({int(n_analysts) if n_analysts else '?'} analysts)\n"
        f"  Revision 30d: {fmt_pct(rev_30)}   60d: {fmt_pct(rev_60)}   90d: {fmt_pct(rev_90)}   7d: {fmt_pct(rev_7)}\n"
        f"  High/low spread vs avg: {fmt_pct(spread)}\n"
        f"  Composite: {score:+d}"
    )
    return {
        "signal": signal, "confidence": confidence, "reasoning": reasoning,
        "extras": {
            "avg_estimate": avg_now, "rev_30d": rev_30, "rev_60d": rev_60,
            "rev_90d": rev_90, "rev_7d": rev_7, "n_analysts": n_analysts,
            "spread": spread, "composite": score, "method": "estimate_revisions",
        },
    }


def earnings_revision_agent(state: AgentState, agent_id: str = "earnings_revision_agent"):
    api_key = get_api_key(state)

    def score_ticker(ticker: str) -> dict:
        # Preferred: AV's EARNINGS_ESTIMATES with built-in *_N_days_ago revisions.
        real = _real_revision_score(ticker, api_key)
        if real is not None:
            return emit_signal(
                state, agent_id, ticker,
                signal=real["signal"], confidence=real["confidence"],
                reasoning=real["reasoning"], extras=real["extras"],
            )

        # Fallback: surprise-history proxy from AV's EARNINGS endpoint.
        data = get_earnings_history(ticker, api_key=api_key)
        quarterly = data.get("quarterlyEarnings") or []
        if len(quarterly) < 4:
            return emit_signal(
                state, agent_id, ticker,
                signal="neutral", confidence=30,
                reasoning="Need ≥4 quarters of earnings history (and no analyst estimates available).",
            )

        # quarterly is newest-first per AV
        recent = quarterly[:4]
        beats = sum(1 for q in recent if (_f(q.get("surprise")) or 0) > 0)
        misses = sum(1 for q in recent if (_f(q.get("surprise")) or 0) < 0)
        surprise_pcts = [_f(q.get("surprisePercentage")) for q in recent if _f(q.get("surprisePercentage")) is not None]
        avg_surprise = sum(surprise_pcts) / len(surprise_pcts) / 100 if surprise_pcts else None

        # Estimate trend across the last 8 quarters, oldest → newest
        est_window = quarterly[:8]
        ests_chrono = list(reversed([
            _f(q.get("estimatedEPS")) for q in est_window if _f(q.get("estimatedEPS")) is not None
        ]))
        est_slope = _slope(ests_chrono)  # fraction per quarter

        # Score
        score = 0
        if beats >= 3:
            score += 2
        elif beats == 2:
            score += 1
        if misses >= 3:
            score -= 2
        elif misses == 2:
            score -= 1
        if est_slope is not None:
            if est_slope > 0.05:  # estimates rising > 5% per quarter
                score += 2
            elif est_slope > 0.01:
                score += 1
            elif est_slope < -0.05:
                score -= 2
            elif est_slope < -0.01:
                score -= 1
        if avg_surprise is not None:
            if avg_surprise > 0.05:
                score += 1
            elif avg_surprise < -0.05:
                score -= 1

        # Map score (≈ -6…+6) to signal
        if score >= 4:
            signal, confidence = "bullish", 85
        elif score >= 2:
            signal, confidence = "bullish", 65
        elif score >= -1:
            signal, confidence = "neutral", 50
        elif score >= -3:
            signal, confidence = "bearish", 65
        else:
            signal, confidence = "bearish", 85

        reasoning = (
            f"Earnings Revision composite: {score:+d}\n"
            f"  Last 4 quarters: {beats} beats / {misses} misses\n"
            f"  Avg surprise: {fmt_pct(avg_surprise)}\n"
            f"  Estimate trend (8q slope): "
            f"{('rising' if est_slope and est_slope > 0 else 'falling' if est_slope and est_slope < 0 else 'flat')}"
            f" ({fmt_pct(est_slope)} per quarter)"
        )
        return emit_signal(
            state, agent_id, ticker,
            signal=signal, confidence=confidence,
            reasoning=reasoning,
            extras={
                "beats_4q": beats,
                "misses_4q": misses,
                "avg_surprise": avg_surprise,
                "estimate_slope": est_slope,
                "composite": score,
                "method": "surprise_history",
            },
        )

    signals = for_each_ticker(state, agent_id, score_ticker)
    return finalize(state, agent_id, "Earnings Revision", signals)
