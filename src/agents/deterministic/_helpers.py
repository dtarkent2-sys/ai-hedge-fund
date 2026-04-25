"""Shared helpers for deterministic agents.

Every deterministic agent follows the same shape:
  1. Pull what it needs from `state.data` and the data layer (api.py).
  2. Compute a numeric score.
  3. Translate the score into {signal, confidence, reasoning} via fixed thresholds.
  4. Write into `state.data.analyst_signals[<agent_id>][ticker]`.
  5. Emit a HumanMessage.

This file holds the boilerplate common to all of them so each agent file is
just the rule logic.
"""
from __future__ import annotations

import json
from typing import Callable, Iterable
from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.api_key import get_api_key_from_state
from src.utils.progress import progress


def to_signal(score: float, *, bullish_at: float, bearish_at: float) -> str:
    """Map a score onto bullish/neutral/bearish given two thresholds."""
    if score >= bullish_at:
        return "bullish"
    if score <= bearish_at:
        return "bearish"
    return "neutral"


def to_confidence(score: float, *, lo: float, hi: float, floor: int = 35, ceil: int = 90) -> int:
    """Linearly map a score in [lo, hi] to a confidence in [floor, ceil]."""
    if hi <= lo:
        return floor
    pct = (score - lo) / (hi - lo)
    pct = max(0.0, min(1.0, pct))
    return int(round(floor + pct * (ceil - floor)))


def safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    try:
        return a / b
    except ZeroDivisionError:
        return None


def fmt_money(v: float | None) -> str:
    if v is None:
        return "n/a"
    if abs(v) >= 1e9:
        return f"${v/1e9:.1f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:.1f}k"
    return f"${v:.2f}"


def fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v*100:.1f}%"


def get_api_key(state: AgentState) -> str | None:
    return get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")


def emit_signal(
    state: AgentState,
    agent_id: str,
    ticker: str,
    signal: str,
    confidence: int,
    reasoning: str,
    extras: dict | None = None,
) -> dict:
    """Build the standard signal dict and stash it on state.

    Deterministic agents may also surface raw factor inputs under `extras` so
    the dashboard can show numbers behind the score.
    """
    payload: dict = {"signal": signal, "confidence": confidence, "reasoning": reasoning}
    if extras:
        payload["factors"] = extras
    state["data"].setdefault("analyst_signals", {}).setdefault(agent_id, {})[ticker] = payload
    return payload


def finalize(
    state: AgentState,
    agent_id: str,
    label: str,
    signals: dict,
) -> dict:
    """Show reasoning if requested + emit the HumanMessage every node needs."""
    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(signals, label)
    progress.update_status(agent_id, None, "Done")
    return {
        "messages": [HumanMessage(content=json.dumps(signals), name=agent_id)],
        "data": state["data"],
    }


def for_each_ticker(
    state: AgentState,
    agent_id: str,
    body: Callable[[str], dict],
) -> dict:
    """Run `body(ticker)` for every ticker, register progress, return a signals dict."""
    signals: dict[str, dict] = {}
    for ticker in state["data"]["tickers"]:
        progress.update_status(agent_id, ticker, "Computing")
        try:
            signals[ticker] = body(ticker)
        except Exception as exc:  # never let one ticker take down the run
            progress.update_status(agent_id, ticker, f"Error: {type(exc).__name__}")
            signals[ticker] = {
                "signal": "neutral",
                "confidence": 30,
                "reasoning": f"Computation failed: {type(exc).__name__}: {exc}",
            }
        progress.update_status(agent_id, ticker, "Done")
    return signals


def latest_and_prior(seq: Iterable, n: int = 2) -> tuple:
    """Return (latest, prior) from a newest-first iterable, padding with None."""
    items = list(seq)
    latest = items[0] if len(items) > 0 else None
    prior = items[1] if len(items) > 1 else None
    return latest, prior
