"""Deterministic factor agents.

Each agent in this package is pure math (Piotroski F-Score, Greenblatt Magic
Formula, momentum, low-vol, multi-factor composite, …). They share the same
node signature as the LLM personas — they read from `state.data`, write a
signal into `state.data.analyst_signals[<agent_id>]`, and emit a HumanMessage —
so the LangGraph workflow treats them identically. The difference is they
NEVER call an LLM; reasoning is plain text built from the rule outputs.
"""

from src.agents.deterministic.piotroski import piotroski_agent
from src.agents.deterministic.greenblatt import greenblatt_agent
from src.agents.deterministic.momentum import momentum_agent
from src.agents.deterministic.low_volatility import low_volatility_agent
from src.agents.deterministic.acquirers_multiple import acquirers_multiple_agent
from src.agents.deterministic.multi_factor import multi_factor_agent
from src.agents.deterministic.earnings_revision import earnings_revision_agent
from src.agents.deterministic.mohanram import mohanram_agent
from src.agents.deterministic.dividend_aristocrat import dividend_aristocrat_agent
from src.agents.deterministic.etf_profile import etf_profile_agent
from src.agents.deterministic.dashan_huang import dashan_huang_agent

__all__ = [
    "piotroski_agent",
    "greenblatt_agent",
    "momentum_agent",
    "low_volatility_agent",
    "acquirers_multiple_agent",
    "multi_factor_agent",
    "earnings_revision_agent",
    "mohanram_agent",
    "dividend_aristocrat_agent",
    "etf_profile_agent",
    "dashan_huang_agent",
]
