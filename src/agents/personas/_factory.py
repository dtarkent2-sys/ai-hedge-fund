"""Shared scaffolding for Validea-style LLM personas.

Each persona supplies:
  • `LINE_ITEMS`   — the financial line items its scoring needs
  • `score(ticker, metrics, line_items, prices) -> dict`
        ↳ deterministic prelim scoring; the dict is embedded into the LLM prompt
  • `prompt_template` (langchain ChatPromptTemplate) — the persona's voice

The factory wraps these into a LangGraph node that:
  1. Fetches financial_metrics + line_items + (optionally) recent prices
  2. Calls `score(...)` to compute the deterministic analysis
  3. Builds the prompt + invokes call_llm with the standard PersonaSignal model
  4. Stores the signal in state.data.analyst_signals[<agent_id>][ticker]
"""
from __future__ import annotations

import json
from typing import Callable, Iterable, Optional, Sequence
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_prices, search_line_items
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress


class PersonaSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float        # 0-100
    reasoning: str


ScoreFn = Callable[[str, list, list, list, AgentState], dict]


def build_persona_agent(
    *,
    agent_id: str,
    label: str,
    line_items: Sequence[str],
    prompt_template: ChatPromptTemplate,
    score: ScoreFn,
    prices_window_days: int = 0,
    period: str = "ttm",
    metrics_limit: int = 5,
    line_items_limit: int = 5,
):
    """Return a LangGraph node function implementing a Validea persona."""

    def agent(state: AgentState, agent_id: str = agent_id):
        api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
        end_date = state["data"]["end_date"]
        tickers = state["data"]["tickers"]
        signals: dict[str, dict] = {}

        for ticker in tickers:
            progress.update_status(agent_id, ticker, f"Fetching data for {label}")
            try:
                metrics = get_financial_metrics(
                    ticker, end_date, period=period, limit=metrics_limit, api_key=api_key,
                )
                items = search_line_items(
                    ticker, list(line_items), end_date,
                    period=period, limit=line_items_limit, api_key=api_key,
                )
                prices: list = []
                if prices_window_days > 0:
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    start_dt = (end_dt - timedelta(days=prices_window_days)).strftime("%Y-%m-%d")
                    prices = get_prices(ticker, start_dt, end_date, api_key=api_key)

                progress.update_status(agent_id, ticker, "Scoring")
                analysis = score(ticker, metrics, items, prices, state)

                progress.update_status(agent_id, ticker, f"Generating {label} narrative")

                def default_signal() -> PersonaSignal:
                    return PersonaSignal(
                        signal="neutral", confidence=40,
                        reasoning="LLM call failed; defaulting to neutral. Prelim analysis embedded in extras.",
                    )

                prompt = prompt_template.invoke({
                    "ticker": ticker,
                    "analysis_data": json.dumps(analysis, indent=2, default=str),
                })
                output: PersonaSignal = call_llm(
                    prompt=prompt,
                    pydantic_model=PersonaSignal,
                    agent_name=agent_id,
                    state=state,
                    default_factory=default_signal,
                )
                signals[ticker] = output.model_dump() if hasattr(output, "model_dump") else dict(output)
                signals[ticker]["analysis"] = analysis  # surface the deterministic prelim too
                progress.update_status(agent_id, ticker, "Done", analysis=output.reasoning)
            except Exception as exc:
                progress.update_status(agent_id, ticker, f"Error: {type(exc).__name__}")
                signals[ticker] = {
                    "signal": "neutral", "confidence": 30,
                    "reasoning": f"Failure in {label}: {type(exc).__name__}: {exc}",
                }

        if state.get("metadata", {}).get("show_reasoning"):
            show_agent_reasoning(signals, label)
        state["data"].setdefault("analyst_signals", {})[agent_id] = signals
        progress.update_status(agent_id, None, "Done")
        return {
            "messages": [HumanMessage(content=json.dumps(signals, default=str), name=agent_id)],
            "data": state["data"],
        }

    return agent


# Common line-item bundles many personas reuse
COMMON_LINE_ITEMS = [
    "revenue", "net_income", "free_cash_flow", "operating_income",
    "earnings_per_share", "outstanding_shares", "total_debt",
    "shareholders_equity", "current_assets", "current_liabilities",
    "total_assets", "gross_profit", "research_and_development",
    "dividends_and_other_cash_distributions", "issuance_or_purchase_of_equity_shares",
    "ebit", "ebitda",
]


def gv(o, a, default=None):
    """Defensive getattr — None-safe."""
    return getattr(o, a, default) if o is not None else default
