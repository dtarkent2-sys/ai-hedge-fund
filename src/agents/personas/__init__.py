"""Validea-style LLM personas.

Each persona file pairs its model's specific scoring criteria (deterministic
prelim) with a prompt template in the investor's voice. The shared
`build_persona_agent` factory in `_factory.py` handles the LangGraph node
plumbing, line-item fetch, and call_llm boilerplate.
"""
from src.agents.personas.kenneth_fisher import kenneth_fisher_agent
from src.agents.personas.motley_fool import motley_fool_agent

__all__ = [
    "kenneth_fisher_agent",
    "motley_fool_agent",
]
