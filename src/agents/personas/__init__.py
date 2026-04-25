"""Validea-style LLM personas.

Each persona file pairs its model's specific scoring criteria (deterministic
prelim) with a prompt template in the investor's voice. The shared
`build_persona_agent` factory in `_factory.py` handles the LangGraph node
plumbing, line-item fetch, and call_llm boilerplate.
"""
from src.agents.personas.kenneth_fisher import kenneth_fisher_agent
from src.agents.personas.oshaughnessy import oshaughnessy_agent
from src.agents.personas.zweig import martin_zweig_agent
from src.agents.personas.joel_greenblatt import joel_greenblatt_agent
from src.agents.personas.john_neff import john_neff_agent
from src.agents.personas.david_dreman import david_dreman_agent
from src.agents.personas.motley_fool import motley_fool_agent
from src.agents.personas.wesley_gray import wesley_gray_agent
from src.agents.personas.meb_faber import meb_faber_agent

__all__ = [
    "kenneth_fisher_agent",
    "oshaughnessy_agent",
    "martin_zweig_agent",
    "joel_greenblatt_agent",
    "john_neff_agent",
    "david_dreman_agent",
    "motley_fool_agent",
    "wesley_gray_agent",
    "meb_faber_agent",
]
