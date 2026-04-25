"""Helper functions for LLM"""

import json
import os
import sys
import traceback
from pydantic import BaseModel
from src.llm.models import get_model, get_model_info
from src.utils.progress import progress
from src.graph.state import AgentState


class RunCancelledError(Exception):
    """Raised when an in-flight run is cancelled by the SSE generator after
    the client disconnects. Distinct from generic LLM errors so the SSE
    layer can surface a clean 'cancelled' event instead of 'error'."""


def _system_default_model() -> tuple[str, str]:
    """Pick a sensible default when the caller gave us no model config.

    If OLLAMA_API_KEY is set, point to Ollama Cloud so stray default-routed
    agents (e.g. portfolio_manager when the frontend didn't wire a model for
    every node) don't silently try to call OpenAI with a placeholder key.

    Users can override via AIHF_DEFAULT_MODEL / AIHF_DEFAULT_PROVIDER.
    """
    name = os.environ.get("AIHF_DEFAULT_MODEL")
    prov = os.environ.get("AIHF_DEFAULT_PROVIDER")
    if name and prov:
        return name, prov
    if os.environ.get("OLLAMA_API_KEY"):
        return "gpt-oss:20b", "Ollama"
    return "gpt-4.1", "OpenAI"


def call_llm(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: AgentState | None = None,
    max_retries: int = 3,
    default_factory=None,
) -> BaseModel:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates and model config extraction
        state: Optional state object to extract agent-specific model configuration
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """
    
    # Extract model configuration if state is provided and agent_name is available
    if state and agent_name:
        model_name, model_provider = get_agent_model_config(state, agent_name)
    else:
        # Use system defaults when no state or agent_name is provided.
        # _system_default_model() returns the right pair based on env:
        # OLLAMA_API_KEY set → ("gpt-oss:20b", "Ollama"), else
        # ("gpt-4.1", "OpenAI"). AIHF_DEFAULT_MODEL/AIHF_DEFAULT_PROVIDER
        # override either path. (The earlier hard-coded "glm-5.1:cloud" /
        # "Ollama" pair is now expressible via those env vars without a
        # code change.)
        model_name, model_provider = _system_default_model()

    # Extract API keys from state if available
    api_keys = None
    if state:
        request = state.get("metadata", {}).get("request")
        if request and hasattr(request, 'api_keys'):
            api_keys = request.api_keys

    model_info = get_model_info(model_name, model_provider)
    llm = get_model(model_name, model_provider, api_keys)

    # For non-JSON support models, we can use structured output
    if not (model_info and not model_info.has_json_mode()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )

    # If the run was cancelled (client disconnected), bail out before
    # spending any more LLM tokens / Ollama Cloud quota on it.
    cancel_event = (state or {}).get("metadata", {}).get("cancel_event") if state else None
    if cancel_event is not None and cancel_event.is_set():
        raise RunCancelledError(f"Run cancelled before {agent_name or 'LLM'} call")

    # Call the LLM with retries
    last_error: Exception | None = None
    for attempt in range(max_retries):
        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelledError(f"Run cancelled mid-retry for {agent_name or 'LLM'}")
        try:
            # Call the LLM
            result = llm.invoke(prompt)

            # For non-JSON support models, we need to extract and parse the JSON manually
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
            else:
                return result

        except Exception as e:
            last_error = e
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")

            if attempt == max_retries - 1:
                # Loud, fully-contextual failure so bad model/provider settings surface immediately
                label = agent_name or "<unknown agent>"
                banner = "!" * 72
                print(
                    f"\n{banner}\n"
                    f"LLM call FAILED for agent={label}\n"
                    f"  model    = {model_name}\n"
                    f"  provider = {model_provider}\n"
                    f"  attempts = {max_retries}\n"
                    f"  error    = {type(e).__name__}: {e}\n"
                    f"{banner}",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc(file=sys.stderr)
                # Still return a default so one flaky agent doesn't tank the whole run —
                # but the failure is now impossible to miss in the output.
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)


def create_default_response(model_class: type[BaseModel]) -> BaseModel:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> dict | None:
    """Recover a JSON object from an LLM response.

    Tries, in order:
      1) Whole-string parse (when the model returned pure JSON).
      2) ```json … ``` fenced block.
      3) ``` … ``` generic fenced block.
      4) First balanced {...} span in the text.
    Returns None if nothing parses.
    """
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None

    # 1) whole string
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) ```json fence
    if "```json" in text:
        try:
            body = text.split("```json", 1)[1].split("```", 1)[0].strip()
            obj = json.loads(body)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3) generic ``` fence
    if "```" in text:
        try:
            body = text.split("```", 1)[1].split("```", 1)[0].strip()
            # strip an optional leading language tag like "json\n"
            if "\n" in body and body.split("\n", 1)[0].isalpha():
                body = body.split("\n", 1)[1]
            obj = json.loads(body)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 4) first balanced-brace span
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start:i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
        start = text.find("{", start + 1)

    return None


def get_agent_model_config(state, agent_name):
    """
    Get model configuration for a specific agent from the state.
    Falls back to global model configuration if agent-specific config is not available.
    Always returns valid model_name and model_provider values.
    """
    request = state.get("metadata", {}).get("request")
    
    if request and hasattr(request, 'get_agent_model_config'):
        # Get agent-specific model configuration
        model_name, model_provider = request.get_agent_model_config(agent_name)
        # Ensure we have valid values
        if model_name and model_provider:
            return model_name, model_provider.value if hasattr(model_provider, 'value') else str(model_provider)
    
    # Fall back to global configuration (system defaults).
    default_name, default_provider = _system_default_model()
    model_name = state.get("metadata", {}).get("model_name") or default_name
    model_provider = state.get("metadata", {}).get("model_provider") or default_provider
    
    # Convert enum to string if necessary
    if hasattr(model_provider, 'value'):
        model_provider = model_provider.value
    
    return model_name, model_provider
