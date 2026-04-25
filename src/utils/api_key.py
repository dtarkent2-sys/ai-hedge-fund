import os


def get_api_key_from_state(state: dict, api_key_name: str) -> str | None:
    """Resolve an API key for the current run.

    Lookup order:
      1) request.api_keys (set by the web frontend / supplied in API call)
      2) os.environ (loaded from .env at backend startup)

    Returning the env fallback means the user only has to set FINANCIAL_DATASETS_API_KEY,
    OLLAMA_API_KEY, etc. in .env once — they don't need to retype them in the
    frontend's API-keys panel.
    """
    if state and state.get("metadata", {}).get("request"):
        request = state["metadata"]["request"]
        if hasattr(request, "api_keys") and request.api_keys:
            value = request.api_keys.get(api_key_name)
            if value:
                return value
    return os.environ.get(api_key_name)