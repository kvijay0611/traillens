"""
Thin wrapper so the rest of the app doesn't care which LLM provider is
behind it. Groq, Together, Cerebras, and OpenRouter all expose an
OpenAI-compatible /chat/completions endpoint, so swapping providers is
just changing LLM_BASE_URL + LLM_API_KEY + LLM_MODEL in .env -- no code
changes needed elsewhere in the app.
"""
from openai import OpenAI
from backend.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client


def complete(system: str, user: str, max_tokens: int = 800) -> tuple[str, dict]:
    """Returns (response_text, {"input_tokens": int, "output_tokens": int})."""
    client = get_client()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content.strip()
    usage = {
        "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
        "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
    }
    return text, usage
