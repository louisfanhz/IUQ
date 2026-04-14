import time
from typing import Dict, Any, Optional, List
from openai import OpenAI
from together import Together

from credentials import openai_api_key, together_api_key

_openai_client: Optional[OpenAI] = None
_together_client: Optional[Together] = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=openai_api_key)
    return _openai_client


def get_together_client() -> Together:
    global _together_client
    if _together_client is None:
        _together_client = Together(api_key=together_api_key)
    return _together_client


def chat_completion(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1000,
    top_p: float = 1.0,
    logprobs: bool = False,
    response_format=None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> Dict[str, Any]:
    """Make a single chat completion request.

    Returns dict with keys: response, tokens, logprobs, usage.
    """
    for attempt in range(max_retries):
        try:
            return _call(provider, model, messages, temperature, max_tokens,
                         top_p, logprobs, response_format)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  API error (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                raise


def _call(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    logprobs: bool,
    response_format=None,
) -> Dict[str, Any]:
    if provider == "openai":
        client = get_openai_client()
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        if response_format is not None:
            kwargs["response_format"] = response_format
    elif provider == "togetherai":
        client = get_together_client()
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if response_format is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__.lower(),
                    "schema": response_format.model_json_schema(),
                },
            }
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if logprobs:
        kwargs["logprobs"] = True

    if provider == "openai" and response_format is not None:
        response = client.chat.completions.parse(**kwargs)
    else:
        response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]

    result: Dict[str, Any] = {
        "response": choice.message.content.strip() if choice.message.content else "",
        "tokens": None,
        "logprobs": None,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        },
    }

    if logprobs and choice.logprobs:
        if provider == "openai" and getattr(choice.logprobs, "content", None):
            result["tokens"] = [c.token for c in choice.logprobs.content]
            result["logprobs"] = [c.logprob for c in choice.logprobs.content]
        elif provider == "togetherai":
            if getattr(choice.logprobs, "tokens", None):
                result["tokens"] = choice.logprobs.tokens
                result["logprobs"] = choice.logprobs.token_logprobs

    return result
