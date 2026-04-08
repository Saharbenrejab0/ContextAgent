"""
Generator module — v2.
Two modes:
  - generate()        : standard — returns full response dict
  - generate_stream() : streaming — yields tokens as they arrive (SSE)
"""

import os
import sys
import re
sys.path.insert(0, ".")

from openai import OpenAI
from dotenv import load_dotenv
from langsmith import traceable 
load_dotenv()

LLM_MODEL   = "gpt-4o-mini"
MAX_TOKENS  = 1000
TEMPERATURE = 0.1


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")
    return OpenAI(api_key=api_key)


def generate(messages: list[dict]) -> dict:
    """Standard generation — waits for full response."""
    client   = get_client()
    response = client.chat.completions.create(
        model       = LLM_MODEL,
        messages    = messages,
        max_tokens  = MAX_TOKENS,
        temperature = TEMPERATURE,
    )
    answer  = response.choices[0].message.content.strip()
    sources = _extract_sources(messages)
    return {
        "answer":  answer,
        "sources": sources,
        "model":   LLM_MODEL,
        "tokens": {
            "prompt":     response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total":      response.usage.total_tokens,
        },
    }


def generate_stream(messages: list[dict]):
    """
    Streaming generation — yields tokens as they arrive.
    Used by the FastAPI SSE endpoint.

    Yields strings: each token as it arrives, then a final
    JSON-encoded metadata object prefixed with [DONE].
    """
    client = get_client()
    stream = client.chat.completions.create(
        model       = LLM_MODEL,
        messages    = messages,
        max_tokens  = MAX_TOKENS,
        temperature = TEMPERATURE,
        stream      = True,
    )

    full_answer    = ""
    prompt_tokens  = 0
    total_tokens   = 0

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            full_answer += delta.content
            yield delta.content

        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            total_tokens  = chunk.usage.total_tokens

    sources = _extract_sources(messages)
    import json
    yield f"[DONE]{json.dumps({'sources': sources, 'model': LLM_MODEL, 'tokens': {'total': total_tokens, 'prompt': prompt_tokens}})}"


def _extract_sources(messages: list[dict]) -> list[str]:
    system_content = messages[0]["content"] if messages else ""
    matches        = re.findall(r"\[Source \d+: (.+?)\]", system_content)
    return list(dict.fromkeys(matches))