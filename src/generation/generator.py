"""
Generator module.
Sends the built messages to GPT-4o-mini and returns
a structured response with the answer, sources, and token usage.
"""

import os
import sys
sys.path.insert(0, ".")

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL       = "gpt-4o-mini"
MAX_TOKENS      = 1000
TEMPERATURE     = 0.1


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Make sure it is set in your .env file."
        )
    return OpenAI(api_key=api_key)


def generate(messages: list[dict]) -> dict:
    """
    Send messages to the LLM and return a structured response.

    Args:
        messages: list of message dicts from build_messages()

    Returns:
        {
            "answer":  the LLM response text,
            "sources": list of unique source filenames cited,
            "model":   model used,
            "tokens":  {"prompt": N, "completion": N, "total": N}
        }
    """
    client   = get_client()

    response = client.chat.completions.create(
        model       = LLM_MODEL,
        messages    = messages,
        max_tokens  = MAX_TOKENS,
        temperature = TEMPERATURE,
    )

    answer = response.choices[0].message.content.strip()
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


def _extract_sources(messages: list[dict]) -> list[str]:
    """
    Extract unique source filenames from the system message context.
    Looks for the [Source N: filename] pattern injected by format_context().

    Args:
        messages: the full messages list

    Returns:
        list of unique filenames found in the context
    """
    import re

    system_content = messages[0]["content"] if messages else ""
    pattern        = r"\[Source \d+: (.+?)\]"
    matches        = re.findall(pattern, system_content)

    return list(dict.fromkeys(matches))


if __name__ == "__main__":
    from src.retrieval.retriever     import retrieve, format_context
    from src.generation.prompt       import build_messages

    print("Testing generator...\n")

    queries = [
        "What is a Python class and how do you define one?",
        "How does error handling work in Python?",
    ]

    for query in queries:
        print(f"  Query: '{query}'")
        print("-" * 50)

        chunks   = retrieve(query, top_k=3)
        context  = format_context(chunks)
        messages = build_messages(question=query, context=context)
        result   = generate(messages)

        print(f"  Answer:\n  {result['answer']}\n")
        print(f"  Sources : {result['sources']}")
        print(f"  Tokens  : {result['tokens']}")
        print(f"  Model   : {result['model']}")
        print("\n" + "=" * 50 + "\n")