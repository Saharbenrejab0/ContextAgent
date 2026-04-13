"""
Prompt module.
Builds the structured messages sent to the LLM.
Separates system instructions, context, memory, and question.
"""

SYSTEM_PROMPT = """You are ContextAgent, a precise technical document assistant.

Your job is to answer questions based ONLY on the provided context.
The context comes from real documents the user has uploaded.

Rules you must follow:
- Answer using ONLY information found in the context below.
- Always cite your sources using the format: (Source: filename)
- If multiple sources support the answer, cite all of them.
- If the context does not contain enough information to answer, say exactly:
  "I could not find a clear answer in the provided documents."
- Do not make up information. Do not use outside knowledge.

How to structure your answer:
- Be complete — cover all key points mentioned in the context.
- Use the exact technical terms from the context (function names, keywords, concepts).
- If the answer has multiple distinct parts, use a numbered list or bullet points.
- Keep answers focused and precise — no filler sentences.
- Always end with the source citation.
"""


def build_messages(
    question: str,
    context: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Build the full messages list for the OpenAI chat API.

    The structure is:
        [system]  → instructions + context
        [user]    → question 1       (from history)
        [assistant] → answer 1       (from history)
        ...
        [user]    → current question

    Args:
        question: the current user question
        context:  formatted context string from format_context()
        history:  list of past turns [{"role": "user/assistant", "content": "..."}]

    Returns:
        list of message dicts ready for the OpenAI API
    """
    system_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT FROM DOCUMENTS ===\n\n"
        f"{context}\n\n"
        f"=== END OF CONTEXT ==="
    )

    messages = [{"role": "system", "content": system_content}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": question})

    return messages


def build_standalone_question(
    question: str,
    history: list[dict],
) -> str:
    """
    When conversation history exists, the user's question might
    reference previous turns (e.g. "what about that first point?").

    This function rewrites the question to be self-contained
    by appending a short history summary.

    Used by the orchestrator before calling retrieve().

    Args:
        question: current user question
        history:  recent conversation history

    Returns:
        a reformulated standalone question string
    """
    if not history:
        return question

    recent = history[-4:]
    summary_parts = []

    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        summary_parts.append(f"{role}: {msg['content'][:200]}")

    summary = "\n".join(summary_parts)

    return (
        f"Given this recent conversation:\n{summary}\n\n"
        f"Answer this follow-up question as a standalone query: {question}"
    )


if __name__ == "__main__":
    print("Testing prompt builder...\n")

    sample_context = (
        "[Source 1: classes.txt]\n"
        "Classes provide a means of bundling data and functionality together. "
        "Creating a new class creates a new type of object.\n\n"
        "---\n\n"
        "[Source 2: classes.txt]\n"
        "A class definition looks like this: class ClassName: ..."
    )

    sample_history = [
        {"role": "user",      "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language."},
    ]

    messages = build_messages(
        question="How do you define a class?",
        context=sample_context,
        history=sample_history,
    )

    print(f"  Total messages built: {len(messages)}\n")

    for i, msg in enumerate(messages):
        role    = msg["role"].upper()
        preview = msg["content"][:120].replace("\n", " ")
        print(f"  [{i}] {role}: {preview}...")
        print()

    standalone = build_standalone_question(
        question="Can you give an example?",
        history=sample_history,
    )

    print(f"  Standalone question:\n  {standalone}")