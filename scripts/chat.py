"""
Chat script.
Interactive CLI interface for ContextAgent.
Talks to the Orchestrator and displays responses cleanly.

Usage:
    python scripts/chat.py
"""

import sys
sys.path.insert(0, ".")

from src.agent.orchestrator import Orchestrator


DIVIDER     = "─" * 60
COMMANDS    = """
  Commands:
    /reset    → clear conversation memory
    /history  → show conversation so far
    /sources  → show chunks used in last answer
    /quit     → exit
"""


def print_welcome() -> None:
    print("\n" + "=" * 60)
    print("  ContextAgent — Document Assistant")
    print("  Ask questions about your documents.")
    print(COMMANDS)
    print("=" * 60 + "\n")


def print_answer(result: dict) -> None:
    """Display the answer and its sources cleanly."""
    print(f"\n{DIVIDER}")
    print(f"\n  {result['answer']}\n")

    if result["sources"]:
        print(f"  Sources : {', '.join(result['sources'])}")

    print(f"  Tokens  : {result['tokens']['total']} "
          f"| Turns: {agent.turn_count()}")
    print(f"\n{DIVIDER}\n")


def handle_reset() -> None:
    agent.reset()
    print("\n  Memory cleared. Starting fresh.\n")


def handle_history() -> None:
    history = agent.history()
    if not history:
        print("\n  No conversation history yet.\n")
        return

    print(f"\n{DIVIDER}")
    print(f"  Conversation history ({agent.turn_count()} turns)\n")
    for msg in history:
        role    = "You" if msg["role"] == "user" else "Agent"
        preview = msg["content"][:200]
        print(f"  [{role}] {preview}")
        print()
    print(f"{DIVIDER}\n")


def handle_sources(last_result: dict | None) -> None:
    if not last_result:
        print("\n  No question asked yet.\n")
        return

    chunks = last_result.get("chunks", [])
    if not chunks:
        print("\n  No chunks retrieved for last answer.\n")
        return

    print(f"\n{DIVIDER}")
    print(f"  Sources used in last answer ({len(chunks)} chunks)\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] {chunk['source']} "
              f"(chunk #{chunk['chunk_index']}, "
              f"dist={chunk['distance']})")
        print(f"      {chunk['text'][:200]}...")
        print()
    print(f"{DIVIDER}\n")


if __name__ == "__main__":
    print_welcome()

    agent       = Orchestrator(top_k=5, verbose=False)
    last_result = None

    while True:
        try:
            question = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye.\n")
            break

        if not question:
            continue

        if question.lower() == "/quit":
            print("\n  Goodbye.\n")
            break

        elif question.lower() == "/reset":
            handle_reset()

        elif question.lower() == "/history":
            handle_history()

        elif question.lower() == "/sources":
            handle_sources(last_result)

        else:
            try:
                last_result = agent.ask(question)
                print_answer(last_result)
            except Exception as e:
                print(f"\n  Error: {e}\n")
                print("  Make sure your .env has a valid OPENAI_API_KEY.\n")