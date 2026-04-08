"""
Orchestrator module.
Coordinates the full RAG pipeline for each user query.
Connects: memory → retrieval → prompt → generation → memory.
This is the single entry point for the chat interface.
"""

import sys
import os
sys.path.insert(0, ".")

# LangSmith tracing — activate by setting LANGCHAIN_TRACING_V2=true in .env
os.environ.setdefault("LANGCHAIN_TRACING_V2",  os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_API_KEY",      os.getenv("LANGCHAIN_API_KEY", ""))
os.environ.setdefault("LANGCHAIN_PROJECT",      os.getenv("LANGCHAIN_PROJECT", "contextagent"))

from src.retrieval.retriever  import retrieve, format_context
from src.generation.prompt    import build_messages, build_standalone_question
from src.generation.generator import generate
from src.memory.buffer        import ConversationBuffer
from src.storage.vector_store import get_collection_count


class Orchestrator:
    """
    The central coordinator of the ContextAgent RAG pipeline.

    Holds one ConversationBuffer for the session.
    Exposes a single public method: ask().

    Usage:
        agent = Orchestrator()
        result = agent.ask("What is a Python class?")
        print(result["answer"])
    """

    def __init__(self, top_k: int = 5, verbose: bool = False):
        """
        Args:
            top_k:   number of chunks to retrieve per query
            verbose: if True, print pipeline steps during ask()
        """
        self.buffer  = ConversationBuffer()
        self.top_k   = top_k
        self.verbose = verbose
        self._check_vector_store()

    def _check_vector_store(self) -> None:
        """
        Warn if ChromaDB is empty.
        The system cannot answer questions without ingested documents.
        """
        count = get_collection_count()
        if count == 0:
            print(
                "\n  Warning: ChromaDB is empty.\n"
                "  Run: python scripts/ingest.py\n"
            )
        elif self.verbose:
            print(f"  ChromaDB ready — {count} chunks available.\n")

    def ask(self, question: str) -> dict:
        """
        Full RAG pipeline for a single user question.

        Steps:
            1. Get conversation history from buffer
            2. Reformulate question if history exists
            3. Retrieve relevant chunks from ChromaDB
            4. Format chunks into context string
            5. Build messages (system + context + history + question)
            6. Generate answer via LLM
            7. Save turn to buffer
            8. Return structured result

        Args:
            question: the user's question as plain text

        Returns:
            {
                "answer":   the LLM response,
                "sources":  list of source filenames,
                "chunks":   the retrieved chunks,
                "tokens":   token usage dict,
                "model":    model name
            }
        """
        if self.verbose:
            print(f"\n  [1/5] Getting conversation history...")

        history = self.buffer.get()

        if self.verbose:
            print(f"  [2/5] Reformulating question...")

        search_query = build_standalone_question(question, history)

        if self.verbose:
            print(f"  [3/5] Retrieving relevant chunks...")

        chunks = retrieve(search_query, top_k=self.top_k)

        if self.verbose:
            print(f"  Retrieved {len(chunks)} chunks from: "
                  f"{list(set(c['source'] for c in chunks))}")

        if self.verbose:
            print(f"  [4/5] Building prompt...")

        context  = format_context(chunks)
        messages = build_messages(
            question = question,
            context  = context,
            history  = history if history else None,
        )

        if self.verbose:
            print(f"  [5/5] Generating answer...")

        result = generate(messages)

        self.buffer.add(
            user_message      = question,
            assistant_message = result["answer"],
        )

        result["chunks"] = chunks
        return result

    def reset(self) -> None:
        """Clear conversation memory. Start fresh."""
        self.buffer.clear()
        if self.verbose:
            print("  Conversation memory cleared.")

    def history(self) -> list[dict]:
        """Return current conversation history."""
        return self.buffer.get()

    def turn_count(self) -> int:
        """Return number of completed conversation turns."""
        return self.buffer.turn_count()


if __name__ == "__main__":
    print("Testing orchestrator...\n")
    print("=" * 50)

    agent = Orchestrator(top_k=3, verbose=True)

    questions = [
        "What is a Python class?",
        "How do you define one?",
        "Can it inherit from multiple classes?",
    ]

    for q in questions:
        print(f"\n  Question: '{q}'")
        print("-" * 50)

        result = agent.ask(q)

        print(f"\n  Answer:\n  {result['answer']}")
        print(f"\n  Sources : {result['sources']}")
        print(f"  Tokens  : {result['tokens']['total']}")
        print(f"  Turns in memory: {agent.turn_count()}")
        print("=" * 50)