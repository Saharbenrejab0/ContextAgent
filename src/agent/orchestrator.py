"""
Orchestrator module.
Coordinates the full RAG pipeline for each user query.
Connects: memory → retrieval → prompt → generation → memory.
"""

import sys
import os
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

# LangSmith tracing
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_API_KEY",    os.getenv("LANGCHAIN_API_KEY", ""))
os.environ.setdefault("LANGCHAIN_PROJECT",    os.getenv("LANGCHAIN_PROJECT", "contextagent"))

from langsmith import traceable

from src.retrieval.retriever  import retrieve, format_context
from src.generation.prompt    import build_messages, build_standalone_question
from src.generation.generator import generate
from src.memory.buffer        import ConversationBuffer
from src.storage.vector_store import get_collection_count


class Orchestrator:
    def __init__(self, top_k: int = 5, verbose: bool = False):
        self.buffer  = ConversationBuffer()
        self.top_k   = top_k
        self.verbose = verbose
        self._check_vector_store()

    def _check_vector_store(self) -> None:
        try:
            count = get_collection_count()
            if count == 0:
                print("\n  Warning: ChromaDB is empty.\n  Run: python scripts/ingest.py\n")
            elif self.verbose:
                print(f"  ChromaDB ready — {count} chunks available.\n")
        except Exception:
            print("\n  Warning: Could not connect to ChromaDB.\n")

    @traceable(name="orchestrator.ask")
    def ask(self, question: str) -> dict:
        """Full RAG pipeline for a single user question."""
        if self.verbose:
            print(f"\n  [1/5] Getting conversation history...")
        history = self.buffer.get()

        if self.verbose:
            print(f"  [2/5] Reformulating question...")
        search_query = build_standalone_question(question, history)

        # Use original question for retrieval — not reformulated
        # The LLM already gets full history in the prompt
        # Reformulation only pollutes the retriever with previous context
        chunks = retrieve(
            query          = question,
            top_k          = self.top_k,
            original_query = question,
        )

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
        self.buffer.clear()
        if self.verbose:
            print("  Conversation memory cleared.")

    def history(self) -> list[dict]:
        return self.buffer.get()

    def turn_count(self) -> int:
        return self.buffer.turn_count()