"""
Memory buffer module.
Maintains a sliding window of recent conversation turns.
Each turn is a user message + assistant response pair.
No persistence — memory lives only for the current session.
"""

import os
from dotenv import load_dotenv

load_dotenv()

MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", 6))


class ConversationBuffer:
    """
    A sliding window buffer that stores the last N conversation turns.

    A "turn" is one user message + one assistant response.
    MEMORY_WINDOW=6 means we keep the last 6 turns = 12 messages.

    Why a class and not just a list?
    Because the buffer has behaviour — add, get, clear, trim.
    A plain list would scatter that logic everywhere.
    """

    def __init__(self, window: int = MEMORY_WINDOW):
        self.window:   int        = window
        self.messages: list[dict] = []

    def add(self, user_message: str, assistant_message: str) -> None:
        """
        Add a completed turn to the buffer.
        Trims the buffer if it exceeds the window size.

        Args:
            user_message:      the question the user asked
            assistant_message: the answer the assistant gave
        """
        self.messages.append({"role": "user",      "content": user_message})
        self.messages.append({"role": "assistant", "content": assistant_message})
        self._trim()

    def get(self) -> list[dict]:
        """
        Return all messages currently in the buffer.
        This list is passed directly to build_messages() as history.
        """
        return self.messages.copy()

    def clear(self) -> None:
        """Wipe the buffer. Called when the user starts a new session."""
        self.messages = []

    def is_empty(self) -> bool:
        """Return True if no conversation has happened yet."""
        return len(self.messages) == 0

    def turn_count(self) -> int:
        """Return the number of completed turns stored."""
        return len(self.messages) // 2

    def _trim(self) -> None:
        """
        Keep only the most recent N turns.
        Each turn = 2 messages (user + assistant).
        So we keep the last window*2 messages.
        """
        max_messages = self.window * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]


if __name__ == "__main__":
    print("Testing memory buffer...\n")

    buffer = ConversationBuffer(window=3)

    turns = [
        ("What is a Python class?",         "A class is a blueprint for objects..."),
        ("How do you define one?",           "You use the class keyword followed by..."),
        ("Can it inherit from other classes?","Yes, Python supports multiple inheritance..."),
        ("What is a method?",                "A method is a function defined inside a class..."),
        ("Give me an example.",              "Here is an example: class Dog(Animal): ..."),
    ]

    for user_msg, assistant_msg in turns:
        buffer.add(user_msg, assistant_msg)
        print(f"  Added turn {buffer.turn_count()} — buffer has {len(buffer.get())} messages")

    print(f"\n  Window size : {buffer.window} turns")
    print(f"  Turns stored: {buffer.turn_count()} (max {buffer.window})")
    print(f"\n  Current buffer contents:\n")

    for msg in buffer.get():
        role    = msg["role"].upper()
        preview = msg["content"][:80]
        print(f"  [{role}] {preview}")

    print(f"\n  Clearing buffer...")
    buffer.clear()
    print(f"  Buffer empty: {buffer.is_empty()}")