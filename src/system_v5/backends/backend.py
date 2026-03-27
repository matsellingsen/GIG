# backends/backend.py

from abc import ABC, abstractmethod

class Backend(ABC):
    """Abstract interface for all inference backends."""

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate text from a prompt."""
        pass