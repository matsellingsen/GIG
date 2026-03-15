"""Abstract interface for all agents."""

from abc import ABC, abstractmethod

class Agent(ABC):
    """Abstract interface for all agents."""

    @abstractmethod
    def run(self, user_input: str) -> str:
        pass
