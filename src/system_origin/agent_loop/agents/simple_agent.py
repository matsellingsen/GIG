from .agent import Agent

class SimpleAgent(Agent):
    """A simple agent that naively creates an OWL 2 KB from the user input all at once, without any intermediate steps or reasoning. Can be seen as a baseline for more complex agents that do multi-step reasoning, tool use, etc."""

    def __init__(self, backend, system_prompt: str = None):
        self.backend = backend

    def run(self, prompt: str) -> str:
    
        # Call backend
        return self.backend.generate(prompt)
