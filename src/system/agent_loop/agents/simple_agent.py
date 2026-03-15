from .agent import Agent

class SimpleAgent(Agent):
    """A simple agent that naively creates an OWL 2 KB from the user input without any reasoning or tool use."""

    def __init__(self, backend, system_prompt: str = None):
        self.backend = backend

    def run(self, prompt: str) -> str:
    
        # Call backend
        return self.backend.generate(prompt, max_new_tokens=300)
