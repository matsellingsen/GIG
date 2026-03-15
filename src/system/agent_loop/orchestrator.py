
class Orchestrator:
    """
    Minimal orchestrator that:
    - repeatedly asks the agent to process input
    - stops when the agent signals completion
    """

    def __init__(self, agent):
        self.agent = agent

    def run_once(self, system_prompt: str, source_text: str) -> str:
        """Single-step execution."""

        # Compose final prompt
        prompt = f"{system_prompt}<|user|>{source_text}<|end|><|Assistant|>"
        print("Composed prompt for agent:")
        print("-------------")
        print(prompt)
        print("-------------")
        return self.agent.run(prompt)

    def run_loop(self, user_input: str, max_steps: int = 3) -> str:
        """
        Multi-step loop:
        - passes the user input to the agent
        - agent may return a final answer or a 'continue' signal
        - stops early if agent is done
        """
        current_input = user_input

        for step in range(max_steps):
            output = self.agent.run(current_input)

            # Minimal stopping condition:
            # If the agent returns a string without a special token, we stop.
            if "<CONTINUE>" not in output:
                return output

            # Otherwise, strip the token and continue looping
            current_input = output.replace("<CONTINUE>", "").strip()

        return output  # fallback if max_steps reached