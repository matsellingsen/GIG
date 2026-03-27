def load_prompt(file_path: str = None) -> str:
    """Get the system prompt to be used as the initial context for the agent."""
    if file_path:
        try:
            with open(file_path, "r") as f:
                prompt = f.read()
        except Exception as e:
            raise ValueError(f"Error reading prompt file: {e}")
    else:
        raise ValueError("No prompt file path provided. Please specify a file path to load the prompt.")
    return prompt