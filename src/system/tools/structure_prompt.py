def structure_prompt(model_type: str, system_prompt: str, user_input: str) -> str:
    """Compose the final prompt for the agent based on the model type."""
    if model_type == "phi-onnx" or model_type == "phi-openvino":
        # For PHI-4-Mini, we use a simple format with special tokens
        prompt = prompt = f"<|system|>{system_prompt}<|end|><|user|>{user_input}<|end|><|assistant|>"

    elif model_type == "qwen-onnx":
        # For Qwen-2.5, we can use a more conversational format
        prompt = prompt = (
        f"<|im_start|>system\n"
        f"{system_prompt}\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_input}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return prompt