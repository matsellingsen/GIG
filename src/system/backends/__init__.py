from .backend import Backend
from .onnx_backend import ONNXINFERENCEBACKEND


def load_backend(name: str = "onnx", **kwargs):
    if name == "onnx":
        return ONNXINFERENCEBACKEND(**kwargs)
    # Add more backends here as needed

    else:
        raise ValueError(f"Unknown backend: {name}")