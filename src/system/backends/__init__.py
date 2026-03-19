from .backend import Backend
from .phi_onnx_backend import PHIONNXINFERENCEBACKEND
from .qwen_onnx_backend import QWENONNXINFERENCEBACKEND
from .openvino_backend import OpenVINOINFERENCEBACKEND
from .phi_openvino_backend import PhiOpenVINOBackend


def load_backend(name: str = "phi-onnx", **kwargs):
    if name == "phi-onnx":
        return PHIONNXINFERENCEBACKEND(**kwargs)
    if name == "qwen-onnx":
        return QWENONNXINFERENCEBACKEND(**kwargs)
    if name == "phi-openvino":
        return PhiOpenVINOBackend(**kwargs)
    # Add more backends here as needed
    if name == "openvino":
        return OpenVINOINFERENCEBACKEND(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {name}")