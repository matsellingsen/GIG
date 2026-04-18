from .backend import Backend
from .phi_openvino_NPU_backend_current import PhiOpenVINONPUBackend

# Add more backends here as needed
def load_backend(name: str = "phi-npu-openvino", **kwargs):
 
    if name == "phi-npu-openvino":
        return PhiOpenVINONPUBackend(**kwargs)

    else:
        raise ValueError(f"Unknown backend: {name}")