from .backend import Backend

from .phi_openvino_NPU_backend import PhiOpenVINONPUBackend
from .phi_openvino_NPU_backend2 import PhiOpenVINONPUBackend as PhiOpenVINONPUBackend2

# Add more backends here as needed
def load_backend(name: str = "phi-npu-openvino", **kwargs):
 
    if name == "phi-npu-openvino":
        #return PhiOpenVINONPUBackend(**kwargs)
        return PhiOpenVINONPUBackend2(**kwargs) # For testing new backend version side by side. Switch to this one when ready.

    else:
        raise ValueError(f"Unknown backend: {name}")