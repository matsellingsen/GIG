from .backend import Backend
#from .phi_onnx_backend import PHIONNXINFERENCEBACKEND # Currently not used due to ONNX Runtime's limited NPU support, but can be enabled for CPU inference or if ONNX NPU support improves.
#from .qwen_onnx_backend import QWENONNXINFERENCEBACKEND # same^
#from .openvino_backend import OpenVINOINFERENCEBACKEND
from .phi_openvino_backend import PhiOpenVINOBackend
from .phi_openvino_NPU_backend import PhiOpenVINONPUBackend

# Add more backends here as needed
def load_backend(name: str = "phi-onnx", **kwargs):
    #if name == "phi-onnx":
    #    return PHIONNXINFERENCEBACKEND(**kwargs)
    #if name == "qwen-onnx":
    #    return QWENONNXINFERENCEBACKEND(**kwargs)
    if name == "phi-openvino":
        return PhiOpenVINOBackend(**kwargs)
    if name == "phi-npu-openvino":
        return PhiOpenVINONPUBackend(**kwargs)
    #if name == "openvino":
    #    return OpenVINOINFERENCEBACKEND(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {name}")