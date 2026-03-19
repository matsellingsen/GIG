from openvino.runtime import Core
from transformers import AutoTokenizer
import numpy as np
from .backend import Backend

class OpenVINOINFERENCEBACKEND(Backend):
    def __init__(self):
        # Paths to your exported IR model
        self.MODEL_XML = "../models/qwen2.5-1.5b-instruct-ir2/openvino_model.xml"
        self.TOKENIZER_DIR = "../models/qwen2.5-1.5b-instruct-ir2"  # folder containing tokenizer.json etc.


        # 1. Load tokenizer fully offline
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.TOKENIZER_DIR,
            local_files_only=True,
            fix_mistral_regex=True
        )

        # 2. Load OpenVINO IR model
        self.core = Core()
        self.model = self.core.read_model(self.MODEL_XML)

        # 3. Compile for Intel NPU
        self.compiled_model = self.core.compile_model(self.model, device_name="CPU")
        self.infer = self.compiled_model.create_infer_request()
        self.static_len = self.compiled_model.input(0).shape[1]

    def generate(self, prompt, max_new_tokens=100):
        enc = self.tokenizer(prompt, return_tensors="np")
        input_ids = enc["input_ids"]

        # Pad to static shape
        pad_len = self.static_len - input_ids.shape[1]
        if pad_len < 0:
            raise ValueError(f"Prompt too long for static shape {self.static_len}")

        input_ids = np.pad(
            input_ids,
            ((0, 0), (0, pad_len)),
            constant_values=self.tokenizer.pad_token_id
        )

        result = self.infer.infer({"input_ids": input_ids})
        return result[self.compiled_model.output(0)]

