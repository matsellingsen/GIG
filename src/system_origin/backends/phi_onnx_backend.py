import openvino
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from .backend import Backend

class PHIONNXINFERENCEBACKEND(Backend):
    def __init__(self):
        # -----------------------------
        # 1. Load tokenizer locally
        # -----------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            "../models/phi-4-mini-instruct-onnx",  # folder containing tokenizer.json etc.
            local_files_only=True
        )

        eos_token_ids = self.tokenizer.eos_token_id
        if isinstance(eos_token_ids, int):
            self.eos_token_ids = {eos_token_ids}
        else:
            self.eos_token_ids = set(eos_token_ids)

        # -----------------------------
        # 2. Load ONNX model on NPU
        # -----------------------------
        self.session = ort.InferenceSession(
            "../models/phi-4-mini-instruct-onnx/model.onnx",
            providers=["CPUExecutionProvider"]
            #providers=[("OpenVINOExecutionProvider", {"device_type": "GPU"})] #Switch to this line to use OpenVINO on NPU instead of ONNX Runtime on CPU. Currently, NPU support in ONNX Runtime is not mature, so we use CPU for now.
        )

        self.input_metas = self.session.get_inputs()
        self.output_metas = self.session.get_outputs()
        self.input_names = [i.name for i in self.input_metas]
        self.output_names = [o.name for o in self.output_metas]
        self.past_input_names = [n for n in self.input_names if n.startswith("past_key_values.")]


    def ort_type_to_np_dtype(self, ort_type: str):
        mapping = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(bool)": np.bool_,
        }
        if ort_type not in mapping:
            raise ValueError(f"Unsupported ONNX input type: {ort_type}")
        return mapping[ort_type]


    def _resolve_dynamic_dim(self, dim_name: str, axis: int, batch_size: int, past_seq_len: int):
        name = str(dim_name).lower()
        if "batch" in name:
            return batch_size
        if "past" in name:
            return past_seq_len
        if "seq" in name or "sequence" in name:
            return past_seq_len if axis >= 2 else 1
        return 1


    def init_past_cache(self, batch_size: int, past_seq_len: int = 0):
        past = {}
        for meta in self.input_metas:
            if not meta.name.startswith("past_key_values."):
                continue

            shape = []
            for axis, d in enumerate(meta.shape):
                if isinstance(d, int):
                    shape.append(d)
                else:
                    shape.append(self._resolve_dynamic_dim(d, axis, batch_size, past_seq_len))

            dtype = self.ort_type_to_np_dtype(meta.type)
            past[meta.name] = np.zeros(shape, dtype=dtype)

        return past


    def extract_next_past_from_outputs(self, outputs):
        next_past = {}
        for name, value in zip(self.output_names, outputs):
            if name.startswith("present."):
                next_past[name.replace("present.", "past_key_values.")] = value

        missing = [n for n in self.past_input_names if n not in next_past]
        if missing:
            raise ValueError(f"Model did not return all present cache tensors. Missing: {missing[:4]}...")

        return next_past


    def run_step(self, input_ids_step, past):
        batch_size = input_ids_step.shape[0]
        any_cache = next(iter(past.values()))
        past_seq_len = any_cache.shape[2]

        # For this export, total_sequence_length = past_sequence_length + current step length.
        attention_mask = np.ones((batch_size, past_seq_len + input_ids_step.shape[1]), dtype=np.int64)

        feed = {
            "input_ids": input_ids_step,
            "attention_mask": attention_mask,
        }
        feed.update(past)

        outputs = self.session.run(None, feed)
        logits = outputs[0]
        next_past = self.extract_next_past_from_outputs(outputs)
        return logits, next_past


    def generate(self, prompt, max_new_tokens=2048):

        prompt_ids = self.tokenizer.encode(prompt, return_tensors="np").astype(np.int64)
        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]

        generated_ids = prompt_ids.copy()
        past = self.init_past_cache(batch_size=batch_size, past_seq_len=0)

        
        print("MAX NEW TOKENS:", max_new_tokens)

        # Token-by-token prefill is more robust with DML GroupQueryAttention than a single long prefill.
        last_logits = None
        for pos in range(prompt_len):
            input_step = prompt_ids[:, pos:pos + 1]
            last_logits, past = self.run_step(input_step, past)

        for _ in range(max_new_tokens):
            next_token_id = int(np.argmax(last_logits[0, -1]))
            if next_token_id in self.eos_token_ids:
                break

            next_token = np.array([[next_token_id]], dtype=np.int64)
            generated_ids = np.concatenate([generated_ids, next_token], axis=1)

            last_logits, past = self.run_step(next_token, past)
        
        # remove the prompt tokens from the output
        generated_ids = generated_ids[:, prompt_len:]
        
        print("length of generated ids:", generated_ids.shape[1])

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)



