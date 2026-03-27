from pathlib import Path
import json

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from .backend import Backend
from tools.owl2fs_v2_renderer import render_owl2fs_v2_document
from tools.schema_constrained_decoder_v2 import SchemaConstrainedDecoderV2


class QWENONNXINFERENCEBACKEND(Backend):
    def __init__(self):
        root_dir = Path(__file__).resolve().parents[3]
        self.model_dir = root_dir / "models" / "qwen2.5-1.5b-instruct-onnx"
        self.model_path = self.model_dir / "model_int8.onnx"
        self.schema_path = root_dir / "src" / "system" / "prompts" / "tools" / "owl2_output_schema_v2.json"

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        eos_token_ids = self.tokenizer.eos_token_id
        self.eos_token_ids = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids)

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_metas = self.session.get_inputs()
        self.output_metas = self.session.get_outputs()
        self.input_names = [i.name for i in self.input_metas]
        self.output_names = [o.name for o in self.output_metas]
        self.past_input_names = [n for n in self.input_names if n.startswith("past_key_values.")]

        self.constrained_decoder = SchemaConstrainedDecoderV2(
            tokenizer=self.tokenizer,
            schema_path=str(self.schema_path),
        )
        self.last_structured_output = None
        self.last_rendered_output = None

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
        step_len = input_ids_step.shape[1]

        attention_mask = np.ones((batch_size, past_seq_len + step_len), dtype=np.int64)

        # Qwen ONNX export expects explicit position_ids.
        position_ids = np.arange(past_seq_len, past_seq_len + step_len, dtype=np.int64)
        position_ids = np.broadcast_to(position_ids[None, :], (batch_size, step_len)).copy()

        feed = {
            "input_ids": input_ids_step,
            "attention_mask": attention_mask,
        }
        if "position_ids" in self.input_names:
            feed["position_ids"] = position_ids
        if "beam_idx" in self.input_names:
            feed["beam_idx"] = np.arange(batch_size, dtype=np.int32)

        feed.update(past)

        outputs = self.session.run(None, feed)
        logits = outputs[0]
        next_past = self.extract_next_past_from_outputs(outputs)
        return logits, next_past

    def generate(self, prompt, max_new_tokens=100):
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="np").astype(np.int64)
        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]

        generated_ids = prompt_ids.copy()
        past = self.init_past_cache(batch_size=batch_size, past_seq_len=0)
        self.constrained_decoder.reset()

        last_logits = None
        for pos in range(prompt_len):
            input_step = prompt_ids[:, pos : pos + 1]
            last_logits, past = self.run_step(input_step, past)

        for _ in range(max_new_tokens):
            next_token_id = self.constrained_decoder.select_next_token(last_logits[0, -1], top_k=4096)
            if next_token_id is None:
                break

            if not self.constrained_decoder.apply_token(next_token_id):
                break

            if next_token_id in self.eos_token_ids and not self.constrained_decoder.is_finished():
                break

            next_token = np.array([[next_token_id]], dtype=np.int64)
            generated_ids = np.concatenate([generated_ids, next_token], axis=1)
            last_logits, past = self.run_step(next_token, past)

            if self.constrained_decoder.is_finished():
                break

        generated_ids = generated_ids[:, prompt_len:]
        structured_output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.last_structured_output = structured_output

        try:
            payload = json.loads(structured_output)
            self.last_rendered_output = render_owl2fs_v2_document(payload)
        except Exception:
            self.last_rendered_output = None

        return structured_output
