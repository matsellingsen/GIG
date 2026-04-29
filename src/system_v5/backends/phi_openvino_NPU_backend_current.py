"""backend for openvino NPU using openvino_genai"""
from openvino_genai import LLMPipeline, SchedulerConfig, GenerationConfig, StructuredOutputConfig
from .backend import Backend
import json


class PhiOpenVINONPUBackend(Backend):
    def __init__(self, model_path="../../models/phi-4-mini-instruct-int4-sym", device="NPU"):
    
        # 1. Compilation Config for NPU Execution
        cache_dir = model_path + "/ov_cache"
        config = { 
                "CACHE_DIR": str(cache_dir),        
                "PERFORMANCE_HINT": "LATENCY", 
                "ENABLE_MMAP": "NO",                # Force full model load into RAM to increase NPU execution speed at the cost of higher memory usage.
                "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32", # Trade memory for math precision
                "NUM_STREAMS": "1",                  # Increased to utilize unused NPU memory for parallel execution
                "MAX_PROMPT_LEN": 4096,              # Must be int for GenAI wrapper
                }

        print(f"Loading GenAI model from {model_path} to {device}...")
        self.pipe = LLMPipeline(
            model_path, 
            device=device, 
            **config
        )
        print("Model loaded.")
    
    def generate(self, prompt: str, max_new_tokens: int = 1024, json_schema: dict = None) -> str:

        # 3. Generation Config (Per-request)
        gen_config = GenerationConfig()
        gen_config.max_new_tokens = max_new_tokens
        
        # DETERMINISM:
        gen_config.do_sample = False # forces deterministic behaviour

        # EFFICIENCY / QUALITY trade-off:
        #gen_config.repetition_penalty = 1.2 # gives a small boost to output diversity without needing sampling, which can be less efficient on NPU. Adjust as needed.

        # Prepare execution arguments
        generate_kwargs = {"generation_config": gen_config}

        #gen_config.eos_token_id = self.pipe.get_tokenizer.eos_token_id # ensure we have a proper stop token to prevent runaway generation, especially important on NPU where you want to control latency.
        
        # 2. Apply Schema if provided
        if json_schema:
            structured_config = StructuredOutputConfig()
            # Depending on your specific OV version, syntax might vary slightly:
            # Option A: Stringified JSON Schema
            structured_config.json_schema = json.dumps(json_schema, sort_keys=True, ensure_ascii=False)
            
            # CRITICAL FIX: Pass as a separate argument, NOT attached to gen_config
            # This ensures the C++ binding receives the configuration.
            generate_kwargs["structured_output_config"] = structured_config
        
      
        # Generate text
        response = self.pipe.generate(prompt, **generate_kwargs)
        return response