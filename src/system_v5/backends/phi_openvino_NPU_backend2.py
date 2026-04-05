"""backend for openvino NPU using openvino_genai"""
from openvino_genai import LLMPipeline, SchedulerConfig, GenerationConfig, StructuredOutputConfig
from .backend import Backend
import json


class PhiOpenVINONPUBackend(Backend):
        def __init__(self, model_path="../../models/phi-4-mini-instruct-int4-sym", device="NPU", max_cache_size=5120):
        
            # 1. Pipeline Config: Pre-allocate memory aggressively for NPU
            scheduler_config = SchedulerConfig()         
            scheduler_config.dynamic_split_fuse = True   
            scheduler_config.cache_size = max_cache_size 
            
            # 2. Compilation Config: Maximize memory mapping and streams
            cache_dir = model_path + "/ov_cache"
            config = { 
                    "CACHE_DIR": str(cache_dir),        
                    "PERFORMANCE_HINT": "LATENCY", 
                    "ENABLE_MMAP": "YES",                # Map weights directly to RAM
                    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32", # Trade memory for math precision
                    "NUM_STREAMS": "2",                  # Allow parallel token processing
                    "MAX_PROMPT_LEN": 4096,       
                    }

            print(f"Loading GenAI model from {model_path} to {device}...")
            self.pipe = LLMPipeline(
                model_path, 
                device=device, 
                scheduler_config=scheduler_config,          # CRITICAL: Uncommented this!
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
        gen_config.repetition_penalty = 1.2 # gives a small boost to output diversity without needing sampling, which can be less efficient on NPU. Adjust as needed.

        # Prepare execution arguments
        generate_kwargs = {"config": gen_config}

        #gen_config.eos_token_id = self.pipe.get_tokenizer.eos_token_id # ensure we have a proper stop token to prevent runaway generation, especially important on NPU where you want to control latency.
        
         # 2. Apply Schema if provided
        if json_schema:
            structured_config = StructuredOutputConfig()
            # Depending on your specific OV version, syntax might vary slightly:
            # Option A: Stringified JSON Schema
            structured_config.json_schema = json.dumps(json_schema) 
            
            # CRITICAL FIX: Pass as a separate argument, NOT attached to gen_config
            # This ensures the C++ binding receives the configuration.
            generate_kwargs["structured_output_config"] = structured_config
        
      
        # Generate text
        response = self.pipe.generate(prompt, **generate_kwargs)
        return response