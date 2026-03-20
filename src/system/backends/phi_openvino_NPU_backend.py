"""backend for openvino NPU using openvino_genai"""
from openvino_genai import LLMPipeline, SchedulerConfig, GenerationConfig
from .backend import Backend

class PhiOpenVINONPUBackend(Backend):
    def __init__(self, model_path="../../models/phi-4-mini-instruct-int4-sym", device="NPU", max_cache_size=2048):
        
        # 1. Pipeline Config
        scheduler_config = SchedulerConfig()         # pre-allocate for max context length to avoid fragmentation and speed up NPU execution.
        scheduler_config.dynamic_split_fuse = True   # Enable dynamic splitting and fusing of operations for better performance on NPU/GPU.
        scheduler_config.cache_size = max_cache_size # Set to your max logical context length to reserve NPU memory
        
        # 2. Compilation Config
        cache_dir = model_path + "/ov_cache"
        config = { "CACHE_DIR": str(cache_dir),        # enables compiling to disk (essential for NPU startup speed) 
                   "PERFORMANCE_HINT": "LATENCY", # Optimize for single-user speed
                   "NUM_STREAMS": "1",  # Force serial execution (often better for NPU latency)
                   "MAX_PROMPT_LEN": 2048,       
                 }

        print(f"Loading GenAI model from {model_path} to {device}...")
        self.pipe = LLMPipeline(
            model_path, 
            device=device, 
            #scheduler_config=scheduler_config,
            **config
        )
        print("Model loaded.")
    
    def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:

        # 3. Generation Config (Per-request)
        gen_config = GenerationConfig()
        gen_config.max_new_tokens = max_new_tokens
        
        # DETERMINISM:
        gen_config.do_sample = False # forces deterministic behaviour

        # EFFICIENCY / QUALITY trade-off:
        gen_config.repetition_penalty = 1.2 # gives a small boost to output diversity without needing sampling, which can be less efficient on NPU. Adjust as needed.
        #gen_config.eos_token_id = self.pipe.get_tokenizer.eos_token_id # ensure we have a proper stop token to prevent runaway generation, especially important on NPU where you want to control latency.
        
        # Generate text
        response = self.pipe.generate(prompt, config=gen_config)
        return response