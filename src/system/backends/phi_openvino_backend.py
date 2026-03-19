from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from .backend import Backend

class PhiOpenVINOBackend(Backend):
    def __init__(self, model_path="../../models/phi-4-mini-instruct-int4-asym", device="GPU"):
        """
        OpenVINO Backend for Phi models.
        
        Args:
           model_path (str): Path to the OpenVINO IR model folder (containing .xml and .bin).
                             Defaults to the Phi-4 Mini Int4 model from development notebooks.
           device (str): Device to run inference on. "GPU" (iGPU) is recommended for best 
                         compatibility/performance locally. Use "NPU" for Neural Processing Unit 
                         (requires specific drivers and static shapes often).
        """
        print(f"Loading OpenVINO model from {model_path} to {device}...")
        
        #place cache_dir in same directory as model.
        cache_dir = model_path + "/ov_cache"
        # Load the model with Optimum-Intel
        # compile=False allows us to configure or resize before the heavy compilation step
        self.model = OVModelForCausalLM.from_pretrained(
            model_path,
            device=device,
            ov_config={
                "CACHE_DIR": cache_dir,
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1",
                # "INFERENCE_PRECISION_HINT": "f32", # Uncomment if NPU gives 'I64' errors
            },
            #compile=False 
        )

        # See which device the model is actually on (for debugging)
        # Check where the inference request is actually running
       

        # Force static shape compilation if targeting NPU reliability
        if device.upper() == "NPU":
            #self.model.reshape(1, 128) # For Phi-4 Mini, we can try a longer sequence if memory allows. Adjust as needed.
            pass

        # Compile the model explicitly
        #self.model.compile()
        try:
            # We must access the request from the underlying OVModel.
            if self.model.request:
                # "EXECUTION_DEVICES" is property key in OpenVINO
                compiled_model = self.model.request.get_compiled_model()
                print(f"OpenVINO execution device: {compiled_model.get_property('EXECUTION_DEVICES')}")
            else:
                print(f"Model compiled for {device} (Device info unavailable via .request)")
        except Exception as e:
            print(f"Could not retrieve device info: {e}")
            

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate(self, prompt: str, max_new_tokens: int = 2048) -> str:
        """
        Generate text using the OpenVINO optimized model.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate with Greedy Search (default)
        # optimum-intel executes this loop efficiently in C++ on the device
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False, # Deterministic for now
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode only the generated part (optimum usually returns full seq, 
        # but decode handles it if we want full context or just new tokens)
        
        # Calculate the length of the input tokens to slice the output
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]

        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return result
