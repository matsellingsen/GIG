import openvino_genai
import inspect

print("Checking Streamer classes:")
for name in ['StreamerBase', 'TextStreamer', 'TextParserStreamer']:
    if hasattr(openvino_genai, name):
         cls = getattr(openvino_genai, name)
         print(f"  {name} found")
         # check init args if possible
         try:
             print(f"    init args: {inspect.signature(cls)}")
         except:
             print("    could not inspect init")

print("\nChecking StructuredOutputConfig:")
if hasattr(openvino_genai, 'StructuredOutputConfig'):
    cls = openvino_genai.StructuredOutputConfig
    print(f"  StructuredOutputConfig found")
    try:
        # Create an instance and dir it
        instance = cls()
        print(f"  Instance dir: {[d for d in dir(instance) if not d.startswith('__')]}")
    except:
        print("  Could not create instance")

print("\nChecking GenerationConfig:")
if hasattr(openvino_genai, 'GenerationConfig'):
    cls = openvino_genai.GenerationConfig
    try:
        instance = cls()
        print(f"  Fields: {[d for d in dir(instance) if not d.startswith('__')]}")
    except:
        print("Could not analyze GenerationConfig")

print("\nChecking StopCriteria:")
if hasattr(openvino_genai, 'StopCriteria'):
    print("StopCriteria found")

print("\nChecking LLMPipeline.generate docstring:")
if hasattr(openvino_genai, 'LLMPipeline'):
    try:
        print(openvino_genai.LLMPipeline.generate.__doc__)
    except:
        print("Could not get docstring")
