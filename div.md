#### line to deactivate hardlink for uv
export UV_LINK_MODE=copy


#### Tool-enabled function-calling format (PHI-4-mini-instruct)
This format is used when the user wants the model to provide function calls based on the given tools. The user should provide the available tools in the system prompt, wrapped by <|tool|> and <|/tool|> tokens. The tools should be specified in JSON format, using a JSON dump structure. Example:

<|system|>You are a helpful assistant with some tools.<|tool|>[{"name": "get_weather_updates", "description": "Fetches weather updates for a given city using the RapidAPI Weather API.", "parameters": {"city": {"description": "The name of the city for which to retrieve weather information.", "type": "str", "default": "London"}}}]<|/tool|><|end|><|user|>What is the weather like in Paris today?<|end|><|assistant|>


#### Which .venvs work with what:

.venv: optimum.intel

.venv5: works with both openvino_genai and optimum.intel


#### findings:
1. do-it-all agent struggles because of two points: 1. too complex task, 2. incrementally increasing found classes and axioms and showing it to suceeding chunks causes the model to loose memory, and effectively degrades instruction following for later chunks. 
  --> Use stateless generation per chunk instead.
2. Flat instance extraction representation causes multiple clasassertions for the same individual. -> switch to hierarchical structuring instead.
