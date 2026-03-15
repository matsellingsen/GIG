from backends import load_backend
from agent_loop.agents.simple_agent import SimpleAgent
from agent_loop.orchestrator import Orchestrator
from tools.load_prompt import load_prompt
from tools.load_source import load_source
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="onnx")
    args = parser.parse_args()

    backend = load_backend(name=args.backend)
    print(f"Loaded backend: {args.backend}")

    agent = SimpleAgent(backend=backend)
    print("Initialized SimpleAgent.")

    orchestrator = Orchestrator(agent)
    print("Initialized Orchestrator.")

    system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system\\prompts\\system\\simple_ontology_construct.txt") 
    source_text = load_source(file_path=None) # change to actual file-path later.
    print("Loaded system prompt and source text.")

    result = orchestrator.run_once(system_prompt=system_prompt, source_text=source_text)
    
    print("Final result:")
    print("-------------")
    print(result)


if __name__ == "__main__":
    main()