from backends import load_backend
from agent_loop.agents.simple_agent import SimpleAgent
from agent_loop.orchestrator import Orchestrator
from tools.load_prompt import load_prompt
from tools.load_chunks import load_chunks
from tools.save_result import save_result
from tools.chunk_md import chunk_article
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="phi-openvino") #switch to "phi-openvino" to use IGPU / NPU with OpenVINO instead of ONNX Runtime on CPU.
    args = parser.parse_args()

    backend = load_backend(name=args.backend)
    print(f"Loaded backend: {args.backend}")

    agent = SimpleAgent(backend=backend)
    print("Initialized SimpleAgent.")

    orchestrator = Orchestrator(agent, model_type=args.backend)
    print("Initialized Orchestrator.")

    #system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system\\prompts\\system\\simple_ontology_construct.txt") 
    #system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system\\prompts\\system\\hard-constrained-ontology_construct_v2.txt")
    #source_text = load_source(file_path="C:\\Users\\matse\\sens-website-hugo\\content\\about\\_index.en.md") # change to actual file-path later.
    chunks = load_chunks("C:\\Users\\matse\\gig\\src\\system\\content\\chunks.jsonl")
    print("Loaded system prompt and chunked text.")

    pipeline_result, prompts = orchestrator.run_pipeline(chunks=chunks)
    print("Pipeline execution completed.")
    print("Final structured ontology result:")
    print(pipeline_result)

    # save the result with metadata
    save_result(pipeline_result, model_name=args.backend, pipeline_info={"num_chunks": len(chunks), "source": "chunks.jsonl", "prompts": prompts})


    #result, full_prompt = orchestrator.run_loop(system_prompt=system_prompt, source_text=chunks[1]["chunk_text_clean"])
    #chunked_source_text = chunk_article(source_text)
    #result, full_prompt = orchestrator.run_once(system_prompt=system_prompt, source_text=chunks[1]["chunk_text_clean"])
    #save_result(result, model_name=args.backend, prompt=full_prompt, source_text=chunked_source_text[1])
    #print("Final result:")
    #print("-------------")
    #print(result)


if __name__ == "__main__":
    main()