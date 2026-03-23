from backends import load_backend
from agent_loop.agents.simple_agent import SimpleAgent
from agent_loop.orchestrator import Orchestrator_v1
from agent_loop.orchestrator_v2 import Orchestrator_v2
from agent_loop.orchestrator_v3 import Orchestrator_v3
from tools.load_prompt import load_prompt
from tools.load_chunks import load_chunks
from tools.save_result import save_result
from tools.chunk_md import chunk_article
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="phi-npu-openvino") # decides which backend to use here.
    parser.add_argument("--orchestrator_version", default="v3") # allows switching between orchestrator versions for testing.
    args = parser.parse_args()

    backend = load_backend(name=args.backend)
    print(f"Loaded backend: {args.backend}")

    if args.orchestrator_version == "v2":
        agent = SimpleAgent(backend=backend)
        print("Initialized SimpleAgent.")
        orchestrator = Orchestrator_v2(agent, model_type=args.backend)
        print("Initialized Orchestrator_v2.")
    else: # newer orchestrator version loads multiple agents internally, so we don't need to initialize them here.
        orchestrator = Orchestrator_v3(backend=backend)
        print("Initialized Orchestrator_v3.")
    # load chunked text data
    chunks = load_chunks("C:\\Users\\matse\\gig\\src\\system\\content\\chunks.jsonl")
    run_chunks = 10
    print("Loaded system prompt and chunked text.")

    pipeline_result, extraction_status, prompts = orchestrator.run_pipeline(chunks=chunks, run_chunks=run_chunks) #run_chunks limits how many chunks we process for testing. Remove this limit for full run.
    print("Pipeline execution completed.")
    #print("Final structured ontology result:")
    #print(pipeline_result)

    # save the result with metadata
    save_result(pipeline_result, model_name=args.backend, pipeline_info={"num_chunks": len(chunks), "chunks_processed": run_chunks, "orchestrator_type": args.orchestrator_version, "source": "chunks.jsonl", "extraction_status": extraction_status, "prompts": prompts})


    #result, full_prompt = orchestrator.run_loop(system_prompt=system_prompt, source_text=chunks[1]["chunk_text_clean"])
    #chunked_source_text = chunk_article(source_text)
    #result, full_prompt = orchestrator.run_once(system_prompt=system_prompt, source_text=chunks[1]["chunk_text_clean"])
    #save_result(result, model_name=args.backend, prompt=full_prompt, source_text=chunked_source_text[1])
    #print("Final result:")
    #print("-------------")
    #print(result)


if __name__ == "__main__":
    main()