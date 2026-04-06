from backends import load_backend
from agent_loop.orchestrator import Orchestrator
from tools.load_chunks import load_chunks
from tools.save_result import save_result
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="phi-npu-openvino") # decides which backend to use here.
    args = parser.parse_args()

    backend = load_backend(name=args.backend)
    print(f"Loaded backend: {args.backend}")

    orchestrator = Orchestrator(backend=backend)
    # load chunked text data
    chunks = load_chunks("C:\\Users\\matse\\gig\\src\\system_v5\\content\\chunks.jsonl")
    run_chunks = None # set to None to run on all chunks, or set to an integer for testing with a limited number of chunks.
    print("Loaded system prompt and chunked text.")

    pipeline_result, extraction_status, prompts = orchestrator.run_pipeline(chunks=chunks, run_chunks=run_chunks) #run_chunks limits how many chunks we process for testing. Remove this limit for full run.
    print("Pipeline execution completed.")
    #print("Final structured ontology result:")
    #print(pipeline_result)

    # save the result with metadata
    save_result(pipeline_result, model_name=args.backend, pipeline_info={"num_chunks": len(chunks), "chunks_processed": run_chunks, "orchestrator_type": "V5", "source": "chunks.jsonl", "extraction_status": extraction_status, "prompts": prompts})



if __name__ == "__main__":
    main()