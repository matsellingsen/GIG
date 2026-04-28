"""Final step of the inference module pipeline: generate an answer to the question based on the retrieved relevant information."""
import sys
sys.path.append("C:\\Users\\matse\\gig\\src\\system_v5") # Adjust this path to point to the root of your project if needed

import argparse
from backends import load_backend
from tools.inference_module.fetch_relevant_info import main as fetch_info_main
from agent_loop.agents.inference_module.generate_answer_agent import GenerateAnswerAgent

def generate_answer(question_info, relevant_info):
    answer, _ = generate_answer_agent.run(question_info=question_info, relevant_info=relevant_info)
    return answer


def main():
    parser = argparse.ArgumentParser(description="Generate an answer to the question based on the retrieved relevant information.")
    parser.add_argument("--backend", default="phi-npu-openvino") # decides which backend to use here.
    args = parser.parse_args()

    backend = load_backend(name=args.backend)
    
    global generate_answer_agent
    generate_answer_agent = GenerateAnswerAgent(backend=backend)
    
    # Fetch relevant info for dummy questions (this simulates the full pipeline up to this point)
    relevant_infos = fetch_info_main() # Runs script on dummy questions and retrieves relevant info for each question.

    for qid, info in relevant_infos.items():
        question_info = info.get("question_info")
        resolved_entity = info.get("resolved_entity")
        relevant_info = info.get("relevant_info")
        print(f"\n\n========== Generating answer for question {qid} ==========")
        print(f"Question info: {question_info}")
        #print(f"Resolved entity: {resolved_entity}")
        #print(f"Relevant info: {relevant_info}")
        print(" == Relevant info details ==")
        for key, value in relevant_info.items():
            print(f"  {key}: {value}")
        print(" ====================")

        answer = generate_answer(question_info=question_info, relevant_info=relevant_info)
        print(f"Generated answer: {answer}")

if __name__ == "__main__":
    main()