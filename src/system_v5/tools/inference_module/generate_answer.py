"""Final step of the inference module pipeline: generate an answer to the question based on the retrieved relevant information."""
import sys
sys.path.append("C:\\Users\\matse\\gig\\src\\system_v5") # Adjust this path to point to the root of your project if needed

import argparse
from backends import load_backend
from tools.inference_module.fetch_relevant_info import main as fetch_info_main
from tools.inference_module.map_answer_to_context import map_answer_to_context
from agent_loop.agents.inference_module.generate_answer_agent import GenerateAnswerAgent
from agent_loop.agents.inference_module.filter_evidence_agent import FilterEvidenceAgent
from agent_loop.agents.inference_module.map_to_context_agent import MapToContextAgent

def generate_answer(question_info, relevant_info):
    answer, _ = generate_answer_agent.run(question_info=question_info, relevant_info=relevant_info)
    return answer


def main():
    
    backend = load_backend(name="phi-npu-openvino")
    
    global generate_answer_agent
    global filter_evidence_agent
    global map_to_context_agent
    generate_answer_agent = GenerateAnswerAgent(backend=backend)
    filter_evidence_agent = FilterEvidenceAgent(backend=backend)
    map_to_context_agent = MapToContextAgent(backend=backend)

    # Fetch relevant info for dummy questions (this simulates the full pipeline up to this point)
    relevant_infos = fetch_info_main() # Runs script on dummy questions and retrieves relevant info for each question.

    for qid, info in relevant_infos.items():
        question_info = info.get("question_info")
        resolved_entity = info.get("resolved_entity")
        relevant_info = info.get("relevant_info")
        print(f"\n\n========== Generating answer for question {qid} ==========")
        #print(f"Question info: {question_info}")
        #print(f"Resolved entity: {resolved_entity}")
        #print(f"Relevant info: {relevant_info}")
        print(" == Relevant info details ==")
        for key, value in relevant_info.items():
            print(f"  {key}: {value}")
        print(" ====================")

        """ NOT IN USE
        # 1. filter the relevant info to only include the most pertinent facts (this simulates the evidence selection step)
        #filtered_relevant_info = filter_evidence_agent.run(question_info=question_info, all_info=relevant_info)
        #print(f"Filtered relevant info: {filtered_relevant_info}")
        #print("================================")
        """

        # 2. generate the final answer based on the relevant info
        answer = generate_answer(question_info=question_info, relevant_info=relevant_info)
        print(f"Generated answer: {answer.get('answer')}")
        print("================================")
        # 3. map the answer back to the ontology context
        answer_text = answer.get("answer")
        mapped_answer = map_answer_to_context(answer=answer_text, context=relevant_info)
        #mapped_answer, _ = map_to_context_agent.run(answer=answer_text, full_context=relevant_info)
        print(f"Mapped answer: {mapped_answer}")
        print("================================")


if __name__ == "__main__":
    main()