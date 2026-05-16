"""Final step of the inference module pipeline: generate an answer to the question based on the retrieved relevant information."""
import sys
sys.path.append("C:\\Users\\matse\\gig\\src\\system_v5") # Adjust this path to point to the root of your project if needed

import argparse
from backends import load_backend
from tools.inference_module.fetch_relevant_info import main as fetch_info_main
from tools.inference_module.map_answer_to_context import map_answer_to_context
from agent_loop.agents.inference_module.generate_answer_agent import GenerateAnswerAgent


def generate_answer(question_info, entity_context, object_context, generate_answer_agent: GenerateAnswerAgent, inference_log: dict = None):
    """Generate answer from question and context. Returns dict on success, error string on failure."""
    answer, _ = generate_answer_agent.run(question_info=question_info, entity_context=entity_context, object_context=object_context)
    print(f"Generated answer: {answer}")
    print("================================")

    # Log generated answer
    if inference_log is not None:
        inference_log["generated_answer"] = answer
        
    # Validate answer
    if not answer:
        error_msg = "Failed to generate an answer from the question and context."
        if inference_log is not None:
            inference_log["error"] = error_msg
        return error_msg   
    if not isinstance(answer, dict):
        error_msg = "Generated answer does not have expected structure."
        if inference_log is not None:
            inference_log["error"] = error_msg
        return error_msg
    
    return answer


def main():
    
    backend = load_backend(name="phi-npu-openvino")
    
    global generate_answer_agent
    generate_answer_agent = GenerateAnswerAgent(backend=backend)


    # Fetch relevant info for dummy questions (this simulates the full pipeline up to this point)
    relevant_infos = fetch_info_main() # Runs script on dummy questions and retrieves relevant info for each question.

    for qid, info in relevant_infos.items():
        question_info = info.get("question_info")
        resolved_entity = info.get("resolved_entity")
        entity_context = info.get("entity_context")
        object_context = info.get("object_context")
        print(f"\n\n========== Generating answer for question {qid} ==========")
        triplets = {"entity": question_info.get("entity"), "relation": question_info.get("relation"), "object": question_info.get("object")}
        #print(f"Extracted triplet: {triplets}")
        #print(f"Question info: {question_info}")
        #print(f"Resolved entity: {resolved_entity}")
        #print(f"Entity context: {entity_context}")
        #print(f"Object context: {object_context}")
        #print(" == Context details ==")
        #for key, value in relevant_info.items():
        #    print(f"  {key}: {value}")
        #print(" ====================")

        """ NOT IN USE
        # 1. filter the relevant info to only include the most pertinent facts (this simulates the evidence selection step)
        #filtered_relevant_info = filter_evidence_agent.run(question_info=question_info, all_info=relevant_info)
        #print(f"Filtered relevant info: {filtered_relevant_info}")
        #print("================================")
        """
        #print("atomic question:", question_info.get("atomic_question"))
        #print("==============================================")
        # 1. generate the final answer based on the relevant info
        answer = generate_answer(question_info=question_info, entity_context=entity_context, object_context=object_context, generate_answer_agent=generate_answer_agent)
        #print(f"Generated answer: {answer.get('answer')}")
        #print("================================")

        # 2. map the answer back to the ontology context
        answer_text = answer.get("answer")
        mapped_entity_answer = map_answer_to_context(answer=answer_text, context=entity_context)
        mapped_object_answer = map_answer_to_context(answer=answer_text, context=object_context) if object_context is not None else None
        #mapped_answer, _ = map_to_context_agent.run(answer=answer_text, full_context=entity_context)
        #print(f"Mapped answer: {mapped_answer}")
        #print("================================")

        """ NOT IN USE"""
        """
        # 3. Validate answer with validation agent (not implemented yet)
        validation_result = validate_answer(answer=answer, 
                                            question_info=question_info, 
                                            entity_context=entity_context,
                                            object_context=object_context,
                                            mapped_entity_answer=mapped_entity_answer,
                                            mapped_object_answer=mapped_object_answer)
        """


if __name__ == "__main__":
    main()