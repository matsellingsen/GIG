"""This script handles an ATOMIC question/statement and converts it into a graph structure compatible with the current ontology. """
import sys
sys.path.append("C:\\Users\\matse\\gig\\src\\system_v5") # Adjust this path to point to the root of your project if needed

import argparse
import re

from agent_loop.agents.inference_module.extract_entity_agent import ExtractEntityAgent
from agent_loop.agents.inference_module.extract_relation_agent import ExtractRelationAgent
from agent_loop.agents.inference_module.extract_question_type_agent import ExtractQuestionTypeAgent
from agent_loop.agents.inference_module.extract_object_agent import ExtractObjectAgent
from agent_loop.agents.inference_module.resolve_answer_form_agent import ResolveAnswerFormAgent

from backends import load_backend
from tools.ttl_handling.resolve_ttl_path import resolve_ttl_path
from tools.ttl_handling.load_ttl import load_ttl
from tools.inference_module.fetch_relevant_info import fetch_relevant_info


def atomic_to_graph(atomic_input: str, extract_question_type_agent: ExtractQuestionTypeAgent, extract_answer_form_agent: ResolveAnswerFormAgent, extract_entity_agent: ExtractEntityAgent, extract_relation_agent: ExtractRelationAgent, extract_object_agent: ExtractObjectAgent, inference_log: dict=None):
    """
    Converts an ATOMIC input string into a triplet + question-type & answer-form.
    Returns dict on success, error string on failure.
    
    Args:
        atomic_input (str): The input in ATOMIC format, e.g., "PersonX goes to the store" or "PersonX wants to buy something."
        
    Returns:
        dict: a dictionary containing the subject, predicate, object, question-type and answer-form.
        str: error message if any step fails.
    """
    
    # 1. Extract question type, answer form, entity, relation, and object candidates from the atomic input using the respective agents.
    question_type, _ = extract_question_type_agent.run(atomic_input)
    if inference_log:
        inference_log["resolved_question_type"] = question_type
    print(f"Extracted question type: {question_type}")

    if not question_type: # Handle empty or None question type
        return "Failed to extract a question type from the input."
    
    answer_form, _ = extract_answer_form_agent.run(atomic_input, question_type=question_type)
    if inference_log:
        inference_log["resolved_answer_form"] = answer_form
    print(f"Extracted answer form: {answer_form}")
    if not answer_form: # Handle empty or None answer form
        return "Failed to extract an answer form from the input."
    
    entity, _ = extract_entity_agent.run(atomic_input, question_classification=question_type)
    if inference_log:
        inference_log["extracted_entity"] = entity
    print(f"Extracted entity: {entity}")
    if not entity: # Handle empty or None entity
        return "Failed to extract an entity from the input."
    
    relation, _ = extract_relation_agent.run(atomic_input, entity_candidate=entity, question_classification=question_type)
    if inference_log:
        inference_log["extracted_relation"] = relation
    print(f"Extracted relation: {relation}")
    if not relation: # Handle empty or None relation
        return "Failed to extract a relation from the input."
    
    obj, _ = extract_object_agent.run(atomic_input, entity_candidate=entity, relation_candidate=relation, question_classification=question_type, answer_form=answer_form)
    if inference_log:
        inference_log["extracted_object"] = obj
    print(f"Extracted object: {obj}")
    if not obj: # Handle empty or None object
        return "Failed to extract an object from the input."
    print("================================")

    #2. Structure the extracted information as a dict with the 5 extracted components + the atomic question.
    result = {
        "atomic_question": atomic_input,
        "question_type": question_type.get("question_type"),
        "answer_form": answer_form.get("answer_form"),
        "entity": {"value": entity.get("primary_entity"), "type": entity.get("entity_type")},
        "relation": relation.get("relation"),
        "object": {"value": obj.get("object"), "type": obj.get("object_type")}
    }
    return result

def main():
    parser = argparse.ArgumentParser(description="Convert ATOMIC input into a graph structure.")
    parser.add_argument("--backend", default="phi-npu-openvino") # decides which backend to use here.
    args = parser.parse_args()

    backend = load_backend(name=args.backend)

    global triple_extraction_agent
    global extract_subject_agent
    global extract_answer_form_agent
    triple_extraction_agent = ExtractTripleAgent(backend=backend)
    extract_subject_agent = ExtractSubjectAgent(backend=backend)
    extract_answer_form_agent = ResolveAnswerFormAgent(backend=backend)
    extract_entity_agent = ExtractEntityAgent(backend=backend)
    extract_relation_agent = ExtractRelationAgent(backend=backend)
    extract_question_type_agent = ExtractQuestionTypeAgent(backend=backend)
    extract_object_agent = ExtractObjectAgent(backend=backend)

    input_str = ["What activity types are included in Sens Motion?", "How small are the sensors?", "is every employee a person?", "who is Mats Ellingsen?", "Who is Kasper?", "Is a rat subtype of a mammal? ", "what is Sens Motion?"]

    for atomic_input in input_str:
        # 1. Extract information from the atomic input using the respective agents.
        result = atomic_to_graph(atomic_input, extract_question_type_agent, extract_answer_form_agent, extract_entity_agent, extract_relation_agent, extract_object_agent)


        # Maybe move this to a different script that takes the result from here.
        # 2. load ttl 
        resolved_ttl_path = resolve_ttl_path(ttl_path=None)
        ttl = load_ttl(file_path=resolved_ttl_path)

        # 3. Fetch relevant information from the TTL based on the extracted question information.
        relevant_info = fetch_relevant_info(question_info=result, ttl=ttl)

if __name__ == "__main__":
    main()