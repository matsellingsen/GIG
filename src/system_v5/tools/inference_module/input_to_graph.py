"""This script handles an ATOMIC question/statement and converts it into a graph structure compatible with the current ontology. """
import sys
sys.path.append("C:\\Users\\matse\\gig\\src\\system_v5") # Adjust this path to point to the root of your project if needed

import argparse
import re

from agent_loop.agents.inference_module.extract_triple_agent import ExtractTripleAgent
from agent_loop.agents.inference_module.extract_subject_agent import ExtractSubjectAgent
from agent_loop.agents.inference_module.extract_entity_agent import ExtractEntityAgent
from agent_loop.agents.inference_module.extract_relation_agent import ExtractRelationAgent
from agent_loop.agents.inference_module.extract_question_type_agent import ExtractQuestionTypeAgent
from agent_loop.agents.inference_module.extract_object_agent import ExtractObjectAgent

from backends import load_backend
from tools.ttl_handling.resolve_ttl_path import resolve_ttl_path
from tools.ttl_handling.load_ttl import load_ttl
from tools.inference_module.fetch_relevant_info import fetch_relevant_info


def atomic_to_graph(atomic_input: str, ttl_path: str = None) -> dict:
    """
    Converts an ATOMIC input string into a triplet + question-type.
    
    Args:
        atomic_input (str): The input in ATOMIC format, e.g., "PersonX goes to the store" or "PersonX wants to buy something."
        
    Returns:
        dict: a dictionary containing the subject, predicate, object and question-type.
    """
    
    print("atomic input:", atomic_input)

    # 1. Extract question type, entity, relation, and object candidates from the atomic input using the respective agents.
    question_type, _ = extract_question_type_agent.run(atomic_input)
    print(f"Extracted question type: {question_type}")
    entity, _ = extract_entity_agent.run(atomic_input, question_classification=question_type)
    print(f"Extracted entity: {entity}")
    relation, _ = extract_relation_agent.run(atomic_input, entity_candidate=entity, question_classification=question_type)
    print(f"Extracted relation: {relation}")
    obj, _ = extract_object_agent.run(atomic_input, entity_candidate=entity, relation_candidate=relation, question_classification=question_type)
    print(f"Extracted object: {obj}")

    #2. Structure the extracted information as a triple with the 4 extracted components.
    result = {
        "question_type": question_type.get("question_type"),
        "entity": {"value": entity.get("entity"), "type": entity.get("entity_type")},
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
    global extract_entity_agent
    global extract_relation_agent
    global extract_question_type_agent
    global extract_object_agent
    triple_extraction_agent = ExtractTripleAgent(backend=backend)
    extract_subject_agent = ExtractSubjectAgent(backend=backend)
    extract_entity_agent = ExtractEntityAgent(backend=backend)
    extract_relation_agent = ExtractRelationAgent(backend=backend)
    extract_question_type_agent = ExtractQuestionTypeAgent(backend=backend)
    extract_object_agent = ExtractObjectAgent(backend=backend)

    input_str = ["What activity types are included in Sens Motion?", "How small are the sensors?", "is every employee a person?", "who is Mats Ellingsen?", "Who is Kasper?", "Is a rat subtype of a mammal? ", "what is Sens Motion?"]

    for atomic_input in input_str:
        # 1. Extract information from the atomic input using the respective agents.
        result = atomic_to_graph(atomic_input)


        # Maybe move this to a different script that takes the result from here.
        # 2. load ttl 
        resolved_ttl_path = resolve_ttl_path(ttl_path=None)
        ttl = load_ttl(file_path=resolved_ttl_path)

        # 3. Fetch relevant information from the TTL based on the extracted question information.
        relevant_info = fetch_relevant_info(question_info=result, ttl=ttl)

if __name__ == "__main__":
    main()