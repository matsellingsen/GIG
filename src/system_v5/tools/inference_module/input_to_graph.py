"""This script handles an ATOMIC question/statement and converts it into a graph structure compatible with the current ontology. """
import sys
sys.path.append("C:\\Users\\matse\\gig\\src\\system_v5") # Adjust this path to point to the root of your project if needed

import argparse
import re
from agent_loop.agents.inference_module.extract_triple_agent import ExtractTripleAgent
from backends import load_backend
from tools.ttl_handling.resolve_ttl_path import resolve_ttl_path
from tools.ttl_handling.load_ttl import load_ttl

def atomic_to_graph(atomic_input: str, ttl_path: str = None) -> dict:
    """
    Converts an ATOMIC input string into a graph structure.
    
    Args:
        atomic_input (str): The input in ATOMIC format, e.g., "PersonX goes to the store" or "PersonX wants to buy something."
        
    Returns:
        dict: A graph representation of the input, e.g., {"nodes": [...], "edges": [...]}
    """
    def normalize_and_parse(atomic_input: str) -> dict:
        normalized_input = atomic_input.strip() # remove leading/trailing whitespace
        normalized_input = normalized_input.lower() # convert to lowercase
        normalized_input = re.sub(r"[^\w\s']", "", normalized_input) # remove punctuation except for question marks and apostrophes
        parsed_input = normalized_input.split() # split into tokens
        return parsed_input 

    def extract_concepts_and_relations(parsed_input: list) -> dict:
        pass  

    def map_to_ontology(parsed_input: list) -> dict:
        pass

    """
    # 1. normalize and parse the input
    parsed_input = normalize_and_parse(atomic_input)
    print(f"Parsed input: {parsed_input}")

    # 2. Extract candidate concepts and relations from the parsed input using simple heuristics (e.g., identifying subjects, verbs, objects)
    """
    # 1. pass atomic text to the triple extraction agent and get back a triple (subject-predicate-object)
    triple, _ = triple_extraction_agent.run(atomic_input)
    print("atomic input:", atomic_input)
    print(f"Extracted triple: {triple}")
    print("----------------------------------------")
    # 3. load ttl 
    resolved_ttl_path = resolve_ttl_path(ttl_path)
    ttl = load_ttl(file_path=resolved_ttl_path)
    
    # 3. map parsed input to ontology concepts and relations


def main():
    parser = argparse.ArgumentParser(description="Convert ATOMIC input into a graph structure.")
    parser.add_argument("--backend", default="phi-npu-openvino") # decides which backend to use here.
    args = parser.parse_args()

    backend = load_backend(name=args.backend)

    global triple_extraction_agent
    triple_extraction_agent = ExtractTripleAgent(backend=backend)

    input_str = ["wasn't a sensor small?", "who is Mats Ellingsen?", "Who is Kasper?", "Is a rat subtype of a mammal? ", "what is Sens Motion?", "What activity types are supported by Sens Motion?"]

    for atomic_input in input_str:
        result = atomic_to_graph(atomic_input)

if __name__ == "__main__":
    main()