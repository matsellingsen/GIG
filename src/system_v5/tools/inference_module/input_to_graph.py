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
from rdflib import RDFS, RDF, Literal
from urllib.parse import unquote
from difflib import SequenceMatcher


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    s = unquote(s)
    s = s.replace('_', ' ')
    # remove punctuation except spaces and alphanumerics
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _labels_for_individual(graph, indiv) -> list:
    labels = []
    # rdfs:label preferred
    for o in graph.objects(indiv, RDFS.label):
        labels.append(str(o))
      
    # fallback to URI local name
    if not labels:
        uri = str(indiv)
        local = uri.split('#')[-1] if '#' in uri else uri.split('/')[-1]
        local = unquote(local)
        if len(_normalize_text(local)) >= 2:
            labels.append(local)

    # include short literal properties as additional textual cues
    for p, o in graph.predicate_objects(indiv):
        if isinstance(o, Literal):
            text = str(o).strip()
            norm = _normalize_text(text)
            # only keep normalized labels of length >= 2 with an alnum character
            if norm and len(norm) >= 2 and re.search(r'\w', norm) and len(text) <= 120:
                labels.append(text)
    
    # dedupe while preserving order
    seen = set()
    out = []
    for l in labels:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out


def _best_label_score(query: str, label: str) -> float:
    q = _normalize_text(query)
    l = _normalize_text(label)
    if not q or not l:
        return 0.0
    if q == l:
        return 1.0
    # substring containment is a strong signal
    if q in l or l in q:
        return 0.9
    # fuzzy ratio as fallback
    ratio = SequenceMatcher(None, q, l).ratio()
    return float(ratio)


def _score_individual(graph, indiv, query: str) -> dict:
    best_label = ""
    best_score = 0.0
    for lab in _labels_for_individual(graph, indiv):
        s = _best_label_score(query, lab)
        if s > best_score:
            best_score = s
            best_label = lab
    types = [str(t) for t in graph.objects(indiv, RDF.type)]
    return {"uri": str(indiv), "label": best_label, "score": round(best_score, 4), "types": types}

def atomic_to_graph(atomic_input: str, ttl_path: str = None) -> dict:
    """
    Converts an ATOMIC input string into a triplet.
    
    Args:
        atomic_input (str): The input in ATOMIC format, e.g., "PersonX goes to the store" or "PersonX wants to buy something."
        
    Returns:
        dict: a dictionary containing the subject, predicate, object and confidence score.
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
    print("atomic input:", atomic_input)
  
    question_type, _ = extract_question_type_agent.run(atomic_input)
    print(f"Extracted question type: {question_type}")

    entity, _ = extract_entity_agent.run(atomic_input, question_classification=question_type)
    print(f"Extracted entity: {entity}")

    relation, _ = extract_relation_agent.run(atomic_input, entity_candidate=entity, question_classification=question_type)
    print(f"Extracted relation: {relation}")

    obj, _ = extract_object_agent.run(atomic_input, entity_candidate=entity, relation_candidate=relation, question_classification=question_type)
    print(f"Extracted object: {obj}")
    
    #subject, _ = extract_subject_agent.run(atomic_input)
    #print(f"Extracted subject: {subject}")
    # 1. pass atomic text to the triple extraction agent and get back a triple (subject-predicate-object)
    #triple, _ = triple_extraction_agent.run(atomic_input)
    
    #print(f"Extracted triple: {triple}")
    print("----------------------------------------")
    triple = None
    return triple

def prune_ontology_by_predicate(triple: dict, ttl: dict) -> dict:
    """Prunes the ontology to only include classes and instances relevant to the predicate of the triple, potentially using the confidence score to set a threshold for how aggressive the pruning should be."""
    predicate = triple.get("predicate")

    # If the predicate is 'unknown' or confidence is low, return an error saying we don't understand the input well enough to provide an answer.
    if predicate == "unknown" or triple.get("confidence", 0) < 0.5:
        raise ValueError("Unable to confidently extract a predicate from the input. Please rephrase your question or provide more context.")
    
    # Otherwise, filter the TTL to only include classes and instances that are relevant to the extracted predicate.


def map_to_ontology(triple: dict, ttl: dict, top_k: int = 3) -> dict:
    """Map triple `subject` and `object` strings to top-k individual candidates from the TTL.

    Returns a dict with `subject_candidates` and `object_candidates`, each a list
    of {uri, label, score, types} sorted by score descending.
    """
    if not isinstance(ttl, dict) or 'graph' not in ttl:
        raise ValueError("ttl must be a dict with a 'graph' key (as returned by load_ttl)")

    g = ttl['graph']
    individuals = ttl.get('individuals', set())

    subj_text = triple.get('subject', '') if triple else ''
    obj_text = triple.get('object', '') if triple else ''

    def rank_slot(query_text: str):
        candidates = []
        q = (query_text or '').strip()
        if not q or q.lower() in {"unknown", "x", "y"}:
            return []
        for ind in individuals:
            scored = _score_individual(g, ind, q)
            if scored['score'] > 0:
                candidates.append(scored)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]

    subject_candidates = rank_slot(subj_text)
    object_candidates = rank_slot(obj_text)

    return {
        'subject_candidates': subject_candidates,
        'object_candidates': object_candidates
    }

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

    input_str = ["How small are the sensors?", "is every employee a person?", "who is Mats Ellingsen?", "Who is Kasper?", "Is a rat subtype of a mammal? ", "what is Sens Motion?", "What activity types are included in Sens Motion?"]

    for atomic_input in input_str:
        # 1. Extract triple from the atomic input using the ExtractTripleAgent
        triple = atomic_to_graph(atomic_input)

        """
        # 2. load ttl 
        resolved_ttl_path = resolve_ttl_path(ttl_path=None)
        ttl = load_ttl(file_path=resolved_ttl_path)

        # 3. prune ontology based on predicate and confidence score
        #pruned_ttl = prune_ontology_by_predicate(triple, ttl)
        pruned_ttl = ttl # for now we skip pruning to test the mapping step with the full ontology. We can add pruning back in later once the mapping step is working end-to-end.

        # 4. map subject and object of the triple to ontology classes and instances
        mapped_triple = map_to_ontology(triple, pruned_ttl)
        for subj_cand in mapped_triple.get('subject_candidates', []):
            print("Subject candidate:", subj_cand.get("label"))
        print("-------------------------------")
        for obj_cand in mapped_triple.get('object_candidates', []):
            print("Object candidate:", obj_cand.get("label"))
        print("========================================")
        """

if __name__ == "__main__":
    main()