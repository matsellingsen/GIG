"""
Final comprehensive script to regenerate gold_answer_agent and gold_mapped_answer.

Strategy:
1. gold_answer_agent: Natural sentence answer plus reasoning
2. gold_mapped_answer: Result of simulating map_answer_to_context
"""

import sys
import json
from rdflib import Graph, RDFS, RDF, OWL, URIRef, Literal
import re
import difflib
from urllib.parse import unquote
from typing import Dict, Any, List

sys.path.insert(0, "C:\\Users\\matse\\gig\\src\\system_v5")

from tools.ttl_handling.load_ttl import load_ttl
from tools.ttl_handling.resolve_ttl_path import resolve_ttl_path
from tools.inference_module.fetch_relevant_info import retrieve_top_candidates, retrieve_full_entity_context
from tools.inference_module.map_answer_to_context import map_answer_to_context as pipeline_map_answer_to_context

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading dataset...")
with open("C:\\Users\\matse\\gig\\src\\system_v5\\tests\\dataset\\ontology_grounded_100.json", "r") as f:
    dataset = json.load(f)

print(f"Dataset: {len(dataset)} items")

print("Loading TTL...")
ttl_path = resolve_ttl_path()
ttl_result = load_ttl(ttl_path)
graph = ttl_result["graph"]

print(f"TTL: {len(graph)} triples")
print()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())

def get_entity_info(entity_name: str, graph) -> tuple:
    """
    Resolve entity and return (entity_dict, entity_type, direct_superclasses_for_classes)
    
    Returns:
        (entity_dict, entity_type, superclasses_or_types)
        where superclasses_or_types is:
        - For Classes: list of direct superclasses
        - For Individuals: list of types
    """
    candidates = retrieve_top_candidates(graph, entity_name, top_n=1)
    if not candidates:
        return None, None, []
    
    entity = candidates[0]
    entity_uri = URIRef(entity["uri"])
    entity_type = entity.get("entity_type")  # "Class" or "NamedIndividual"
    
    if entity_type == "Class":
        # For classes, get direct superclasses
        superclasses = [
            unquote(str(o).split("#")[-1])
            for _, _, o in graph.triples((entity_uri, RDFS.subClassOf, None))
        ]
        return entity, entity_type, superclasses
    
    elif entity_type == "NamedIndividual":
        # For individuals, get types
        types_list = [
            unquote(str(o).split("#")[-1])
            for _, _, o in graph.triples((entity_uri, RDF.type, None))
            if o not in (OWL.Class, OWL.NamedIndividual)
        ]
        return entity, entity_type, types_list
    
    return entity, entity_type, []

def formulate_answer_with_evidence(item: Dict[str, Any], entity_dict: Dict[str, Any], entity_type: str, superclasses_or_types: List[str]) -> str:
    """
    Formulate a natural sentence answer with evidence.
    """
    question_type = item.get("question_type", "")
    answer_form = item.get("answer_form", "")
    
    # Extract from flat structure
    entity_dict_data = item.get("entity", {})
    entity = entity_dict_data.get("value", "") if isinstance(entity_dict_data, dict) else entity_dict_data
    
    relation = item.get("relation", "")
    object_dict = item.get("object", {})
    obj = object_dict.get("value", "") if isinstance(object_dict, dict) else object_dict
    
    # =========================================================================
    # DEFINITION: "What is X?"
    # =========================================================================
    if question_type == "definition":
        if not superclasses_or_types:
            return "I can't answer the question."
        
        primary = superclasses_or_types[0]
        # Evidence sentence: "X is a [type/superclass]"
        return f"{entity} is a {primary}."
    
    # =========================================================================
    # TAXONOMIC/ASSERTION: "Is X a Y?"
    # =========================================================================
    elif question_type == "taxonomic" and answer_form == "assertion":
        if not obj:
            return "I can't answer the question."
        
        obj_norm = normalize(obj)
        match_found = False
        
        # Check if obj matches any in superclasses_or_types
        for item_class in superclasses_or_types:
            if normalize(item_class) == obj_norm or normalize(item_class).startswith(obj_norm):
                match_found = True
                break
        
        if match_found:
            return f"Yes, {entity} is a {obj}."
        else:
            return f"No, {entity} is not a {obj}."
    
    # For other types, return placeholder
    return "I can't answer the question."

# ============================================================================
# REGENERATE ANSWERS AND MAPPINGS
# ============================================================================

print("Regenerating answers and mappings...")
print()

updated_count = 0
error_count = 0

for idx, item in enumerate(dataset):
    item_id = item.get("id", idx)
    
    # Progress indicator
    if (idx + 1) % 20 == 0:
        print(f"[{idx+1}/100] Processed so far...")
    
    try:
        # Check if absent
        ontology_presence = item.get("ontology_presence", "")
        
        if ontology_presence == "absent":
            item["gold_answer_agent"] = {
                "reasoning": "No supporting ontology facts were found for the entity.",
                "answer": "I can't answer the question."
            }
            item["gold_mapped_answer"] = {
                "entity_side": {
                    "types": [],
                    "superclasses": [],
                    "equivalent_classes": [],
                    "properties": [],
                    "property_values": {}
                }
            }
            continue
        
        # Get entity name
        entity_dict_data = item.get("entity", {})
        entity_name = entity_dict_data.get("value", "") if isinstance(entity_dict_data, dict) else entity_dict_data
        
        if not entity_name:
            print(f"  {item_id}: No entity")
            error_count += 1
            continue
        
        # Resolve entity and get context
        entity_dict, entity_type, superclasses_or_types = get_entity_info(entity_name, graph)
        
        if not entity_dict:
            print(f"  {item_id}: Could not resolve '{entity_name}'")
            error_count += 1
            continue
        
        # Generate answer with evidence
        gold_answer = formulate_answer_with_evidence(item, entity_dict, entity_type, superclasses_or_types)
        
        # Get full context for mapping (use retrieve_full_entity_context)
        full_context = retrieve_full_entity_context(entity_dict, graph)
        
        # Simulate map_answer_to_context
        try:
            mapped = pipeline_map_answer_to_context(gold_answer, full_context)
        except Exception as e:
            print(f"  {item_id}: Error in mapping: {str(e)[:50]}")
            mapped = {
                "types": [],
                "superclasses": [],
                "equivalent_classes": [],
                "properties": [],
                "property_values": {}
            }
        
        # Update item
        item["gold_answer_agent"] = {
            "reasoning": f"The ontology evidence supports the answer: {gold_answer}",
            "answer": gold_answer
        }
        item["gold_mapped_answer"] = {
            "entity_side": mapped
        }
        
        updated_count += 1
        
    except Exception as e:
        print(f"  {item_id}: Unexpected error: {str(e)[:100]}")
        error_count += 1
        continue

print()
print(f"Completed: {updated_count} updated, {error_count} errors")

# ============================================================================
# SAVE FILES
# ============================================================================

print()
print("Saving updated dataset...")
with open("C:\\Users\\matse\\gig\\src\\system_v5\\tests\\dataset\\ontology_grounded_100.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Saved: ontology_grounded_100.json")

# Synchronize natural version
print("Synchronizing natural version...")
with open("C:\\Users\\matse\\gig\\src\\system_v5\\tests\\dataset\\ontology_grounded_100_natural.json", "r") as f:
    natural_dataset = json.load(f)

for nat_item, orig_item in zip(natural_dataset, dataset):
    nat_item["gold_answer_agent"] = orig_item["gold_answer_agent"]
    nat_item["gold_mapped_answer"] = orig_item["gold_mapped_answer"]

with open("C:\\Users\\matse\\gig\\src\\system_v5\\tests\\dataset\\ontology_grounded_100_natural.json", "w") as f:
    json.dump(natural_dataset, f, indent=2)

print("Saved: ontology_grounded_100_natural.json")

print()
print("✅ Regeneration complete!")
