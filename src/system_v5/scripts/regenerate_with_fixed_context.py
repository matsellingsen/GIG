"""
Final comprehensive regeneration with proper context for Classes.
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
# BETTER CONTEXT BUILDER
# ============================================================================

def get_class_context_for_mapping(entity_uri: URIRef, entity_label: str, graph) -> Dict[str, Any]:
    """
    Build context for a Class entity (suitable for map_answer_to_context).
    
    Returns context with proper superclasses for text matching.
    """
    # Get direct superclasses
    superclasses_list = [
        unquote(str(o).split("#")[-1])
        for _, _, o in graph.triples((entity_uri, RDFS.subClassOf, None))
    ]
    
    return {
        "scope_note": "Context for Class entity suitable for answer mapping.",
        "uri": str(entity_uri),
        "label": entity_label,
        "types": [],
        "superclasses": {entity_label: superclasses_list},  # Group by entity label
        "equivalent_classes": [],
        "properties_by_type": {},
        "annotations": {},
        "class_descriptions": {},
        "object_property_descriptions": {},
        "provenance": []
    }

def get_entity_context_for_mapping(entity_dict: Dict[str, Any], entity_type: str, graph) -> Dict[str, Any]:
    """
    Get context suitable for map_answer_to_context, handling both Classes and Individuals.
    """
    entity_uri = URIRef(entity_dict["uri"])
    entity_label = entity_dict.get("label", "")
    
    if entity_type == "Class":
        # For Classes, use special builder
        return get_class_context_for_mapping(entity_uri, entity_label, graph)
    else:
        # For Individuals, use the standard function
        return retrieve_full_entity_context(entity_dict, graph)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())

def get_entity_info(entity_name: str, graph) -> tuple:
    """
    Resolve entity and return (entity_dict, entity_type, direct_superclasses_for_classes)
    """
    candidates = retrieve_top_candidates(graph, entity_name, top_n=1)
    if not candidates:
        return None, None, []
    
    entity = candidates[0]
    entity_uri = URIRef(entity["uri"])
    entity_type = entity.get("entity_type")
    
    if entity_type == "Class":
        superclasses = [
            unquote(str(o).split("#")[-1])
            for _, _, o in graph.triples((entity_uri, RDFS.subClassOf, None))
        ]
        return entity, entity_type, superclasses
    
    elif entity_type == "NamedIndividual":
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
    
    # DEFINITION
    if question_type == "definition":
        if not superclasses_or_types:
            return "I can't answer the question."
        primary = superclasses_or_types[0]
        return f"{entity} is a {primary}."
    
    # TAXONOMIC/ASSERTION
    elif question_type == "taxonomic" and answer_form == "assertion":
        if not obj:
            return "I can't answer the question."
        obj_norm = normalize(obj)
        match_found = False
        
        for item_class in superclasses_or_types:
            if normalize(item_class) == obj_norm or normalize(item_class).startswith(obj_norm):
                match_found = True
                break
        
        if match_found:
            return f"Yes, {entity} is a {obj}."
        else:
            return f"No, {entity} is not a {obj}."
    
    return "I can't answer the question."

# ============================================================================
# REGENERATE
# ============================================================================

print("Regenerating answers and mappings...")
print()

updated_count = 0
error_count = 0

for idx, item in enumerate(dataset):
    item_id = item.get("id", idx)
    
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
        
        # Resolve entity
        entity_dict, entity_type, superclasses_or_types = get_entity_info(entity_name, graph)
        
        if not entity_dict:
            print(f"  {item_id}: Could not resolve '{entity_name}'")
            error_count += 1
            continue
        
        # Generate answer
        gold_answer = formulate_answer_with_evidence(item, entity_dict, entity_type, superclasses_or_types)
        
        # Get context (using fixed builder)
        context = get_entity_context_for_mapping(entity_dict, entity_type, graph)
        
        # Simulate mapping
        try:
            mapped = pipeline_map_answer_to_context(gold_answer, context)
        except Exception as e:
            mapped = {
                "types": [],
                "superclasses": [],
                "equivalent_classes": [],
                "properties": [],
                "property_values": {}
            }
        
        # Update
        item["gold_answer_agent"] = {
            "reasoning": f"The ontology evidence supports the answer: {gold_answer}",
            "answer": gold_answer
        }
        item["gold_mapped_answer"] = {
            "entity_side": mapped
        }
        
        updated_count += 1
        
    except Exception as e:
        print(f"  {item_id}: Error: {str(e)[:50]}")
        error_count += 1
        continue

print()
print(f"Complete: {updated_count} updated, {error_count} errors")

# ============================================================================
# SAVE
# ============================================================================

print()
print("Saving...")
with open("C:\\Users\\matse\\gig\\src\\system_v5\\tests\\dataset\\ontology_grounded_100.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Saved: ontology_grounded_100.json")

# Sync natural version
print("Syncing natural version...")
with open("C:\\Users\\matse\\gig\\src\\system_v5\\tests\\dataset\\ontology_grounded_100_natural.json", "r") as f:
    natural_dataset = json.load(f)

for nat_item, orig_item in zip(natural_dataset, dataset):
    nat_item["gold_answer_agent"] = orig_item["gold_answer_agent"]
    nat_item["gold_mapped_answer"] = orig_item["gold_mapped_answer"]

with open("C:\\Users\\matse\\gig\\src\\system_v5\\tests\\dataset\\ontology_grounded_100_natural.json", "w") as f:
    json.dump(natural_dataset, f, indent=2)

print("Saved: ontology_grounded_100_natural.json")
print()
print("✅ Complete!")
