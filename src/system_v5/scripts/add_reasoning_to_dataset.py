"""
Add a reasoning field (and agent-style answer object) to each dataset item.

Creates `gold_answer_agent` = {"answer": <existing string>, "reasoning": <generated reasoning>}.
Also synchronizes the natural dataset.
"""
import sys
import json
import re
from urllib.parse import unquote

sys.path.insert(0, "C:\\Users\\matse\\gig\\src\\system_v5")
from tools.ttl_handling.load_ttl import load_ttl
from tools.ttl_handling.resolve_ttl_path import resolve_ttl_path

from tools.inference_module.fetch_relevant_info import retrieve_top_candidates, retrieve_full_entity_context

# Load dataset
MAIN_PATH = "C:\\Users\\matse\\gig\\src\\system_v5\\tests\\dataset\\ontology_grounded_100.json"
NATURAL_PATH = "C:\\Users\\matse\\gig\\src\\system_v5\\tests\\dataset\\ontology_grounded_100_natural.json"

with open(MAIN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(NATURAL_PATH, "r", encoding="utf-8") as f:
    natural = json.load(f)

# Load TTL
ttl_path = resolve_ttl_path()
ttl_result = load_ttl(ttl_path)
graph = ttl_result["graph"]

# Helpers

def normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def build_reasoning_from_mapping(entity_name: str, mapping: dict, full_context: dict) -> str:
    """Construct a concise reasoning paragraph from mapping+context."""
    parts = []
    entity_disp = entity_name or full_context.get("label", "the entity")

    types = mapping.get("types", []) or []
    superclasses = mapping.get("superclasses", []) or []
    eq = mapping.get("equivalent_classes", []) or []
    prop_vals = mapping.get("property_values", {}) or {}

    # If superclasses present, prefer that
    if superclasses:
        sc = superclasses[0]
        parts.append(f"The ontology lists '{entity_disp}' with a superclass '{sc}', which supports classifying {entity_disp} as a {sc}.")
        if len(superclasses) > 1:
            others = ", ".join(superclasses[1:])
            parts.append(f"Other related superclasses found: {others}.")
    elif types:
        t0 = types[0]
        parts.append(f"The ontology assigns type(s) {', '.join(types)} to '{entity_disp}', supporting classification as {t0}.")
    elif eq:
        parts.append(f"The ontology contains equivalent class(es) {', '.join(eq)} linked to '{entity_disp}', suggesting the same classification.")

    # Property evidence
    if prop_vals:
        pv_parts = []
        for p, v in prop_vals.items():
            pv_parts.append(f"'{p}' = '{v}'")
        parts.append("Property assertions supporting the answer: " + ", ".join(pv_parts) + ".")

    # Fallback: use labels/annotations from full_context
    if not parts:
        ann = full_context.get("annotations", {})
        labels = ann.get("http://www.w3.org/2000/01/rdf-schema#label") if isinstance(ann, dict) else None
        if labels:
            parts.append(f"The entity has labels {labels}, but no explicit type/superclass was found in the extracted mapping.")
        else:
            parts.append(f"No explicit supporting ontology facts were found for '{entity_disp}'.")

    reasoning = " ".join(parts)
    # Make it more like the example: two sentences, first about classification, second about subclass/individual
    return reasoning

# Process items
updated = 0
for i, item in enumerate(data):
    entity_dict = item.get("entity", {})
    entity_name = entity_dict.get("value") if isinstance(entity_dict, dict) else entity_dict

    # Ensure we have a full_context for fallback
    context = None
    try:
        candidates = retrieve_top_candidates(graph, entity_name or "", top_n=1)
        if candidates:
            context = retrieve_full_entity_context(candidates[0], graph)
        else:
            context = {}
    except Exception:
        context = {}

    mapping = (item.get("gold_mapped_answer") or {}).get("entity_side") or {}

    reasoning = build_reasoning_from_mapping(entity_name, mapping, context)

    # Build agent-style answer object
    existing_answer = (
        (item.get("gold_answer_agent") or {}).get("answer")
        or "I can't answer the question."
    )
    agent_obj = {
        "reasoning": reasoning,
        "answer": existing_answer
    }

    item["gold_answer_agent"] = agent_obj
    updated += 1

# Sync natural dataset
for nd_item, main_item in zip(natural, data):
    nd_item["gold_answer_agent"] = main_item["gold_answer_agent"]

# Save files
with open(MAIN_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

with open(NATURAL_PATH, "w", encoding="utf-8") as f:
    json.dump(natural, f, indent=2, ensure_ascii=False)

print(f"Updated {updated} items with agent-style reasoning and saved datasets.")
