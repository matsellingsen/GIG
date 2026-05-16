import re
import difflib
from typing import List, Dict, Any

# ---------------------------------------------------------
# Merging mappings utility
# ---------------------------------------------------------
def merge_mappings(map_reasoning: Dict[str, Any],
                   map_answer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministically merge the reasoning-based mapping and the answer-based mapping.
    - Lists are merged by set-union and sorted.
    - Dicts (e.g., property_values) are merged with answer overriding reasoning.
    - Missing keys are treated as empty.
    """

    merged = {}

    all_keys = set(map_reasoning.keys()).union(map_answer.keys())

    for key in all_keys:
        v_r = map_reasoning.get(key)
        v_a = map_answer.get(key)

        # Case 1: list-like fields (types, superclasses, eq_classes, annotation_labels, properties)
        if isinstance(v_r, list) or isinstance(v_a, list):
            merged[key] = sorted(set((v_r or []) + (v_a or [])))
            continue

        # Case 2: dict-like fields (property_values)
        if isinstance(v_r, dict) or isinstance(v_a, dict):
            merged[key] = {**(v_r or {}), **(v_a or {})}
            continue

        # Case 3: fallback (rare)
        merged[key] = v_a if v_a is not None else v_r

    return merged

# ---------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def tokenize_chunks(answer: str, max_n: int = 4) -> List[str]:
    """
    Tokenize the answer into 1–4 token chunks.
    Example: ["monitoring", "system", "monitoring system", ...]
    """
    tokens = normalize(answer).split()
    chunks = []

    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            chunk = " ".join(tokens[i:i+n])
            chunks.append(chunk)

    return chunks


# ---------------------------------------------------------
# Matching utilities
# ---------------------------------------------------------

def exact_match(chunk: str, candidate: str) -> bool:
    return normalize(chunk) == normalize(candidate)


def fuzzy_match(chunk: str, candidate: str, threshold: float = 0.85) -> bool:
    """
    Chunk-level fuzzy match.
    Threshold tuned to avoid false positives like "part of a collection" → isPartOf.
    """
    chunk_norm = normalize(chunk)
    cand_norm = normalize(candidate)
    ratio = difflib.SequenceMatcher(None, chunk_norm, cand_norm).ratio()
    return ratio >= threshold


def match_candidates(answer: str, candidates: List[str]) -> List[str]:
    """
    Perform exact + fuzzy matching using 1–4 token chunks.
    """
    chunks = tokenize_chunks(answer)
    matches = set()

    # Exact matches
    for c in candidates:
        for chunk in chunks:
            if exact_match(chunk, c):
                matches.add(c)

    # Fuzzy matches
    for c in candidates:
        if c not in matches:
            for chunk in chunks:
                if fuzzy_match(chunk, c):
                    matches.add(c)

    return sorted(matches)

# ---------------------------------------------------------
# Class description keyword extraction
# ---------------------------------------------------------
def extract_ntoken_chunks(text: str, n: int = 4) -> List[str]:
    """
    Extract 4-token chunks from normalized text.
    """
    tokens = normalize(text).split()
    chunks = []

    for i in range(len(tokens) - n + 1):
        chunk = " ".join(tokens[i:i+n])
        chunks.append(chunk)

    return chunks

def extract_chunks_from_class_descriptions(context: Dict[str, Any], chunk_size: int = 4):
    """
    Extract chunks from class descriptions for fuzzy matching.
    """
    types = context.get("types", [])
    superclasses = []
    for vals in context.get("superclasses", {}).values():
        superclasses.extend(vals)
    eq_classes = context.get("equivalent_classes", [])

    types_chunks = {}
    superClasses_chunks = {}
    eqClasses_chunks = {}
    for cls, desc in context.get("class_descriptions", {}).items():
        if not desc or not desc.get("description"):
            continue

        text = normalize(desc["description"])
        chunks = extract_ntoken_chunks(text, n=chunk_size)

        if cls in types:
            types_chunks[cls] = chunks

        if cls in superclasses:
            superClasses_chunks[cls] = chunks

        if cls in eq_classes:
            eqClasses_chunks[cls] = chunks

    return types_chunks, superClasses_chunks, eqClasses_chunks

def extract_chunks_from_annotation_comments(context: Dict[str, Any], chunk_size: int = 4):
    """
    Extract chunks from annotation comments for fuzzy matching.
    """
    annotations = context.get("annotations", {})
    annotation_label = annotations.get("label", "")
    annotation_comment = annotations.get("comment", "")
    annotation_chunks = {}
    
    text = normalize(annotation_comment)
    chunks = extract_ntoken_chunks(text, n=chunk_size)
    annotation_chunks[annotation_label] = chunks

    return annotation_chunks

def match_class_description_chunks(answer: str, chunk_map: Dict[str, List[str]]) -> List[str]:
    answer_chunks = tokenize_chunks(answer)
    matches = []

    for cls, chunks in chunk_map.items():
        for chunk in chunks:
            for answer_chunk in answer_chunks:
                if fuzzy_match(chunk, answer_chunk, threshold=0.9):
                    matches.append(cls)
                    break

    return matches


# ---------------------------------------------------------
# Main deterministic mapping function
# ---------------------------------------------------------

def map_answer_to_context(answer: str, context: Dict[str, Any], inference_log: Dict[str, Any] = None):
    """Map answer to context. Returns dict on success, error string on failure."""
    # Validate inputs
    if not answer or not isinstance(answer, str):
        error_msg = "Invalid answer provided to map_answer_to_context: answer must be a non-empty string."
        if inference_log is not None:
            inference_log["error"] = error_msg
        return error_msg
    
    if not context or not isinstance(context, dict):
        error_msg = "Invalid context provided to map_answer_to_context: context must be a non-empty dict."
        if inference_log is not None:
            inference_log["error"] = error_msg
        return error_msg
    
    result = {}

    # Precompute super-class description keywords
    types_keywords, superClasses_keywords, eqClasses_keywords = extract_chunks_from_class_descriptions(context, chunk_size=4)

    # Precompute annotation keywords
    annotation_keywords = extract_chunks_from_annotation_comments(context, chunk_size=4)

    # -----------------------------------------------------
    # 1. TYPES
    # -----------------------------------------------------
    types = context.get("types", [])
    type_matches = match_candidates(answer, types)
    type_matches += match_class_description_chunks(answer, types_keywords)
    result["types"] = sorted(set(type_matches))

    # -----------------------------------------------------
    # 2. SUPERCLASSES
    # -----------------------------------------------------
    superclasses = []
    for vals in context.get("superclasses", {}).values():
        superclasses.extend(vals)
    superclass_matches = match_candidates(answer, superclasses)
    superclass_matches += match_class_description_chunks(answer, superClasses_keywords)
    result["superclasses"] = sorted(set(superclass_matches))

    # -----------------------------------------------------
    # 3. EQUIVALENT CLASSES
    # -----------------------------------------------------
    eq_classes = context.get("equivalent_classes", [])
    eq_matches = match_candidates(answer, eq_classes)
    #eq_matches += match_class_description_keywords(answer, eqClasses_keywords)
    eq_matches += match_class_description_chunks(answer, eqClasses_keywords)    
    result["equivalent_classes"] = sorted(set(eq_matches))

    # -----------------------------------------------------
    # 4. PROPERTIES (value-first deterministic mapping)
    # -----------------------------------------------------

    prop_matches = set()
    value_matches = {}  # optional: map property → matched value

    answer_norm = normalize(answer)

    # Collect all property-value pairs
    property_values = []  # list of (property_name, value_string)

    for t, pdata in context.get("properties_by_type", {}).items():

        # Outgoing object properties
        for item in pdata.get("outgoing_object_properties", []):
            property_values.append((item["property"], normalize(item["object"])))

        # Outgoing data properties
        for item in pdata.get("outgoing_data_properties", []):
            property_values.append((item["property"], normalize(item["value"])))

        # Incoming object properties
        for item in pdata.get("incoming_object_properties", []):
            property_values.append((item["property"], normalize(item["subject"])))

        # Incoming data properties
        for item in pdata.get("incoming_data_properties", []):
            property_values.append((item["property"], normalize(item["value"])))


    # Match values and property-names in the answer
    for prop, val in property_values:
        if prop == "providesBatteryEstimation":
            print(f"Checking property '{prop}' with value '{val}' against answer: '{answer}'")

        # normalize property and value for matching
        prop_norm = normalize(prop)
        val_norm = normalize(val)

        # Exact match
        if val_norm and val_norm in answer_norm:
            prop_matches.add(prop)
            value_matches[prop] = val
            continue
        if prop_norm and prop_norm in answer_norm:
            prop_matches.add(prop)
            value_matches[prop] = val
            continue

        # Fuzzy match (only if value is long enough)
        if len(val_norm) > 5:
            ratio = difflib.SequenceMatcher(None, answer_norm, val_norm).ratio()
            if ratio >= 0.85:
                prop_matches.add(prop)
                value_matches[prop] = val
        if len(prop_norm) >= 5: 
            ratio = difflib.SequenceMatcher(None, answer_norm, prop_norm).ratio()
            if ratio >= 0.85:
                prop_matches.add(prop)
                value_matches[prop] = val


    # Disambiguation: avoid false positives like "part of a collection" → isPartOf
    if "isPartOf" in prop_matches and "collection" in answer_norm:
        prop_matches.remove("isPartOf")
        value_matches.pop("isPartOf", None)

    result["properties"] = sorted(prop_matches)
    result["property_values"] = value_matches  # optional but extremely useful

    # -----------------------------------------------------
    # 5. ANNOTATIONS (entity-level labels/comments)
    # -----------------------------------------------------
    annotation_matches = match_class_description_chunks(answer, annotation_keywords)
    result["annotations"] = sorted(set(annotation_matches))

    # -----------------------------------------------------
    # 6 MEMBERS (if the resolved entity is a class)
    # -----------------------------------------------------
    member_matches = []
    for member in context.get("members", []):
        if member in answer:
            member_matches.append(member)
    result["members"] = sorted(set(member_matches))

    # ------------------------------------------------------
    # 7. CHUNK_ID (provenance)
    # ------------------------------------------------------
    chunk_id_matches = []
    for cid in context.get("chunk_id", []):
        if cid in answer:
            chunk_id_matches.append(cid)
    result["chunk_id"] = sorted(set(chunk_id_matches))

    # 7. OBJECT PROPERTY DESCRIPTIONS
    # -----------------------------------------------------
    #obj_prop_keys = list(context.get("object_property_descriptions", {}).keys())
    #obj_prop_matches = match_candidates(answer, obj_prop_keys)

    # Disambiguation: "part of a collection" ≠ isPartOf
    #if "isPartOf" in obj_prop_matches and "collection" in normalize(answer):
    #    obj_prop_matches.remove("isPartOf")

    #result["object_property_descriptions"] = obj_prop_matches

    # Log mapping summary for diagnostics
    if inference_log is not None:
        inference_log.setdefault("map_answer_to_context", []).append({
            "answer": answer,
            "types": result.get("types"),
            "superclasses": result.get("superclasses"),
            "equivalent_classes": result.get("equivalent_classes"),
            "properties": result.get("properties"),
            "property_values": result.get("property_values"),
            "annotations": result.get("annotations"),
            "members": result.get("members"),
            "chunk_id": result.get("chunk_id")
        })

    return result  # Return dict (success) or error string (failure)
