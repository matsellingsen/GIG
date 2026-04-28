import re
import difflib
from typing import List, Dict, Any


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

def map_answer_to_context(answer: str, context: Dict[str, Any]) -> Dict[str, Any]:
    result = {}

    # Precompute super-class description keywords
    types_keywords, superClasses_keywords, eqClasses_keywords = extract_chunks_from_class_descriptions(context, chunk_size=4)

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
    #superclasses = [val for val in vals for vals in context.get("superclasses", {}).values()]
    print(f"Superclasses candidates: {superclasses}")
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


    # Match values in the answer
    for prop, val in property_values:

        # Exact match
        if val and val in answer_norm:
            prop_matches.add(prop)
            value_matches[prop] = val
            continue

        # Fuzzy match (only if value is long enough)
        if len(val) > 5:
            ratio = difflib.SequenceMatcher(None, answer_norm, val).ratio()
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
    # 6. OBJECT PROPERTY DESCRIPTIONS
    # -----------------------------------------------------
    obj_prop_keys = list(context.get("object_property_descriptions", {}).keys())
    obj_prop_matches = match_candidates(answer, obj_prop_keys)

    # Disambiguation: "part of a collection" ≠ isPartOf
    if "isPartOf" in obj_prop_matches and "collection" in normalize(answer):
        obj_prop_matches.remove("isPartOf")

    result["object_property_descriptions"] = obj_prop_matches

    return result
