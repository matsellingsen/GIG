"""Convert an ATOMIC question into a candidate triple and ground the predicate
to a base-ontology axiom using a rule-first approach with an LLM fallback.

This file implements small, well-delineated functions so the extractor can
be tested independently and integrated into the existing pipeline.
"""

import sys
sys.path.append("C:\\Users\\matse\\gig\\src\\system_v5")

import argparse
import re
import difflib
import logging
from typing import Dict, List, Tuple, Optional

from agent_loop.agents.inference_module.extract_triple_agent import ExtractTripleAgent
from backends import load_backend
from tools.base_ontology.load_base_ontology import load_classes, load_axioms
from tools.ttl_handling.resolve_ttl_path import resolve_ttl_path
from tools.ttl_handling.load_ttl import load_ttl

LOG = logging.getLogger(__name__)

# Module-level cache for base ontology indexes
_BASE_INDEXES: Optional[Dict] = None


def load_base_indexes() -> Dict:
    """Load base ontology classes and axioms and build lookup indexes.

    Returns a dictionary with:
      - classes: dict[class_name -> description]
      - properties: dict[normalized_label -> axiom_dict]
      - raw_property_labels: list of original property labels (for fuzzy matching)
    """
    global _BASE_INDEXES
    if _BASE_INDEXES is not None:
        return _BASE_INDEXES

    classes = load_classes()
    axioms = load_axioms()

    properties = {}
    raw_property_labels = []

    def keyify(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    for ax in axioms:
        label = ax.get("type", "").strip()
        if not label:
            continue
        lower = label.lower()
        properties[lower] = ax
        # also add simplified key for substring/compact matches
        properties[keyify(lower)] = ax
        raw_property_labels.append(lower)

    _BASE_INDEXES = {
        "classes": classes,
        "properties": properties,
        "raw_property_labels": raw_property_labels,
    }
    return _BASE_INDEXES


QUESTION_WORDS = ["who", "what", "where", "when", "why", "how", "how many"]


def normalize_label(text: str) -> str:
    """Return a minimal normalized form for matching: lowercased, stripped,
    punctuation removed (except inner apostrophes).
    """
    if text is None:
        return ""
    t = text.strip().lower()
    # keep letters/digits and spaces and apostrophes
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()



def generate_predicate_candidates(predicate_text: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
    """Generate a ranked list of (axiom, lexical_score) candidates for the
    given predicate_text using the base ontology properties index.
    """
    idx = load_base_indexes()
    properties = idx["properties"]
    raw_labels = idx["raw_property_labels"]

    cleaned = normalize_label(predicate_text or "")
    if not cleaned:
        return []

    # synonyms map (minimal starter set)
    synonyms = {
        "located": "haslocation",
        "location": "haslocation",
        "where": "haslocation",
        "has": "hasmember",
        "contains": "hasmember",
        "member": "hasmember",
        "is": "subclassof",
        "type": "classifies",
        "executes": "executesTask",
        "performs": "executesTask",
        "realizes": "realizes",
    }

    key = re.sub(r"[^a-z0-9]", "", cleaned.lower())
    candidates: List[Tuple[Dict, float]] = []

    # exact normalized match
    if cleaned in properties:
        ax = properties[cleaned]
        candidates.append((ax, 1.0))

    # keyified exact match
    if key in properties:
        ax = properties[key]
        candidates.append((ax, 0.95))

    # synonyms
    for syn, mapped in synonyms.items():
        if syn in cleaned:
            mapped_key = mapped.lower()
            if mapped_key in properties:
                candidates.append((properties[mapped_key], 0.9))

    # fuzzy matches using difflib on raw labels
    matches = difflib.get_close_matches(cleaned, raw_labels, n=top_k, cutoff=0.5)
    for m in matches:
        score = difflib.SequenceMatcher(a=cleaned, b=m).ratio()
        ax = properties.get(m.lower()) or properties.get(re.sub(r"[^a-z0-9]", "", m.lower()))
        if ax:
            candidates.append((ax, score))

    # token-overlap fallback: measure simple token intersection
    pred_tokens = set(cleaned.split())
    for label in raw_labels:
        label_tokens = set(label.split())
        if pred_tokens and label_tokens:
            overlap = pred_tokens.intersection(label_tokens)
            if overlap:
                ax = properties.get(label.lower())
                score = len(overlap) / max(len(pred_tokens), len(label_tokens))
                candidates.append((ax, 0.5 + 0.5 * score))

    # deduplicate preserving highest score
    seen = {}
    for ax, s in candidates:
        key_ax = ax.get("type", "").lower()
        seen[key_ax] = max(seen.get(key_ax, 0.0), float(s))

    result = []
    for k_ax, s in seen.items():
        ax = properties.get(k_ax) or properties.get(re.sub(r"[^a-z0-9]", "", k_ax))
        if ax:
            result.append((ax, s))

    # sort by score descending
    result.sort(key=lambda x: x[1], reverse=True)
    return result[:top_k]


def validate_domain_range(candidate_axiom: Dict, subject_text: str, object_text: str) -> float:
    """Return a domain/range compatibility score in [0,1]. Higher is better.

    This is a heuristic check: if the subject/object text contains class
    names from the base ontology that match the axiom domains/ranges, score
    is higher. If subject/object are question placeholders (e.g., 'who'), we
    return a moderate score rather than failing.
    """
    idx = load_base_indexes()
    classes = idx["classes"]

    domains = [d.lower() for d in candidate_axiom.get("domains", []) if isinstance(d, str)]
    ranges = [r.lower() for r in candidate_axiom.get("ranges", []) if isinstance(r, str)]

    subj = normalize_label(subject_text or "")
    obj = normalize_label(object_text or "")

    def match_side(text: str, side_list: List[str]) -> bool:
        if not text:
            return False
        for cname in side_list:
            # check containment of the class name in the text or vice versa
            if cname in text or text in cname:
                return True
            # also check if class name appears as a token in text
            if re.search(r"\b" + re.escape(cname) + r"\b", text):
                return True
        return False

    subj_score = 0.0
    obj_score = 0.0

    # question placeholders -> moderate score so it doesn't block
    if subj in QUESTION_WORDS:
        subj_score = 0.5
    elif match_side(subj, domains):
        subj_score = 1.0

    if obj in QUESTION_WORDS:
        obj_score = 0.5
    elif match_side(obj, ranges):
        obj_score = 1.0

    # If ontology domains/ranges are empty, treat as permissive
    if not domains:
        subj_score = max(subj_score, 0.6)
    if not ranges:
        obj_score = max(obj_score, 0.6)

    return (subj_score + obj_score) / 2.0


def ground_predicate(predicate_text: str, subject_text: str, object_text: str) -> Dict:
    """Ground a predicate text to the best-matching base-axiom.

    Returns a dict with keys: `axiom` (the chosen axiom dict or None),
    `predicate_label`, `predicate_uri` (axiom type string), `confidence`,
    and `scores` describing lexical and domain-range contributions.
    """
    candidates = generate_predicate_candidates(predicate_text)
    if not candidates:
        return {"axiom": None, "predicate_label": predicate_text, "predicate_uri": None, "confidence": 0.0, "scores": {}}

    scored = []
    for ax, lexical_score in candidates:
        dr_score = validate_domain_range(ax, subject_text, object_text)
        # final weighting: lexical 60%, domain/range 40%
        final = 0.6 * float(lexical_score) + 0.4 * float(dr_score)
        scored.append((ax, float(lexical_score), float(dr_score), final))

    scored.sort(key=lambda x: x[3], reverse=True)
    best_ax, lex, dr, final_score = scored[0]

    return {
        "axiom": best_ax,
        "predicate_label": best_ax.get("type"),
        "predicate_uri": best_ax.get("type"),
        "confidence": float(final_score),
        "scores": {"lexical": lex, "domain_range": dr},
    }


def atomic_to_graph(atomic_input: str, ttl_path: str = None, confidence_threshold: float = 0.65) -> Dict:
    """Top-level function: produce a candidate grounded triple for an atomic
    question. Returns a structured dict with grounding and fallback metadata.
    """
    LOG.debug("atomic input: %s", atomic_input)

    # Agent-first extraction: constrained LLM agent output. If the agent fails or is unavailable, we return empty extraction and metadata indicating fallback.
    subj = ""
    pred = ""
    obj = ""
    fallback_used = False
    fallback_agent_output = None

    agent_available = "triple_extraction_agent" in globals() and triple_extraction_agent is not None
    if agent_available:
        try:
            triple, meta = triple_extraction_agent.run(atomic_input)
            # agent should return a dict per schema, but handle common variants
            if isinstance(triple, dict):
                subj = triple.get("subject", "") or ""
                pred = triple.get("predicate", "") or ""
                obj = triple.get("object", "") or ""
            elif isinstance(triple, str):
                import json
                try:
                    parsed = json.loads(triple)
                    if isinstance(parsed, dict):
                        subj = parsed.get("subject", "") or ""
                        pred = parsed.get("predicate", "") or ""
                        obj = parsed.get("object", "") or ""
                except Exception:
                    # unexpected string output - fall through to rule fallback
                    raise
            else:
                raise ValueError("Agent produced unexpected output type")

            fallback_agent_output = triple#{"triple": triple, "meta": meta}
        except Exception as e:
            LOG.exception("Agent extraction failed; returning empty extraction: %s", e)
            subj = ""
            pred = ""
            obj = ""
            fallback_used = True
            fallback_agent_output = {"error": str(e)}
    else:
        LOG.warning("No agent available; returning empty extraction.")
        subj = ""
        pred = ""
        obj = ""
        fallback_used = True
        fallback_agent_output = {"error": "no_agent_available"}

    # Ground the predicate using the base ontology
    grounding = ground_predicate(pred, subj, obj)

    # 4) load TTL (optional) for downstream graph construction
    resolved_ttl_path = resolve_ttl_path(ttl_path)
    try:
        ttl = load_ttl(file_path=resolved_ttl_path)
    except Exception:
        ttl = None

    # Simple ABox fuzzy-linking: match subject/object to individual URIs by
    # local-name or rdfs:label if TTL is available. This is intentionally
    # lightweight; a more robust matcher can be plugged in later.
    abox_subject_candidates = []
    abox_object_candidates = []
    try:
        if ttl:
            g = ttl.get("graph") if isinstance(ttl, dict) else None
            individuals = ttl.get("individuals", set()) if isinstance(ttl, dict) else set()

            def find_individuals_by_label(text: str) -> List[str]:
                if not text:
                    return []
                tnorm = normalize_label(text)
                results: List[str] = []

                # Match by local-name (URI fragment or last path segment)
                for ind in individuals:
                    s = str(ind)
                    local = s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
                    if tnorm == normalize_label(local) or tnorm in normalize_label(local) or normalize_label(local) in tnorm:
                        results.append(s)

                # If graph available, try rdfs:label matching
                if g is not None:
                    from rdflib.namespace import RDFS
                    for ind in set(g.subjects()):
                        for lbl in g.objects(ind, RDFS.label):
                            try:
                                lstr = lbl.toPython() if hasattr(lbl, "toPython") else str(lbl)
                            except Exception:
                                lstr = str(lbl)
                            if normalize_label(lstr) == tnorm or tnorm in normalize_label(lstr) or normalize_label(lstr) in tnorm:
                                results.append(str(ind))

                # Deduplicate while preserving order
                seen = set()
                deduped: List[str] = []
                for r in results:
                    if r not in seen:
                        seen.add(r)
                        deduped.append(r)
                return deduped

            abox_subject_candidates = find_individuals_by_label(subj)
            abox_object_candidates = find_individuals_by_label(obj)
    except Exception as e:
        LOG.exception("ABox fuzzy matching failed: %s", e)

    out = {
        "subject_text": subj,
        "object_text": obj,
        "predicate_label": grounding.get("predicate_label"),
        "predicate_uri": grounding.get("predicate_uri"),
        "confidence": grounding.get("confidence", 0.0),
        "grounding_scores": grounding.get("scores", {}),
        "fallback_used": fallback_used,
        "fallback_agent_output": fallback_agent_output,
        "raw_extraction": {"subject_text": subj, "predicate_text": pred, "object_text": obj},
        "ttl_loaded": bool(ttl),
        "abox_subject_candidates": abox_subject_candidates,
        "abox_object_candidates": abox_object_candidates,
    }

    LOG.debug("Agent-first grounded output: %s", out)
    return out


def main():
    parser = argparse.ArgumentParser(description="Convert ATOMIC input into a graph structure.")
    parser.add_argument("--backend", default="phi-npu-openvino")
    args = parser.parse_args()

    backend = load_backend(name=args.backend)

    global triple_extraction_agent
    triple_extraction_agent = ExtractTripleAgent(backend=backend)

    input_str = [
        "wasn't a sensor small?",
        "who is Mats?",
        "what is Sens Motion?",
        "What activity types are supported by Sens Motion?",
    ]

    for atomic_input in input_str:
        result = atomic_to_graph(atomic_input)
        print(atomic_input, "->", result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()