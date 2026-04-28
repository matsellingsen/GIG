import sys


sys.path.append("C:\\Users\\matse\\gig\\src\\system_v5") # Adjust this path to point to the root of your project if needed

from urllib.parse import unquote
from rdflib import RDF, RDFS, OWL, URIRef
import difflib
import re

from backends import load_backend
from tools.ttl_handling.load_ttl import load_ttl
from tools.ttl_handling.resolve_ttl_path import resolve_ttl_path
from agent_loop.agents.inference_module.resolve_entity_agent import ResolveEntityAgent
import urllib.parse


def canonical_relation_to_axiom_type(rel: str) -> str:
    rel = rel.lower().strip()

    # Taxonomic
    if rel in ["be subtype of", "be superclass of"]:
        return "ClassAxiom"

    # Object properties (TBox or ABox)
    if rel in [
        "has member", "is member of",
        "has part", "is part of",
        "include", "contain",
        "participate in", "has participant",
        "located in", "has location",
        "has component", "is component of",
        "precede", "follow",
        "has role", "express",
        "has precondition", "has postcondition",
        "execute task", "act through",
        "satisfy"
    ]:
        return "ObjectPropertyAssertion"

    # Data properties
    if rel in ["have property", "measure", "have value"]:
        return "DataPropertyAssertion"

    # Definition questions
    if rel == "be":
        return "AnnotationProperty"

    return "Unknown"

def canonical_relation_to_base_axiom(rel: str) -> str:
    """
    Map a canonical natural-language relation to a base ontology axiom.
    Returns the ontology axiom/property name as a string.
    """

    rel = rel.lower().strip()

    mapping = {
        # Taxonomic
        "be subtype of": "subClassOf",
        "be superclass of": "subClassOf",

        # Mereology / membership
        "has member": "hasMember",
        "is member of": "isMemberOf",
        "include": "hasMember",
        "contain": "hasPart",
        "has part": "hasPart",
        "is part of": "isPartOf",
        "consist of": "hasComponent",

        # Qualities / data values
        "have property": "hasQuality",
        "has quality": "hasQuality",
        "measure": "hasDataValue",
        "have value": "hasDataValue",

        # Participation / events
        "participate in": "participatesIn",
        "has participant": "hasParticipant",

        # Location / time
        "located in": "hasLocation",
        "has location": "hasLocation",
        "happen at": "hasTime",
        "occur at": "hasTime",

        # Aboutness
        "be about": "isAbout",

        # Components
        "has component": "hasComponent",
        "is component of": "isComponentOf",

        # Ordering
        "precede": "precedes",
        "follow": "follows",

        # Roles
        "has role": "hasRole",

        # Expression
        "express": "expresses",

        # Preconditions / postconditions
        "has precondition": "hasPrecondition",
        "has postcondition": "hasPostcondition",

        # Tasks / actions
        "execute task": "executesTask",
        "act through": "actsThrough",

        # Satisfaction
        "satisfy": "satisfies",

        # Definition questions
        "be": "isAbout"
    }

    # Exact match
    if rel in mapping:
        return mapping[rel]

    # Partial match fallback
    for key in mapping:
        if key in rel:
            return mapping[key]

    return "unknown"

from rdflib import RDF, RDFS, OWL

def resolve_primary_entity(question_info: dict, graph):
    """
    Resolve an entity name to a URI in the TTL graph.
    Performs exact match on label and URI fragment, then fuzzy match.
    """
    entity = question_info.get("entity", {}).get("value")

    #1. Fetch candidates
    top_candidates = retrieve_top_candidates(graph, entity, top_n=3)
    print("Candidate entities retrieved from TTL:")
    for candidate in top_candidates:
        print("label: ", candidate["label"])
        print("entity type(s): ", candidate["types"])
        print("superclasses (if class): ", candidate["superclasses"])
        print("object properties: ", candidate["object_properties"])
        print("data properties: ", candidate["data_properties"])
        print("annotations: ", candidate["annotations"])
        print("score: ", candidate["score"])
        print("-----------------------------!!")
    print("---------------------------------")
    ''
    #2. resolve candidates with rules or ResolveEntityAgent
    if not top_candidates: # no candidates found above threshold
        print("No candidates found in TTL for entity:", entity)
        return None
    if len(top_candidates) == 1 or top_candidates[0]["score"] == 1.0: # exact match found (score 1.0) or only 1 candidate. Agent is redundant.
        print("High confidence candidate found, selecting without agent:", top_candidates[0]["label"])
        return top_candidates[0]
    
    # Multiple candidates are found, use agent to resolve
    selected_candidate, _ = resolve_entity_agent.run(question_info, candidates=top_candidates, )
    print("Selected candidate after resolution:", selected_candidate)
    selected_candidate_label = selected_candidate.get("selected_label") if isinstance(selected_candidate, dict) else selected_candidate
    final_candidate = next((c for c in top_candidates if c["label"] == selected_candidate_label), None)

    return final_candidate

def normalize(text: str) -> str:
    text = text.lower().strip()
    if text.endswith("s"):
        text = text[:-1]
    return text

def tokenize(text: str):
    return re.findall(r"[a-zA-Z]+", text.lower())

def retrieve_top_candidates(graph, entity: str, top_n: int = 10):
    """
    Retrieve top ontology candidates (classes + individuals) for an entity string.
    Deduplicates by URI, aggregates labels, and enriches each candidate with:
    - entity_type
    - types (for individuals)
    - superclasses (for classes)
    - object_properties
    - data_properties
    - annotations
    """

    entity_norm = normalize(entity)
    entity_tokens = set(tokenize(entity_norm))

    uri_to_best = {}  # uri -> best label + score + all labels

    # Helper: compute similarity score
    def score_candidate(label, frag):
        label_norm = normalize(label)
        frag_norm = normalize(frag)

        sim_label = difflib.SequenceMatcher(None, entity_norm, label_norm).ratio()
        sim_frag = difflib.SequenceMatcher(None, entity_norm, frag_norm).ratio()

        label_tokens = {t for t in tokenize(label_norm) if len(t) >= 4}
        overlap = len(entity_tokens & label_tokens) / max(len(entity_tokens), 1)

        substring = 1.0 if any(t in label_tokens for t in entity_tokens) else 0.0

        return (0.45 * sim_label) + (0.35 * sim_frag) + (0.15 * overlap) + (0.05 * substring)

    # Iterate over all classes + individuals
    for s, _, o in graph.triples((None, RDF.type, None)):
        if o not in (OWL.Class, OWL.NamedIndividual):
            continue

        # Collect labels
        labels = [str(lbl) for _, _, lbl in graph.triples((s, RDFS.label, None))]
        if not labels:
            frag = unquote(s.split("#")[-1])
            labels = [frag]

        frag = unquote(s.split("#")[-1])

        # Score each label
        for lbl in labels:
            lbl_norm = normalize(lbl)

            if lbl_norm == entity_norm or normalize(frag) == entity_norm:
                score = 1.0
            else:
                score = score_candidate(lbl, frag)
                if score < 0.45:
                    continue

            if s not in uri_to_best or score > uri_to_best[s]["score"]:
                uri_to_best[s] = {
                    "label": lbl,
                    "score": score,
                    "labels": labels
                }

    # Now enrich each URI with ontology context
    enriched_candidates = []

    for uri, data in uri_to_best.items():
        # Determine entity type
        entity_type = None
        if (uri, RDF.type, OWL.Class) in graph:
            entity_type = "Class"
        elif (uri, RDF.type, OWL.NamedIndividual) in graph:
            entity_type = "NamedIndividual"

        # Types (for individuals)
        types = [
            unquote(str(o).split("#")[-1])
            for _, _, o in graph.triples((uri, RDF.type, None))
            if o not in (OWL.Class, OWL.NamedIndividual)
        ]

        # Superclasses (for classes)
        superclasses = [
            unquote(str(o).split("#")[-1])
            for _, _, o in graph.triples((uri, RDFS.subClassOf, None))
        ]

        # Object properties
        object_properties = {}
        for p, _, o in graph.triples((None, None, None)):
            pass  # placeholder

        object_properties = {}
        for p, _, o in graph.triples((uri, None, None)):
            if (p, RDF.type, OWL.ObjectProperty) in graph:
                pname = unquote(str(p).split("#")[-1])
                oval = unquote(str(o).split("#")[-1])
                object_properties.setdefault(pname, []).append(oval)

        # Data properties
        data_properties = {}
        for p, _, o in graph.triples((uri, None, None)):
            if (p, RDF.type, OWL.DatatypeProperty) in graph:
                pname = unquote(str(p).split("#")[-1])
                data_properties.setdefault(pname, []).append(str(o))

        # Annotations
        annotations = {}
        for p, _, o in graph.triples((uri, None, None)):
            if p in (RDFS.label, RDFS.comment):
                pname = unquote(str(p).split("#")[-1])
                annotations.setdefault(pname, []).append(str(o))

        enriched_candidates.append({
            "uri": uri,
            "label": data["label"],
            "score": data["score"],
            "labels": data["labels"],
            "entity_type": entity_type,
            "types": types,
            "superclasses": superclasses,
            "object_properties": object_properties,
            "data_properties": data_properties,
            "annotations": annotations
        })

    enriched_candidates.sort(key=lambda x: x["score"], reverse=True)

    # drop candidates with a score less than 0.5
    filtered_candidates = [cand for cand in enriched_candidates if cand["score"] >= 0.5]
    return filtered_candidates[:top_n]


def retrieve_axioms_for_entity(entity: dict, graph):
    """
    Retrieve all axioms involving the resolved entity.
    Returns a dict with keys:
        - class_axioms
        - object_property_assertions
        - data_property_assertions
        - annotations
    """
    entity_uri = entity["uri"]
    results = {
        "class_axioms": [],
        "object_property_assertions": [],
        "data_property_assertions": [],
        "annotations": []
    }

    # --- Class axioms (TBox) ---
    # entity is a class
    if (entity_uri, RDF.type, OWL.Class) in graph:
        # SubClassOf axioms
        for _, _, superclass in graph.triples((entity_uri, RDFS.subClassOf, None)):
            results["class_axioms"].append(("subClassOf", entity_uri, superclass))

        # EquivalentClass axioms
        for _, _, eq in graph.triples((entity_uri, OWL.equivalentClass, None)):
            results["class_axioms"].append(("equivalentClass", entity_uri, eq))

    # --- Class assertions (ABox) ---
    # entity is an individual
    if (entity_uri, RDF.type, OWL.NamedIndividual) in graph:
        for _, _, cls in graph.triples((entity_uri, RDF.type, None)):
            if cls != OWL.NamedIndividual:
                results["class_axioms"].append(("type", entity_uri, cls))

    # --- Object property assertions ---
    for _, prop, obj in graph.triples((entity_uri, None, None)):
        if (prop, RDF.type, OWL.ObjectProperty) in graph:
            results["object_property_assertions"].append((prop, entity_uri, obj))

    # --- Data property assertions ---
    for _, prop, lit in graph.triples((entity_uri, None, None)):
        if (prop, RDF.type, OWL.DatatypeProperty) in graph:
            results["data_property_assertions"].append((prop, entity_uri, lit))

    # --- Annotations (labels, comments) ---
    for _, p, o in graph.triples((entity_uri, None, None)):
        if p in [RDFS.label, RDFS.comment]:
            results["annotations"].append((p, entity_uri, o))

    return results
from rdflib import RDF, RDFS, OWL, Literal
from rdflib.namespace import XSD

def retrieve_full_entity_context(entity: dict, graph):
    """
    Retrieve ALL relevant OWL information about an entity, including SAME-AS MERGING:
    - direct types
    - superclasses (transitive)
    - equivalent classes
    - outgoing object properties
    - outgoing data properties
    - incoming object properties
    - incoming data properties
    - annotations on the entity
    - annotations on its classes
    - property semantics (domain, range, labels, comments)
    - provenance (chunk_id)
    - SAME-AS: merge context across all owl:sameAs-linked individuals
    """

    entity_uri = entity["uri"]

    # ---------------------------------------------------------
    # 0. Collect all URIs equivalent via owl:sameAs
    # ---------------------------------------------------------
    equivalent_uris = set([URIRef(entity_uri)])
    queue = [URIRef(entity_uri)]

    while queue:
        current = queue.pop()

        # outgoing sameAs
        for eq in graph.objects(current, OWL.sameAs):
            if eq not in equivalent_uris:
                equivalent_uris.add(eq)
                queue.append(eq)

        # incoming sameAs
        for eq in graph.subjects(OWL.sameAs, current):
            if eq not in equivalent_uris:
                equivalent_uris.add(eq)
                queue.append(eq)
    print(f"Equivalent URIs for entity {entity_uri}: {[str(uri) for uri in equivalent_uris]}")

    # Helper to get readable labels
    def get_label(x):
        for lbl in graph.objects(x, RDFS.label):
            return str(lbl)
        return str(x).split("#")[-1]

    # ---------------------------------------------------------
    # 1. Direct types (most specific)
    # ---------------------------------------------------------
    all_types = list(graph.objects(entity_uri, RDF.type))

    def is_subclass_of(c, d):
        if c == d:
            return False  # strict subclass only
        to_visit = [c]
        visited = set()
        while to_visit:
            cur = to_visit.pop()
            for sup in graph.objects(cur, RDFS.subClassOf):
                if sup == d:
                    return True
                if sup not in visited:
                    visited.add(sup)
                    to_visit.append(sup)
        return False

    # ✅ Correct: keep the *most specific* classes
    direct_types = [c for c in all_types if not any(is_subclass_of(d, c) for d in all_types if d != c)]

    # ---------------------------------------------------------
    # 2. Superclasses seperated by direct types (transitive)
    # ---------------------------------------------------------
    superclasses_by_direct_type = {}

    def collect_supers(cls, acc):
        for sup in graph.objects(cls, RDFS.subClassOf):
            if sup not in acc:
                acc.add(sup)
                collect_supers(sup, acc)

    for dt in direct_types:
        acc = set()
        collect_supers(dt, acc)
        superclasses_by_direct_type[str(dt)] = [str(s) for s in acc]

    # ---------------------------------------------------------
    # 3. Equivalent classes
    # ---------------------------------------------------------
    equivalent_classes = []
    for uri in equivalent_uris:
        for eq in graph.objects(uri, OWL.equivalentClass):
            equivalent_classes.append(str(eq))

    # ---------------------------------------------------------
    # PROPERTY GROUPING BY DIRECT TYPE
    # ---------------------------------------------------------

    # Helper: check if a property belongs to a class via rdfs:domain
    def property_applies_to_type(prop, cls):
        for dom in graph.objects(prop, RDFS.domain):
            if dom == cls or is_subclass_of(cls, dom):
                return True
        return False

    # Initialize structure
    properties_by_type = {
        str(dt): {
            "outgoing_object_properties": [],
            "outgoing_data_properties": [],
            "incoming_object_properties": [],
            "incoming_data_properties": []
        }
        for dt in direct_types
    }

    # Iterate over equivalent URIs
    for uri in equivalent_uris:

        # Outgoing object properties
        for p, o in graph.predicate_objects(uri):
            if isinstance(o, URIRef):
                for dt in direct_types:
                    if property_applies_to_type(p, dt):
                        properties_by_type[str(dt)]["outgoing_object_properties"].append({
                            "property": str(p),
                            "object": str(o)
                        })

        # Outgoing data properties
        for p, o in graph.predicate_objects(uri):
            if not isinstance(o, URIRef):
                for dt in direct_types:
                    if property_applies_to_type(p, dt):
                        properties_by_type[str(dt)]["outgoing_data_properties"].append({
                            "property": str(p),
                            "value": str(o),
                            "datatype": str(o.datatype) if hasattr(o, "datatype") else None
                        })

        # Incoming object properties
        for s, p in graph.subject_predicates(uri):
            if isinstance(s, URIRef):
                for dt in direct_types:
                    if property_applies_to_type(p, dt):
                        properties_by_type[str(dt)]["incoming_object_properties"].append({
                            "subject": str(s),
                            "property": str(p)
                        })

        # Incoming data properties
        for s, p in graph.subject_predicates(uri):
            if not isinstance(s, URIRef):
                for dt in direct_types:
                    if property_applies_to_type(p, dt):
                        properties_by_type[str(dt)]["incoming_data_properties"].append({
                            "subject": str(s),
                            "property": str(p)
                        })
    # ---------------------------------------------------------
    # 8. Entity annotations
    # ---------------------------------------------------------
    annotations = {}
    for uri in equivalent_uris:
        for p, o in graph.predicate_objects(uri):
            if (p, RDF.type, OWL.AnnotationProperty) in graph or p in (RDFS.label, RDFS.comment):
                annotations[str(p)] = str(o)

    # ---------------------------------------------------------
    # 9. Class descriptions
    # ---------------------------------------------------------
    class_descriptions = {}
    for uri in equivalent_uris:
        for t in graph.objects(uri, RDF.type):
            class_descriptions[str(t)] = {
                "description": next((str(c) for c in graph.objects(t, RDFS.comment)), None)
            }

    # ---------------------------------------------------------
    # 10. Object Property Descriptions
    # ---------------------------------------------------------
    object_property_descriptions = {}
    for uri in equivalent_uris:
        for p in graph.predicates(uri, None):
            if (p, RDF.type, OWL.ObjectProperty) in graph:
                object_property_descriptions[str(p)] = {
                    "description": next((str(c) for c in graph.objects(p, RDFS.comment)), None),
                    "domain": [str(d) for d in graph.objects(p, RDFS.domain)],
                    "range": [str(r) for r in graph.objects(p, RDFS.range)]
                }

    # ---------------------------------------------------------
    # 11. Provenance (chunk_id)
    # ---------------------------------------------------------
    provenance = []
    CHUNK = URIRef("http://example.org/sensInnovationAps_ontology#chunk_id")

    for uri in equivalent_uris:
        for cid in graph.objects(uri, CHUNK):
            provenance.append(str(cid))

    # ---------------------------------------------------------
    # Return merged context
    # ---------------------------------------------------------
    return {
        "uri": str(entity_uri),
        "label": get_label(URIRef(entity_uri)),
        "types": direct_types,
        "superclasses": superclasses_by_direct_type,
        "equivalent_classes": equivalent_classes,
        "properties_by_type": properties_by_type,
        "annotations": annotations,
        "class_descriptions": class_descriptions,
        "object_property_descriptions": object_property_descriptions,
        "provenance": provenance
    }


def filter_axioms(retrieved, base_axiom, owl_axiom_type):
    """
    Filter retrieved axioms based on the base axiom and OWL axiom type.
    """

    # --- Class axioms ---
    if owl_axiom_type == "ClassAxiom":
        return [
            ax for ax in retrieved["class_axioms"]
            if ax[0].lower() == base_axiom.lower()
        ]

    # --- Object property assertions ---
    if owl_axiom_type == "ObjectPropertyAssertion":
        return [
            ax for ax in retrieved["object_property_assertions"]
            if ax[0].split("#")[-1].lower() == base_axiom.lower()
        ]

    # --- Data property assertions ---
    if owl_axiom_type == "DataPropertyAssertion":
        return [
            ax for ax in retrieved["data_property_assertions"]
            if ax[0].split("#")[-1].lower() == base_axiom.lower()
        ]

    # --- Annotation properties (definitions) ---
    if owl_axiom_type == "AnnotationProperty":
        return retrieved["annotations"]

    return []

def visualize_retrieved_axioms(entity: dict, retrieved):
    """
    Display retrieved TTL axioms as a simple node-edge diagram.
    """
    entity_uri = entity["uri"]
    entity_label = unquote(entity["uri"].split("#")[-1])

    print("\n================ TTL ENTITY GRAPH ================")
    print(f"Entity: {entity_label}")
    print("--------------------------------------------------")

    # --- Class axioms ---
    if retrieved["class_axioms"]:
        print("\n[CLASS AXIOMS]")
        for axiom_type, subj, obj in retrieved["class_axioms"]:
            subj_label = unquote(subj.split("#")[-1])
            obj_label = unquote(obj.split("#")[-1])
            print(f"  {subj_label} -- {axiom_type} --> {obj_label}")

    # --- Object properties ---
    if retrieved["object_property_assertions"]:
        print("\n[OBJECT PROPERTIES]")
        for prop, subj, obj in retrieved["object_property_assertions"]:
            prop_label = unquote(prop.split("#")[-1])
            subj_label = unquote(subj.split("#")[-1])
            obj_label = unquote(obj.split("#")[-1])
            print(f"  {subj_label} -- {prop_label} --> {obj_label}")

    # --- Data properties ---
    if retrieved["data_property_assertions"]:
        print("\n[DATA PROPERTIES]")
        for prop, subj, lit in retrieved["data_property_assertions"]:
            prop_label = unquote(prop.split("#")[-1])
            subj_label = unquote(subj.split("#")[-1])
            print(f"  {subj_label} -- {prop_label} --> \"{lit}\"")

    # --- Annotations ---
    if retrieved["annotations"]:
        print("\n[ANNOTATIONS]")
        for p, subj, o in retrieved["annotations"]:
            prop_label = unquote(p.split("#")[-1])
            subj_label = unquote(subj.split("#")[-1])
            print(f"  {subj_label} -- {prop_label} --> \"{o}\"")

    print("==================================================\n")

def filter_context(question_info, full_context):
    """Filter context based on question type."""
    qtype = question_info["question_type"]

    if qtype == "definition":
        return {
            "definition": {
                "types": full_context["types"],
                "superclasses": full_context["superclasses"],
                "annotations": full_context["annotations"],
                "class_descriptions": full_context["class_descriptions"]
            }
        }

    if qtype == "taxonomic":
        return {
            "taxonomic": {
                "types": full_context["types"],
                "superclasses": full_context["superclasses"],
                "equivalent_classes": full_context["equivalent_classes"]
            }
        }

    if qtype == "capability":
        return {
            "capability": {
                "actions": full_context["outgoing_object_properties"],
                "participations": full_context["incoming_object_properties"],
                "object_property_semantics": full_context["object_property_semantics"]
            }
        }

    if qtype == "property":
        return {
            "property": {
                "data_properties": full_context["outgoing_data_properties"],
                "object_properties": full_context["outgoing_object_properties"],
                "object_property_semantics": full_context["object_property_semantics"]
            }
        }

    if qtype == "membership":
        return {
            "membership": {
                "outgoing": full_context["outgoing_object_properties"],
                "incoming": full_context["incoming_object_properties"]
            }
        }

    if qtype == "comparative":
        return {
            "comparative": {
                "entity_A_properties": full_context["outgoing_data_properties"],
                "object_property_semantics": full_context["object_property_semantics"]
            }
        }

    if qtype == "quantification":
        return {
            "quantification": {
                "countable_relations": full_context["outgoing_object_properties"],
                "numeric_properties": full_context["outgoing_data_properties"]
            }
        }

    if qtype == "existential":
        return {
            "existential": {
                "matching_relations": full_context["outgoing_object_properties"],
                "matching_properties": full_context["outgoing_data_properties"]
            }
        }

    return {"unknown": {"message": "Unsupported or malformed question."}}

def normalize_and_clean_context_for_llm(relevant_info):
    """
    Normalize interpreted context for LLM consumption:
    - Remove all URIs (convert to readable labels)
    - URL-decode labels
    - Recursively normalize nested lists/dicts
    - Keep only property label + comment (drop domain/range)
    """
    #def clean_structure(val: dict):

    def strip_uri(value):
        """Convert URI → label, decode URL-encoded fragments."""
        if not isinstance(value, str):
            return value
        if "#" in value:
            value = value.split("#")[-1]
        return urllib.parse.unquote(value)

    def normalize_value(val):
        """Recursively normalize any value."""
        # String → strip URI
        if isinstance(val, str):
            return strip_uri(val)

        # List → normalize each element
        if isinstance(val, list):
            return [normalize_value(x) for x in val]

        # Dict → normalize keys and values
        if isinstance(val, dict):
            normalized = {}
            for k, v in val.items():
                # Special case: object property semantics → keep only label + description
                if k == "object_property_descriptions":
                    normalized[k] = normalize_object_property_semantics(v)
                else:
                    normalized[strip_uri(k)] = normalize_value(v)
            return normalized

        # Other types → return as-is
        return val

    def normalize_object_property_semantics(semantics_dict):
        """
        Keep only:
        - description
        Remove:
        - domain
        - range
        """
        cleaned = {}
        for prop_uri, info in semantics_dict.items():
            prop_name = strip_uri(prop_uri)
            cleaned[prop_name] = {
                "description": info.get("description")
            }
        return cleaned

    # ---------------------------------------------------------
    # Normalize each question_type block
    # ---------------------------------------------------------
    normalized = normalize_value(relevant_info)

    normalized_and_cleaned = normalized #clean_structure(normalized)

    return normalized_and_cleaned

def fetch_relevant_info(question_info: dict, ttl: dict):
    """
    Fetch relevant information from the TTL based on the extracted question information.
    
    Args:
        question_info (dict): A dictionary containing the extracted question information, including:
            - question_type: The type of the question (e.g., "definition", "taxonomic", etc.)
            - entity: A dictionary with 'value' and 'type' keys for the primary entity.
            - relation: The canonical relation extracted from the input.
            - object: A dictionary with 'value' and 'type' keys for the object.
    """
    # 0. Setup variables
    atomic_question = question_info.get("atomic_question")
    question_type = question_info.get("question_type")
    entity = question_info.get("entity", {}).get("value")
    entity_type = question_info.get("entity", {}).get("type")
    relation = question_info.get("relation")
    obj = question_info.get("object", {}).get("value")
    object_type = question_info.get("object", {}).get("type")

    # 1. Determine the axiom type and base axiom based on the canonical relation
    axiom_type = canonical_relation_to_axiom_type(relation)
    base_axiom = canonical_relation_to_base_axiom(relation)
    print("===============================")
    print("input information:", question_info)
    print("---------------------------------")
    print("axiom type:", axiom_type)
    print("base axiom:", base_axiom)

    #2. Resolve the primary entity and object to URIs in the TTL graph
    resolved_entity = resolve_primary_entity(question_info, ttl["graph"])
      
    
    print("resolved entity:", resolved_entity)
    if resolved_entity is None:
        print("Could not resolve primary entity to a URI in the TTL.")
        return None
    
    # 3. Retrieve full context for the resolved entity
    full_entity_context = retrieve_full_entity_context(resolved_entity, ttl["graph"])
    #print("== FULL ENTITY CONTEXT ==")
    #for key, value in full_entity_context.items():
    #    print(f"{key}: {value}")
    #print("==========================")
    

    # 4. Filter the full context based on the question type to get the most relevant information for answering the question.
    #relevant_info = filter_context(question_info, full_entity_context)
    relevant_info = full_entity_context # for now, skip filtering to show all retrieved info. We can re-introduce this later once we have the full context retrieval working well.
    
    # 5. Normalize the relevant information to be LLM-friendly (no URIs, only human-readable labels and comments).
    relevant_info_normalized = normalize_and_clean_context_for_llm(relevant_info)

    # 5. group together all information and return it.
    final_output = {
        "question_info": question_info,
        "resolved_entity": resolved_entity,
        "relevant_info": relevant_info_normalized
    }
    return final_output


def main():
    dummy_questions = {
        "q0": {
            "atomic_question": "What is Sens Motion?",
            "question_type": "definition",
            "entity": {"value": "SENS Motion", "type": "individual"},
            "relation": "be",
            "object": {"value": "unknown", "type": "unknown"}
        },
        "q1": {
            "atomic_question": "What parts does Sens Motion have?",
            "question_type": "membership",
            "entity": {"value": "Sens Motion", "type": "individual"},
            "relation": "has member",
            "object": {"value": "parts", "type": "class"}
        },

        "q2": {
            "atomic_question": "Who is Mats Ellingsen?",
            "question_type": "definition",
            "entity": {"value": "Mats Ellingsen", "type": "individual"},
            "relation": "be",
            "object": {"value": "unknown", "type": "unknown"}
        },

        "q3": {
            "atomic_question": "What is the DOI of Sens Motion?",
            "question_type": "property",
            "entity": {"value": "Sens Motion", "type": "individual"},
            "relation": "have property",
            "object": {"value": "DOI", "type": "literal"}
        },

        "q4": {
            "atomic_question": "What color does Sens Motion have?",
            "question_type": "property",
            "entity": {"value": "Sens Motion", "type": "individual"},
            "relation": "have property",
            "object": {"value": "color", "type": "literal"}
        },

        "q5": {
            "atomic_question": "Is employee a subtype of person?",
            "question_type": "taxonomic",
            "entity": {"value": "employee", "type": "class"},
            "relation": "be subtype of",
            "object": {"value": "person", "type": "class"}
        }
    }
    
    # start backend and init agents/classes needed for entity resolution and TTL retrieval
    global backend
    backend = load_backend(name="phi-npu-openvino")
    global resolve_entity_agent
    resolve_entity_agent = ResolveEntityAgent(backend=backend)

    # load ttl 
    resolved_ttl_path = resolve_ttl_path(ttl_path=None)
    ttl = load_ttl(file_path=resolved_ttl_path) 

    outputs = {}
    for qid, qinfo in dummy_questions.items():
        print(f"\n\n========== Processing question {qid} ==========")
        outputs[qid] = fetch_relevant_info(question_info=qinfo, ttl=ttl)

    return outputs

if __name__ == "__main__":
    main()