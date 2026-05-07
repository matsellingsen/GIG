from pathlib import Path
from urllib.parse import unquote, quote
import json
from rdflib import Graph, RDF, RDFS, OWL, URIRef

TTL_PATH = Path(r"c:\Users\matse\gig\src\system_v5\KB\current\phi-npu-openvino_ontology_20260406_232546_classResolved_phase4_cached__fullyResolved_fullyResolved_and_logged.ttl")
OUT_PATH = Path(r"c:\Users\matse\gig\src\system_v5\tests\dataset\ontology_grounded_100.json")
EX = "http://example.org/sensInnovationAps_ontology#"


g = Graph()
g.parse(str(TTL_PATH), format="turtle")


def u(local: str) -> URIRef:
    return URIRef(EX + quote(local, safe=""))


def human(local: str) -> str:
    return unquote(local).replace("%20", " ")


def label(node) -> str:
    if isinstance(node, URIRef):
        labels = list(g.objects(node, RDFS.label))
        if labels:
            return str(labels[0])
        return human(str(node).split("#")[-1])
    return str(node)


def types_of(node):
    return [label(t) for t in g.objects(node, RDF.type) if t != OWL.NamedIndividual]


def direct_supers(node):
    return [label(s) for s in g.objects(node, RDFS.subClassOf)]


def equivalents(node):
    vals = []
    for pred in (OWL.sameAs, OWL.equivalentClass):
        for o in g.objects(node, pred):
            vals.append(label(o))
    return vals


def is_structural(pred):
    return pred in {RDF.type, RDFS.label, RDFS.comment, RDFS.subClassOf, RDFS.domain, RDFS.range, OWL.sameAs, OWL.equivalentClass}


def properties(node):
    vals = []
    for p, _ in g.predicate_objects(node):
        if is_structural(p):
            continue
        vals.append(label(p))
    return sorted(set(vals))


def prop_values(node):
    out = {}
    for p, o in g.predicate_objects(node):
        if is_structural(p):
            continue
        key = label(p)
        val = label(o) if isinstance(o, URIRef) else str(o)
        out.setdefault(key, [])
        if val not in out[key]:
            out[key].append(val)
    return out


def mapping_for(node):
    return {
        "types": types_of(node),
        "superclasses": direct_supers(node),
        "equivalent_classes": equivalents(node),
        "properties": properties(node),
        "property_values": prop_values(node),
    }


def empty_mapping():
    return {
        "types": [],
        "superclasses": [],
        "equivalent_classes": [],
        "properties": [],
        "property_values": {},
    }


def count_parts(node):
    return sum(1 for _ in g.objects(node, u("hasPart")))


def count_subclasses(node):
    return sum(1 for _ in g.subjects(RDFS.subClassOf, node))


def sameas_targets(node):
    vals = [label(o) for o in g.objects(node, OWL.sameAs)]
    vals += [label(o) for o in g.objects(node, OWL.equivalentClass)]
    return vals


PRESENT_SPECS = [
    # Definition
    ("definition", "entity", "App", "class", None, None, "What is App?"),
    ("definition", "entity", "WebApplication", "class", None, None, "What is WebApplication?"),
    ("definition", "entity", "SensorPatch", "class", None, None, "What is SensorPatch?"),
    ("definition", "entity", "SENS%20motion%20app", "instance", None, None, "What is SENS motion app?"),
    ("definition", "entity", "SENS%20motion%20web%20app", "instance", None, None, "What is SENS motion web app?"),
    ("definition", "entity", "version%203.9", "instance", None, None, "What is version 3.9?"),

    # Taxonomic
    ("taxonomic", "boolean", "App", "class", "InformationObject", "class", "Is App a subclass of InformationObject?"),
    ("taxonomic", "boolean", "WebApplication", "class", "SocialObject", "class", "Is WebApplication a subclass of SocialObject?"),
    ("taxonomic", "boolean", "ActivityType", "class", "ActivityCategory", "class", "Is ActivityType a subclass of ActivityCategory?"),
    ("taxonomic", "boolean", "SENS%20motion%20app", "instance", "SmartphoneApp", "class", "Is SENS motion app an instance of SmartphoneApp?"),
    ("taxonomic", "boolean", "support.sens.dk", "instance", "WebApplication", "class", "Is support.sens.dk an instance of WebApplication?"),
    ("taxonomic", "boolean", "SENS%20motion%20activity%20sensor", "instance", "ActivityDataSensor", "class", "Is SENS motion activity sensor an instance of ActivityDataSensor?"),

    # Capability
    ("capability", "boolean", "WebApplication", "class", "browser-accessible", "literal", "Can a WebApplication be accessed through a web browser?"),
    ("capability", "boolean", "SensorPatch", "class", "wearable", "literal", "Can a SensorPatch be worn on the body to monitor health metrics?"),
    ("capability", "boolean", "SENS%20motion%20app%20on%20Android", "instance", "nearby sensors", "literal", "Does SENS motion app on Android have nearby sensors?"),
    ("capability", "boolean", "version%203.9", "instance", "usability enhancements", "literal", "Does version 3.9 include usability enhancements?"),
    ("capability", "entity", "support.sens.dk", "instance", "ExportFunctionality", "class", "What feature does support.sens.dk have?"),
    ("capability", "entity", "SENS%20motion%20app%20on%20Android", "instance", "SensorFavoriteList", "class", "What kind of list does SENS motion app on Android expose?"),

    # Property
    ("property", "literal", "WebApplication", "class", "label", "literal", "What is the label of WebApplication?"),
    ("property", "literal", "SensorPatch", "class", "label", "literal", "What is the label of SensorPatch?"),
    ("property", "literal", "ActivityType", "class", "comment", "literal", "What is the comment of ActivityType?"),
    ("property", "literal", "SENS%20motion%20web%20app", "instance", "hasAppUrl", "literal", "What is the app URL of SENS motion web app?"),
    ("property", "literal", "SENS%20motion%20web%20app", "instance", "hasLoginUrl", "literal", "What is the login URL of SENS motion web app?"),
    ("property", "literal", "version%203.9", "instance", "releaseNumber", "literal", "What is the release number of version 3.9?"),

    # Membership
    ("membership", "list", "ActivityCategory", "class", "subclasses", "class-list", "What are the direct subclasses of ActivityCategory?"),
    ("membership", "list", "WebApplication", "class", "subclasses", "class-list", "What are the direct subclasses of WebApplication?"),
    ("membership", "list", "Publication", "class", "subclasses", "class-list", "What are the direct subclasses of Publication?"),
    ("membership", "list", "support.sens.dk", "instance", "parts", "list", "What parts does support.sens.dk have?"),
    ("membership", "list", "SENS%20motion%20activity%20sensor", "instance", "parts", "list", "What parts does SENS motion activity sensor have?"),
    ("membership", "list", "pop-up%20modal", "instance", "parts", "list", "What parts does pop-up modal have?"),

    # Comparative
    ("comparative", "entity", "support.sens.dk", "instance", "SENS%20motion%20app", "instance", "Which has more parts, support.sens.dk or SENS motion app?"),
    ("comparative", "entity", "SENS%20motion%20activity%20sensor", "instance", "version%203.9", "instance", "Which has more parts, SENS motion activity sensor or version 3.9?"),
    ("comparative", "entity", "pop-up%20modal", "instance", "guide", "instance", "Which has more parts, pop-up modal or guide?"),
    ("comparative", "entity", "SENS%20motion%20web%20app", "instance", "SENS%20motion%20app%20on%20Android", "instance", "Which has more parts, SENS motion web app or SENS motion app on Android?"),
    ("comparative", "literal", "support.sens.dk", "instance", "SENS%20motion%20app", "instance", "How many more parts does support.sens.dk have than SENS motion app?"),
    ("comparative", "literal", "SENS%20motion%20activity%20sensor", "instance", "version%203.9", "instance", "How many more parts does SENS motion activity sensor have than version 3.9?"),
    ("comparative", "literal", "support.sens.dk", "instance", "SENS%20motion%20app%20on%20Android", "instance", "How many more direct types does support.sens.dk have than SENS motion app on Android?"),
    ("comparative", "literal", "guide", "instance", "pop-up%20modal", "instance", "How many more parts does guide have than pop-up modal?"),

    # Quantification
    ("quantification", "entity", "support.sens.dk", "instance", "14 parts", "literal", "Which entity has exactly 14 parts?"),
    ("quantification", "entity", "SENS%20motion%20activity%20sensor", "instance", "10 parts", "literal", "Which entity has exactly 10 parts?"),
    ("quantification", "entity", "ActivityCategory", "class", "4 direct subclasses", "literal", "Which class has exactly 4 direct subclasses?"),
    ("quantification", "entity", "SENS%20motion%20app%20on%20Android", "instance", "1 sameAs link", "literal", "Which entity has exactly 1 sameAs link?"),
    ("quantification", "list", "Publication", "class", "direct subclasses", "class-list", "Which direct subclasses does Publication have?"),
    ("quantification", "list", "SENS%20motion%20app%20on%20Android", "instance", "sameAs targets", "list", "Which entities are directly sameAs-linked to SENS motion app on Android?"),

    # Existential
    ("existential", "entity", "SENS%20motion%20app%20on%20Android", "instance", "sameAs counterpart", "instance", "Which entity is the sameAs counterpart of SENS motion app on Android?"),
    ("existential", "entity", "SENS%20motion%20web%20app", "instance", "sameAs counterpart", "instance", "Which entity is the sameAs counterpart of SENS motion web app?"),
    ("existential", "entity", "support.sens.dk", "instance", "web app feature", "class", "Which entity has web app feature ExportFunctionality?"),
    ("existential", "entity", "version%203.9", "instance", "release number", "literal", "Which entity has release number 3.9?"),
    ("existential", "list", "SENS%20motion%20app%20on%20Android", "instance", "sameAs targets", "list", "Which entities have a sameAs link to SENS motion app on Android?"),
    ("existential", "list", "SENS%20motion%20activity%20sensor", "instance", "parts", "list", "Which entities are parts of SENS motion activity sensor?"),
]

assert len(PRESENT_SPECS) == 50


def build_present_item(idx, spec):
    qtype, form, entity_local, entity_kind, object_local, object_kind, question = spec
    entity_node = u(entity_local)
    entity_label = label(entity_node)
    obj_node = u(object_local) if object_local and object_kind not in {"literal", "class-list", "list"} else None

    if qtype == "definition":
        if entity_kind == "class":
            gold = direct_supers(entity_node)[0] if direct_supers(entity_node) else entity_label
        else:
            gold = types_of(entity_node)[0] if types_of(entity_node) else entity_label
    elif qtype == "taxonomic":
        gold = "yes"
    elif qtype == "capability":
        if form == "boolean":
            gold = "yes" if entity_local in {"WebApplication", "SensorPatch", "SENS%20motion%20app%20on%20Android", "version%203.9"} else "no"
        else:
            gold = object_local if object_local else entity_label
    elif qtype == "property":
        if entity_local == "WebApplication":
            gold = "WebApplication"
        elif entity_local == "SensorPatch":
            gold = "SensorPatch"
        elif entity_local == "ActivityType":
            gold = "ActivityType"
        elif entity_local == "SENS%20motion%20web%20app" and object_local == "hasAppUrl":
            gold = "https://app.sens.dk"
        elif entity_local == "SENS%20motion%20web%20app" and object_local == "hasLoginUrl":
            gold = "https://app.sens.dk/r/login"
        elif entity_local == "version%203.9":
            gold = "3.9"
        else:
            gold = entity_label
    elif qtype == "membership":
        if object_kind == "class-list":
            gold = [label(x) for x in g.subjects(RDFS.subClassOf, entity_node)]
        else:
            gold = [label(x) for x in g.objects(entity_node, u("hasPart"))]
    elif qtype == "comparative":
        left = count_parts(entity_node)
        right = count_parts(u(object_local))
        if form == "entity":
            gold = entity_label if left >= right else label(u(object_local))
        else:
            gold = str(abs(left - right))
    elif qtype == "quantification":
        if form == "entity":
            if entity_local == "support.sens.dk":
                gold = entity_label
            elif entity_local == "SENS%20motion%20activity%20sensor":
                gold = entity_label
            elif entity_local == "ActivityCategory":
                gold = entity_label
            elif entity_local == "SENS%20motion%20app%20on%20Android":
                gold = entity_label
            else:
                gold = entity_label
        else:
            if entity_local == "Publication":
                gold = [label(x) for x in g.subjects(RDFS.subClassOf, entity_node)]
            else:
                gold = sameas_targets(entity_node)
    elif qtype == "existential":
        if form == "entity":
            if entity_local == "SENS%20motion%20app%20on%20Android":
                gold = "SENS motion app on iOS"
            elif entity_local == "SENS%20motion%20web%20app":
                gold = "SENS web app"
            elif entity_local == "support.sens.dk":
                gold = "ExportFunctionality"
            elif entity_local == "version%203.9":
                gold = entity_label
            else:
                gold = entity_label
        else:
            if entity_local == "SENS%20motion%20app%20on%20Android":
                gold = sameas_targets(entity_node)
            else:
                gold = [label(x) for x in g.objects(entity_node, u("hasPart"))]
    else:
        gold = entity_label

    if isinstance(gold, list):
        acceptable = sorted({x.lower() for x in gold})
    elif form == "boolean":
        acceptable = ["yes", "true"] if str(gold).lower() == "yes" else ["no", "false"]
    else:
        acceptable = [str(gold).lower(), str(gold)]

    if qtype in {"definition", "taxonomic"}:
        relation = "be"
    elif qtype == "capability":
        relation = "capability"
    elif qtype == "property":
        relation = object_local
    elif qtype == "membership":
        relation = "has member"
    elif qtype == "comparative":
        relation = "compare"
    elif qtype == "quantification":
        relation = "count"
    elif qtype == "existential":
        relation = "exists"
    else:
        relation = qtype

    item = {
        "id": f"present_{idx:02d}",
        "atomic_input": question,
        "question_type": qtype,
        "answer_form": form,
        "entity": {"value": entity_label, "type": entity_kind},
        "relation": relation,
        "object": {"value": None if object_local is None else (label(u(object_local)) if object_kind not in {"literal", "class-list", "list"} else (object_local if object_kind == "literal" else None)), "type": object_kind},
        "ontology_presence": "present",
        "gold_answer_agent": {
            "reasoning": "",
            "answer": gold,
        },
        "gold_mapped_answer": {
            "entity_side": mapping_for(entity_node),
            "object_side": None if qtype != "comparative" else mapping_for(u(object_local)),
        },
        "expected_decision": "answer",
        "acceptable_labels": acceptable,
        "stage_labels": {
            "input_to_graph": {
                "question_type": qtype,
                "answer_form": form,
                "entity": entity_label,
                "relation": relation,
                "object": None if object_local is None else label(u(object_local)),
            },
            "resolve_entity": entity_label,
            "fetch_relevant_info": "present",
            "generate_answer": gold,
            "map_answer_to_context": "grounded",
        },
    }
    return item


present_items = [build_present_item(i + 1, spec) for i, spec in enumerate(PRESENT_SPECS)]

# Mirror the present items into absent items by replacing the entity/object with invented labels.
absent_items = []
for i, base in enumerate(present_items, start=1):
    fake_entity = f"Imaginary {base['entity']['value']}"
    fake_object = None
    if base['object']['value'] is not None:
        fake_object = f"Imaginary {base['object']['value']}"
    question = base['atomic_input']
    question = question.replace(base['entity']['value'], fake_entity)
    if base['object']['value']:
        question = question.replace(str(base['object']['value']), fake_object)
    absent_items.append({
        "id": f"absent_{i:02d}",
        "atomic_input": question,
        "question_type": base["question_type"],
        "answer_form": base["answer_form"],
        "entity": {"value": fake_entity, "type": base["entity"]["type"]},
        "relation": base["relation"],
        "object": {"value": fake_object, "type": base["object"]["type"]},
        "ontology_presence": "absent",
        "gold_answer_agent": {
            "reasoning": "",
            "answer": "I can't answer the question",
        },
        "gold_mapped_answer": {
            "entity_side": empty_mapping(),
            "object_side": empty_mapping(),
        },
        "expected_decision": "abstain",
        "acceptable_labels": ["i don't know", "unknown", "not in ontology", "not found", "can't answer the question", "I can't answer the question"],
        "stage_labels": {
            "input_to_graph": {
                "question_type": base["question_type"],
                "answer_form": base["answer_form"],
                "entity": fake_entity,
                "relation": base["relation"],
                "object": fake_object,
            },
            "resolve_entity": None,
            "fetch_relevant_info": "absent",
            "generate_answer": "I can't answer the question",
            "map_answer_to_context": "empty",
        },
    })

final_dataset = present_items + absent_items
assert len(final_dataset) == 100, len(final_dataset)
assert sum(1 for x in final_dataset if x["ontology_presence"] == "present") == 50
assert sum(1 for x in final_dataset if x["ontology_presence"] == "absent") == 50

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(final_dataset, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT_PATH}")
print(f"items={len(final_dataset)} present=50 absent=50")
