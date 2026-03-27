import json


def _normalize_base_iri(base_iri: str) -> str:
    if base_iri.endswith("#") or base_iri.endswith("/"):
        return base_iri
    return base_iri + "#"


def _entity_iri(base_iri: str, local_name: str) -> str:
    return f"<{_normalize_base_iri(base_iri)}{local_name}>"


def _datatype_iri(datatype: str) -> str:
    suffix = datatype.split(":", 1)[1]
    return f"<http://www.w3.org/2001/XMLSchema#{suffix}>"


def _escape_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_ontology_axiom(base_iri: str, axiom: dict) -> str:
    axiom_type = axiom["type"]
    if axiom_type == "DeclareClass":
        return f"Declaration(Class({_entity_iri(base_iri, axiom['id'])}))"
    if axiom_type == "DeclareObjectProperty":
        return f"Declaration(ObjectProperty({_entity_iri(base_iri, axiom['id'])}))"
    if axiom_type == "DeclareDataProperty":
        return f"Declaration(DataProperty({_entity_iri(base_iri, axiom['id'])}))"
    if axiom_type == "SubClassOf":
        return (
            f"SubClassOf({_entity_iri(base_iri, axiom['subclass'])} "
            f"{_entity_iri(base_iri, axiom['superclass'])})"
        )
    if axiom_type == "ObjectPropertyDomain":
        return (
            f"ObjectPropertyDomain({_entity_iri(base_iri, axiom['property'])} "
            f"{_entity_iri(base_iri, axiom['class'])})"
        )
    if axiom_type == "ObjectPropertyRange":
        return (
            f"ObjectPropertyRange({_entity_iri(base_iri, axiom['property'])} "
            f"{_entity_iri(base_iri, axiom['class'])})"
        )
    raise ValueError(f"Unsupported ontology axiom type: {axiom_type}")


def render_instance_axiom(base_iri: str, axiom: dict) -> str:
    axiom_type = axiom["type"]
    if axiom_type == "DeclareIndividual":
        return f"Declaration(NamedIndividual({_entity_iri(base_iri, axiom['id'])}))"
    if axiom_type == "ClassAssertion":
        return (
            f"ClassAssertion({_entity_iri(base_iri, axiom['class'])} "
            f"{_entity_iri(base_iri, axiom['individual'])})"
        )
    if axiom_type == "ObjectPropertyAssertion":
        return (
            f"ObjectPropertyAssertion({_entity_iri(base_iri, axiom['property'])} "
            f"{_entity_iri(base_iri, axiom['subject'])} "
            f"{_entity_iri(base_iri, axiom['object'])})"
        )
    if axiom_type == "DataPropertyAssertion":
        literal = f'"{_escape_literal(axiom["value"])}"^^{_datatype_iri(axiom["datatype"])}'
        return (
            f"DataPropertyAssertion({_entity_iri(base_iri, axiom['property'])} "
            f"{_entity_iri(base_iri, axiom['subject'])} {literal})"
        )
    raise ValueError(f"Unsupported instance axiom type: {axiom_type}")


def render_owl2fs_v2_sections(payload: dict | str) -> tuple[str, str]:
    if isinstance(payload, str):
        payload = json.loads(payload)

    base_iri = payload["base_iri"]
    ontology_text = "\n".join(render_ontology_axiom(base_iri, axiom) for axiom in payload["ontology"])
    instances_text = "\n".join(render_instance_axiom(base_iri, axiom) for axiom in payload["instances"])
    return ontology_text, instances_text


def render_owl2fs_v2_document(payload: dict | str) -> str:
    if isinstance(payload, str):
        payload = json.loads(payload)

    base_iri = payload["base_iri"]
    ontology_lines = [render_ontology_axiom(base_iri, axiom) for axiom in payload["ontology"]]
    instance_lines = [render_instance_axiom(base_iri, axiom) for axiom in payload["instances"]]
    body = "\n".join(ontology_lines + instance_lines)
    return f"Ontology(<{base_iri}>\n{body}\n)"