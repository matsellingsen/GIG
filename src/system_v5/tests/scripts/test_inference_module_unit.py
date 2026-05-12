import os
import pytest

from system_v5.tools.ttl_handling.load_ttl import load_ttl
from system_v5.tools.inference_module.fetch_relevant_info import retrieve_full_entity_context
from system_v5.tools.inference_module.map_answer_to_context import map_answer_to_context


TINY_TTL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "fixtures", "tiny_ontology.ttl")
)


@pytest.fixture(scope="session")
def tiny_ttl():
    return load_ttl(file_path=TINY_TTL_PATH)


def has_object_property(entries, prop_suffix, obj_suffix):
    for item in entries:
        prop = str(item.get("property", ""))
        obj = str(item.get("object", ""))
        if prop.endswith(prop_suffix) and obj.endswith(obj_suffix):
            return True
    return False


def has_data_property(entries, prop_suffix, value):
    for item in entries:
        prop = str(item.get("property", ""))
        val = str(item.get("value", ""))
        if prop.endswith(prop_suffix) and val == value:
            return True
    return False


def test_retrieve_full_entity_context_merges_sameas(tiny_ttl):
    from rdflib import URIRef

    entity = {"uri": URIRef("http://example.org/sensInnovationAps_ontology#SENS_motion_v2")}
    context = retrieve_full_entity_context(type="subject", entity=entity, graph=tiny_ttl["graph"])

    assert "1" in context.get("provenance", [])
    assert "2" in context.get("provenance", [])
    assert any(str(t).endswith("Sensor") for t in context.get("types", []))

    sensor_key = next((k for k in context.get("properties_by_type", {}) if k.endswith("Sensor")), None)
    assert sensor_key is not None

    superclasses = context.get("superclasses", {}).get(sensor_key, [])
    assert any(str(s).endswith("Device") for s in superclasses)

    outgoing_objects = context["properties_by_type"][sensor_key]["outgoing_object_properties"]
    outgoing_data = context["properties_by_type"][sensor_key]["outgoing_data_properties"]

    assert has_object_property(outgoing_objects, "hasPart", "SensorCore")
    assert has_data_property(outgoing_data, "serialNumber", "SM-001")
    assert has_data_property(outgoing_data, "weightKg", "0.15")


def test_map_answer_to_context_matches_values():
    context = {
        "types": ["Sensor", "Device"],
        "superclasses": {"Sensor": ["Device"]},
        "equivalent_classes": ["Gadget"],
        "class_descriptions": {"Sensor": {"description": "A device that senses."}},
        "properties_by_type": {
            "Sensor": {
                "outgoing_object_properties": [
                    {"property": "hasPart", "object": "SensorCore"}
                ],
                "outgoing_data_properties": [
                    {"property": "serialNumber", "value": "SM-001"}
                ],
                "incoming_object_properties": [],
                "incoming_data_properties": [],
            }
        },
    }

    answer = "The sensor has serial number SM-001 and includes SensorCore."
    mapped = map_answer_to_context(answer, context)

    assert "Sensor" in mapped.get("types", [])
    assert "serialNumber" in mapped.get("properties", [])
    assert "hasPart" in mapped.get("properties", [])
    assert mapped.get("property_values", {}).get("serialNumber") == "sm-001"
    assert mapped.get("property_values", {}).get("hasPart") == "sensorcore"


def test_map_answer_to_context_uses_class_description():
    context = {
        "types": ["Sensor"],
        "superclasses": {"Sensor": ["Device"]},
        "equivalent_classes": [],
        "class_descriptions": {"Sensor": {"description": "A device that senses."}},
        "properties_by_type": {},
    }

    answer = "It is a device that senses."
    mapped = map_answer_to_context(answer, context)

    assert "Sensor" in mapped.get("types", [])


def test_map_answer_to_context_disambiguates_ispartof_collection():
    context = {
        "types": [],
        "superclasses": {},
        "equivalent_classes": [],
        "class_descriptions": {},
        "properties_by_type": {
            "Device": {
                "outgoing_object_properties": [
                    {"property": "isPartOf", "object": "Collection"}
                ],
                "outgoing_data_properties": [],
                "incoming_object_properties": [],
                "incoming_data_properties": [],
            }
        },
    }

    answer = "It is part of a collection."
    mapped = map_answer_to_context(answer, context)

    assert "isPartOf" not in mapped.get("properties", [])
