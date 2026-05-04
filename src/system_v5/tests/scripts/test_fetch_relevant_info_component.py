import os
import pytest

from system_v5.tools.ttl_handling.load_ttl import load_ttl
from system_v5.tools.inference_module.fetch_relevant_info import fetch_relevant_info


TINY_TTL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "fixtures", "tiny_ontology.ttl")
)


class StubResolveEntityAgent:
    def __init__(self, selected_label=None):
        self.selected_label = selected_label

    def run(self, question_info=None, candidates=None):
        if self.selected_label:
            return {"selected_label": self.selected_label}, None
        if candidates:
            return {"selected_label": candidates[0]["label"]}, None
        return {"selected_label": "unknown"}, None


@pytest.fixture(scope="session")
def tiny_ttl():
    return load_ttl(file_path=TINY_TTL_PATH)


def collect_property_names(relevant_info):
    properties = set()
    for _, data in relevant_info.get("properties_by_type", {}).items():
        for entry in data.get("outgoing_object_properties", []):
            properties.add(entry.get("property"))
        for entry in data.get("outgoing_data_properties", []):
            properties.add(entry.get("property"))
    return properties


def test_fetch_relevant_info_single_candidate(tiny_ttl):
    question_info = {
        "question_type": "definition",
        "entity": {"value": "SensorCore", "type": "individual"},
        "relation": "be",
        "object": {"value": "unknown", "type": "unknown"},
    }

    result = fetch_relevant_info(
        question_info=question_info,
        ttl=tiny_ttl,
        resolve_entity_agent=StubResolveEntityAgent(),
    )

    assert result != "noPrimaryEntityFound"
    assert "relevant_info" in result
    types = result["relevant_info"].get("types", [])
    assert "Device" in types


def test_fetch_relevant_info_properties(tiny_ttl):
    question_info = {
        "question_type": "property",
        "entity": {"value": "SENS motion", "type": "individual"},
        "relation": "have property",
        "object": {"value": "serial number", "type": "literal"},
    }

    result = fetch_relevant_info(
        question_info=question_info,
        ttl=tiny_ttl,
        resolve_entity_agent=StubResolveEntityAgent(selected_label="SENS motion"),
    )

    assert result != "noPrimaryEntityFound"
    relevant_info = result["relevant_info"]
    properties = collect_property_names(relevant_info)
    assert "hasPart" in properties
    assert "hasManufacturer" in properties
    assert "serialNumber" in properties
    assert "weightKg" in properties


def test_fetch_relevant_info_sameas_merge(tiny_ttl):
    question_info = {
        "question_type": "definition",
        "entity": {"value": "SENS motion v2", "type": "individual"},
        "relation": "be",
        "object": {"value": "unknown", "type": "unknown"},
    }

    result = fetch_relevant_info(
        question_info=question_info,
        ttl=tiny_ttl,
        resolve_entity_agent=StubResolveEntityAgent(selected_label="SENS motion v2"),
    )

    assert result != "noPrimaryEntityFound"
    provenance = result["relevant_info"].get("provenance", [])
    assert "1" in provenance
    assert "2" in provenance
