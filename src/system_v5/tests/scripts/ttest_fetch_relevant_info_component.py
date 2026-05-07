import json
import os
import pytest

from system_v5.tools.ttl_handling.load_ttl import load_ttl
from system_v5.tools.inference_module.fetch_relevant_info import fetch_relevant_info


TINY_TTL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "fixtures", "tiny_ontology.ttl")
)
REPORT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "reports", "fetch_relevant_info_component_results.json")
)

RESULTS = []


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


@pytest.fixture(scope="session", autouse=True)
def write_report():
    yield
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2)


def record_check(checks, name, passed, expected=None, actual=None):
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "expected": expected,
            "actual": actual,
        }
    )


def collect_property_names(relevant_info):
    properties = set()
    for _, data in relevant_info.get("properties_by_type", {}).items():
        for entry in data.get("outgoing_object_properties", []):
            properties.add(entry.get("property"))
        for entry in data.get("outgoing_data_properties", []):
            properties.add(entry.get("property"))
    return properties


def test_fetch_relevant_info_single_candidate(tiny_ttl):
    checks = []
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

    record_check(
        checks,
        "resolved_entity",
        result != "noPrimaryEntityFound",
        expected="found",
        actual="noPrimaryEntityFound" if result == "noPrimaryEntityFound" else "found",
    )
    record_check(checks, "has_relevant_info", "relevant_info" in result, expected=True, actual="relevant_info" in result)
    types = result.get("relevant_info", {}).get("types", []) if result != "noPrimaryEntityFound" else []
    record_check(checks, "type_contains_device", "Device" in types, expected=True, actual=types)

    RESULTS.append({"case": {"test": "single_candidate", "question_info": question_info}, "checks": checks})
    failures = [c for c in checks if not c["passed"]]
    assert not failures, f"{len(failures)} checks failed: {', '.join(c['name'] for c in failures)}"


def test_fetch_relevant_info_properties(tiny_ttl):
    checks = []
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

    record_check(
        checks,
        "resolved_entity",
        result != "noPrimaryEntityFound",
        expected="found",
        actual="noPrimaryEntityFound" if result == "noPrimaryEntityFound" else "found",
    )
    relevant_info = result.get("relevant_info", {}) if result != "noPrimaryEntityFound" else {}
    properties = collect_property_names(relevant_info)
    record_check(checks, "hasPart_present", "hasPart" in properties, expected=True, actual=sorted(properties))
    record_check(checks, "hasManufacturer_present", "hasManufacturer" in properties, expected=True, actual=sorted(properties))
    record_check(checks, "serialNumber_present", "serialNumber" in properties, expected=True, actual=sorted(properties))
    record_check(checks, "weightKg_present", "weightKg" in properties, expected=True, actual=sorted(properties))

    RESULTS.append({"case": {"test": "properties", "question_info": question_info}, "checks": checks})
    failures = [c for c in checks if not c["passed"]]
    assert not failures, f"{len(failures)} checks failed: {', '.join(c['name'] for c in failures)}"


def test_fetch_relevant_info_sameas_merge(tiny_ttl):
    checks = []
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

    record_check(
        checks,
        "resolved_entity",
        result != "noPrimaryEntityFound",
        expected="found",
        actual="noPrimaryEntityFound" if result == "noPrimaryEntityFound" else "found",
    )
    provenance = result.get("relevant_info", {}).get("provenance", []) if result != "noPrimaryEntityFound" else []
    record_check(checks, "provenance_has_1", "1" in provenance, expected=True, actual=provenance)
    record_check(checks, "provenance_has_2", "2" in provenance, expected=True, actual=provenance)

    RESULTS.append({"case": {"test": "sameas_merge", "question_info": question_info}, "checks": checks})
    failures = [c for c in checks if not c["passed"]]
    assert not failures, f"{len(failures)} checks failed: {', '.join(c['name'] for c in failures)}"
