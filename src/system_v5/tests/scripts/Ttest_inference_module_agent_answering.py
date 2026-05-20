import json
import os
import re
import pytest
import sys
sys.path.append("C:\\Users\\matse\\gig\\src\\system_v5") # Adjust this path to point to the root of your project if needed


from agent_loop.agents.inference_module.resolve_entity_agent import ResolveEntityAgent
from agent_loop.agents.inference_module.generate_answer_agent import GenerateAnswerAgent
from backends import load_backend
from tools.inference_module.fetch_relevant_info import flatten_entity_context


REPORT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "reports", "agent_component_answering_results.json")
)

RESULTS = []


def normalize_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def matches_any_label(answer, labels):
    answer_norm = normalize_text(answer)
    for label in labels:
        if normalize_text(label) in answer_norm:
            return True
    return False


def record_check(checks, name, passed, expected=None, actual=None):
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "expected": expected,
            "actual": actual,
        }
    )


@pytest.fixture(scope="session")
def backend():
    return load_backend()


@pytest.fixture(scope="session")
def agents(backend):
    return {
        "resolve_entity": ResolveEntityAgent(backend),
        "generate_answer": GenerateAnswerAgent(backend),
        "validate_answer": ValidateAnswerAgent(backend),
    }


@pytest.fixture(scope="session", autouse=True)
def write_report():
    yield
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2)


RESOLVE_CASES = [
    {
        "id": "resolve_sens_motion_v2",
        "question_info": {
            "question_type": "definition",
            "relation": "be",
            "entity": {"value": "SENS motion v2", "type": "individual"},
            "object": {"value": "unknown", "type": "unknown"},
        },
        "candidates": [
            {
                "label": "SENS motion v2",
                "score": 0.99,
                "uri": "http://example.org/sensInnovationAps_ontology#SENS_motion_v2",
            },
            {
                "label": "SENS motion",
                "score": 0.85,
                "uri": "http://example.org/sensInnovationAps_ontology#SENS_motion",
            },
        ],
        "expected_label": "SENS motion v2",
    }
]


@pytest.mark.llm
@pytest.mark.parametrize("case", RESOLVE_CASES)
def test_resolve_entity_agent(case, agents):
    checks = []

    result, _ = agents["resolve_entity"].run(
        question_info=case["question_info"],
        candidates=case["candidates"],
    )

    actual = result.get("selected_label") if isinstance(result, dict) else None

    record_check(
        checks,
        "selected_label_match",
        actual == case["expected_label"],
        expected=case["expected_label"],
        actual=actual,
    )

    RESULTS.append({"case": {"id": case["id"]}, "checks": checks, "selected_label": actual})
    failures = [c for c in checks if not c["passed"]]
    assert not failures, f"{len(failures)} checks failed: {', '.join(c['name'] for c in failures)}"


ANSWER_CASES = [
    {
        "id": "answer_serial_number",
        "question_info": {
            "atomic_question": "What is the serial number of SENS motion?",
            "question_type": "property",
            "answer_form": "value",
            "entity": {"value": "SENS motion", "type": "individual"},
            "relation": "have property",
            "object": {"value": "serial number", "type": "literal"},
        },
        "entity_context": {
            "uri": "http://example.org/sensInnovationAps_ontology#SENS_motion",
            "label": "SENS motion",
            "types": ["Sensor", "Product"],
            "superclasses": {"Sensor": ["Device"]},
            "equivalent_classes": [],
            "properties_by_type": {
                "Sensor": {
                    "outgoing_object_properties": [
                        {"property": "hasPart", "object": "SensorCore"}
                    ],
                    "outgoing_data_properties": [
                        {"property": "serialNumber", "value": "SM-001", "datatype": None}
                    ],
                    "incoming_object_properties": [],
                    "incoming_data_properties": [],
                }
            },
            "annotations": {},
            "class_descriptions": {},
            "object_property_descriptions": {},
            "provenance": [],
        },
        "acceptable_labels": ["SM-001"],
    },
    {
        "id": "answer_parts",
        "question_info": {
            "atomic_question": "What parts does SENS motion have?",
            "question_type": "membership",
            "answer_form": "list",
            "entity": {"value": "SENS motion", "type": "individual"},
            "relation": "has member",
            "object": {"value": "parts", "type": "class"},
        },
        "entity_context": {
            "uri": "http://example.org/sensInnovationAps_ontology#SENS_motion",
            "label": "SENS motion",
            "types": ["Sensor", "Product"],
            "superclasses": {"Sensor": ["Device"]},
            "equivalent_classes": [],
            "properties_by_type": {
                "Sensor": {
                    "outgoing_object_properties": [
                        {"property": "hasPart", "object": "SensorCore"}
                    ],
                    "outgoing_data_properties": [],
                    "incoming_object_properties": [],
                    "incoming_data_properties": [],
                }
            },
            "annotations": {},
            "class_descriptions": {},
            "object_property_descriptions": {},
            "provenance": [],
        },
        "acceptable_labels": ["SensorCore"],
    },
]


@pytest.mark.llm
@pytest.mark.parametrize("case", ANSWER_CASES)
def test_generate_answer_agent(case, agents):
    checks = []

    flat_context = flatten_entity_context(case["entity_context"])

    result, _ = agents["generate_answer"].run(
        question_info=case["question_info"],
        entity_context=case["entity_context"],
        object_context=None,
    )

    answer_text = result.get("answer") if isinstance(result, dict) else ""
    matched = matches_any_label(answer_text, case["acceptable_labels"])

    record_check(
        checks,
        "answer_matches_labels",
        matched,
        expected=case["acceptable_labels"],
        actual=answer_text,
    )

    RESULTS.append(
        {
            "case": {"id": case["id"]},
            "checks": checks,
            "atomic question": case["question_info"].get("atomic_question"),
            "triplet": {
                "entity": case["question_info"].get("entity", {}).get("value"),
                "relation": case["question_info"].get("relation"),
                "object": case["question_info"].get("object", {}).get("value"),
            },
            "answer": answer_text,
            "reasoning": result.get("reasoning") if isinstance(result, dict) else None,
            "entity_context": case["entity_context"],
        }
    )
    failures = [c for c in checks if not c["passed"]]
    assert not failures, f"{len(failures)} checks failed: {', '.join(c['name'] for c in failures)}"


VALIDATE_CASES = [
    {
        "id": "validate_correct_serial",
        "atomic_question": "What is the serial number of SENS motion?",
        "question_type": "property",
        "extracted_triplet": {
            "entity": "SENS motion",
            "relation": "serialNumber",
            "object": "SM-001",
        },
        "generated_answer": "SM-001",
        "mapped_answer_triplets": [
            {"entity": "SENS motion", "relation": "serialNumber", "object": "SM-001"}
        ],
        "expected_decision": True,
    },
    {
        "id": "validate_incorrect_serial",
        "atomic_question": "What is the serial number of SENS motion?",
        "question_type": "property",
        "extracted_triplet": {
            "entity": "SENS motion",
            "relation": "serialNumber",
            "object": "SM-001",
        },
        "generated_answer": "SM-999",
        "mapped_answer_triplets": [],
        "expected_decision": False,
    },
]


@pytest.mark.llm
@pytest.mark.parametrize("case", VALIDATE_CASES)
def test_validate_answer_agent(case, agents):
    checks = []

    result, _ = agents["validate_answer"].run(
        atomic_question=case["atomic_question"],
        question_type=case["question_type"],
        extracted_triplet=case["extracted_triplet"],
        generated_answer=case["generated_answer"],
        mapped_answer_triplets=case["mapped_answer_triplets"],
    )

    decision = result.get("decision") if isinstance(result, dict) else None

    record_check(
        checks,
        "decision_match",
        decision == case["expected_decision"],
        expected=case["expected_decision"],
        actual=decision,
    )

    RESULTS.append(
        {
            "case": {"id": case["id"]},
            "checks": checks,
            "decision": decision,
            "reasoning": result.get("reasoning") if isinstance(result, dict) else None,
        }
    )
    failures = [c for c in checks if not c["passed"]]
    assert not failures, f"{len(failures)} checks failed: {', '.join(c['name'] for c in failures)}"
