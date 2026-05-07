import json
import os
import re

import pytest

from system_v5.backends import load_backend
from system_v5.tools.ttl_handling.load_ttl import load_ttl
from system_v5.tools.ttl_handling.resolve_ttl_path import resolve_ttl_path
from system_v5.tools.inference_module.input_to_graph import atomic_to_graph
from system_v5.tools.inference_module.fetch_relevant_info import fetch_relevant_info
from system_v5.tools.inference_module.generate_answer import generate_answer
from system_v5.tools.inference_module.map_answer_to_context import map_answer_to_context

from agent_loop.agents.inference_module.extract_question_type_agent import ExtractQuestionTypeAgent
from agent_loop.agents.inference_module.resolve_answer_form_agent import ResolveAnswerFormAgent
from agent_loop.agents.inference_module.extract_entity_agent import ExtractEntityAgent
from agent_loop.agents.inference_module.extract_relation_agent import ExtractRelationAgent
from agent_loop.agents.inference_module.extract_object_agent import ExtractObjectAgent
from agent_loop.agents.inference_module.resolve_entity_agent import ResolveEntityAgent
from agent_loop.agents.inference_module.generate_answer_agent import GenerateAnswerAgent


DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset", "ontology_grounded_100_natural.json")
)
REPORT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "reports", "ontology_grounded_100_results.json")
)

with open(DATASET_PATH, encoding="utf-8") as f:
    CASES = json.load(f)


RESULTS = []


def normalize(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def matches_any_label(answer, labels):
    answer_norm = normalize(answer)
    return any(normalize(label) in answer_norm for label in labels)


def record_check(checks, name, passed, expected=None, actual=None):
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "expected": expected,
            "actual": actual,
        }
    )


def mapping_covers_expected(actual_mapping, expected_mapping):
    expected_mapping = expected_mapping or {}
    expected_entity_side = expected_mapping.get("entity_side", {})

    expected_superclasses = set(expected_entity_side.get("superclasses", []))
    expected_types = set(expected_entity_side.get("types", []))
    expected_equivalent_classes = set(expected_entity_side.get("equivalent_classes", []))
    expected_properties = set(expected_entity_side.get("properties", []))

    actual_superclasses = set(actual_mapping.get("superclasses", []))
    actual_types = set(actual_mapping.get("types", []))
    actual_equivalent_classes = set(actual_mapping.get("equivalent_classes", []))
    actual_properties = set(actual_mapping.get("properties", []))

    return (
        expected_superclasses.issubset(actual_superclasses)
        and expected_types.issubset(actual_types)
        and expected_equivalent_classes.issubset(actual_equivalent_classes)
        and expected_properties.issubset(actual_properties)
    )


@pytest.fixture(scope="session")
def ttl_fixture():
    return load_ttl(file_path=resolve_ttl_path())


@pytest.fixture(scope="session")
def backend():
    return load_backend()


@pytest.fixture(scope="session")
def agents(backend):
    return {
        "extract_question_type": ExtractQuestionTypeAgent(backend),
        "resolve_answer_form": ResolveAnswerFormAgent(backend),
        "extract_entity": ExtractEntityAgent(backend),
        "extract_relation": ExtractRelationAgent(backend),
        "extract_object": ExtractObjectAgent(backend),
        "resolve_entity": ResolveEntityAgent(backend),
        "generate_answer": GenerateAnswerAgent(backend),
    }


@pytest.fixture(scope="session", autouse=True)
def write_report():
    yield
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2)


@pytest.mark.llm
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_ontology_grounded_100(case, ttl_fixture, agents):
    checks = []

    question_info = atomic_to_graph(
        case["atomic_input"],
        extract_question_type_agent=agents["extract_question_type"],
        extract_answer_form_agent=agents["resolve_answer_form"],
        extract_entity_agent=agents["extract_entity"],
        extract_relation_agent=agents["extract_relation"],
        extract_object_agent=agents["extract_object"],
    )
    question_info["atomic_question"] = case["atomic_input"]

    fetched = fetch_relevant_info(
        question_info=question_info,
        ttl=ttl_fixture,
        resolve_entity_agent=agents["resolve_entity"],
    )

    if fetched == "noPrimaryEntityFound":
        record_check(
            checks,
            "entity_resolution",
            case["expected_decision"] == "abstain",
            expected=case["expected_decision"],
            actual="noPrimaryEntityFound",
        )
        RESULTS.append(
            {
                "case": {"id": case["id"]},
                "checks": checks,
                "question_info": question_info,
                "fetched": fetched,
                "answer": None,
                "reasoning": None,
                "mapped_entity_answer": None,
                "mapped_reasoning_answer": None,
            }
        )
        failures = [c for c in checks if not c["passed"]]
        assert not failures, f"{len(failures)} checks failed: {', '.join(c['name'] for c in failures)}"
        return

    entity_context = fetched.get("entity_context", {})
    object_context = fetched.get("object_context", {})

    answer = generate_answer(
        question_info=question_info,
        entity_context=entity_context,
        object_context=object_context,
        generate_answer_agent=agents["generate_answer"],
    )

    answer_text = answer.get("answer", "") if isinstance(answer, dict) else ""
    reasoning_text = answer.get("reasoning", "") if isinstance(answer, dict) else ""

    mapped_entity_answer = map_answer_to_context(answer=answer_text, context=entity_context)
    mapped_reasoning_answer = map_answer_to_context(answer=reasoning_text, context=entity_context)

    expected_decision = case["expected_decision"]
    acceptable_labels = case.get("acceptable_labels", [])
    gold_answer_agent = case.get("gold_answer_agent") or {}
    gold_answer = gold_answer_agent.get("answer")
    gold_reasoning = gold_answer_agent.get("reasoning")
    gold_mapped_answer = case.get("gold_mapped_answer")

    if expected_decision == "answer":
        record_check(
            checks,
            "answer_non_empty",
            bool(answer_text),
            expected="non-empty",
            actual=answer_text,
        )
        record_check(
            checks,
            "answer_matches_labels",
            matches_any_label(answer_text, acceptable_labels),
            expected=acceptable_labels,
            actual=answer_text,
        )
        record_check(
            checks,
            "gold_answer_matches_labels",
            matches_any_label(gold_answer, acceptable_labels),
            expected=acceptable_labels,
            actual=gold_answer,
        )
        record_check(
            checks,
            "mapping_covers_expected_context",
            mapping_covers_expected(mapped_entity_answer, gold_mapped_answer),
            expected=gold_mapped_answer,
            actual=mapped_entity_answer,
        )
        if gold_reasoning:
            record_check(
                checks,
                "reasoning_non_empty",
                bool(reasoning_text),
                expected="non-empty",
                actual=reasoning_text,
            )
    else:
        record_check(
            checks,
            "abstain_expected",
            expected_decision == "abstain",
            expected="abstain",
            actual=expected_decision,
        )

    RESULTS.append(
        {
            "case": {"id": case["id"]},
            "checks": checks,
            "question_info": question_info,
            "fetched": fetched,
            "entity_context": entity_context,
            "object_context": object_context,
            "answer": answer_text,
            "reasoning": reasoning_text,
            "gold_answer_agent": gold_answer_agent,
            "mapped_entity_answer": mapped_entity_answer,
            "mapped_reasoning_answer": mapped_reasoning_answer,
        }
    )

    failures = [c for c in checks if not c["passed"]]
    assert not failures, f"{len(failures)} checks failed: {', '.join(c['name'] for c in failures)}"