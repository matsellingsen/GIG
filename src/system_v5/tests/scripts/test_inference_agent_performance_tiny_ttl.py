import json
import os
import pytest
import re

from system_v5.backends import load_backend
from system_v5.tools.ttl_handling.load_ttl import load_ttl
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
    os.path.join(os.path.dirname(__file__), "..", "dataset", "agent_performance_tiny_ttl.json")
)
TINY_TTL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "fixtures", "tiny_ontology.ttl")
)
REPORT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "reports", "agent_performance_tiny_ttl_results.json")
)

with open(DATASET_PATH) as f:
    CASES = json.load(f)

RESULTS = []


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def matches_any_label(answer, labels):
    answer_norm = normalize(answer)
    for label in labels:
        if normalize(label) in answer_norm:
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
def ttl_fixture():
    return load_ttl(file_path=TINY_TTL_PATH)


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
@pytest.mark.parametrize("case", CASES)
def test_agent_performance_tiny_ttl(case, ttl_fixture, agents):
    checks = []
    case_meta = {
        "id": case["id"],
        "atomic_input": case["atomic_input"],
        "expected_decision": case["expected_decision"],
        "acceptable_labels": case["acceptable_labels"],
    }
    expected_decision = case["expected_decision"]

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
        resolved_ok = expected_decision != "answer"
        record_check(
            checks,
            "resolved_entity",
            resolved_ok,
            expected="found" if expected_decision == "answer" else "not required",
            actual="not found",
        )
        answer_text = ""
        relevant_info = {}
    else:
        record_check(checks, "resolved_entity", True, expected="found", actual="found")
        relevant_info = fetched.get("relevant_info", {})
        answer = generate_answer(
            question_info=question_info,
            relevant_info=relevant_info,
            generate_answer_agent=agents["generate_answer"],
        )
        answer_text = answer.get("answer", "") if isinstance(answer, dict) else ""

    mapped = map_answer_to_context(answer=answer_text, context=relevant_info)

    acceptable_labels = case["acceptable_labels"]
    matched_label = matches_any_label(answer_text, acceptable_labels)

    if expected_decision == "answer":
        record_check(checks, "answer_non_empty", bool(answer_text), expected="non-empty", actual=answer_text)

    if expected_decision == "answer":
        record_check(
            checks,
            "answer_matches_labels",
            matched_label,
            expected=acceptable_labels,
            actual=answer_text,
        )
    else:
        abstained = matched_label or fetched == "noPrimaryEntityFound"
        record_check(
            checks,
            "answer_abstains",
            abstained,
            expected=acceptable_labels,
            actual="noPrimaryEntityFound" if fetched == "noPrimaryEntityFound" else answer_text,
        )

    RESULTS.append({"case": case_meta, "checks": checks, "mapped": mapped})
    failures = [c for c in checks if not c["passed"]]
    assert not failures, f"{len(failures)} checks failed: {', '.join(c['name'] for c in failures)}"
