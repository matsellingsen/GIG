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
from system_v5.tools.inference_module.map_answer_to_context import map_answer_to_context, merge_mappings

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


def build_stage_log(question_info, fetched=None, answer=None, reasoning=None):
    fetched = fetched or {}
    return {
        "input_to_graph": {
            "question_type": question_info.get("question_type"),
            "answer_form": question_info.get("answer_form"),
            "entity": question_info.get("entity", {}).get("value"),
            "relation": question_info.get("relation"),
            "object": question_info.get("object", {}).get("value"),
        },
        "fetch_relevant_info": {
            "resolved_entity": fetched.get("resolved_entity") if isinstance(fetched, dict) else None,
            "resolved_object": fetched.get("resolved_object") if isinstance(fetched, dict) else None,
            "status": "noPrimaryEntityFound"
            if fetched == "noPrimaryEntityFound"
            else "noComparativeObjectFound"
            if fetched == "noComparativeObjectFound"
            else "present",
        },
        "generate_answer": {
            "answer": answer.get("answer") if isinstance(answer, dict) else None,
            "reasoning": reasoning,
        },
        "map_answer_to_context": {
            "entity_answer": None,
            "reasoning_answer": None,
        },
    }


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

    generated_question_stage = {
        "question_type": question_info.get("question_type"),
        "answer_form": question_info.get("answer_form"),
        "entity": question_info.get("entity", {}).get("value"),
        "entity_type": question_info.get("entity", {}).get("type"),
        "relation": question_info.get("relation"),
        "object": question_info.get("object", {}).get("value"),
        "object_type": question_info.get("object", {}).get("type"),
    }

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
                "stage_log": build_stage_log(question_info, fetched=fetched),
                "generated_question_stage": generated_question_stage,
                "answer": None,
                "reasoning": None,
                "mapped_entity_answer": None,
                "mapped_reasoning_answer": None,
            }
        )
        failures = [c for c in checks if not c["passed"]]
        assert not failures, f"{len(failures)} checks failed: {', '.join(c['name'] for c in failures)}"
        return
    elif fetched == "noComparativeObjectFound":
        record_check(
            checks,
            "object_resolution",
            case["expected_decision"] == "abstain",
            expected=case["expected_decision"],
            actual="noComparativeObjectFound",
        )
        RESULTS.append(
            {
                "case": {"id": case["id"]},
                "checks": checks,
                "question_info": question_info,
                "fetched": fetched,
                "stage_log": build_stage_log(question_info, fetched=fetched),
                "generated_question_stage": generated_question_stage,
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
    mapped_entity_reasoning = map_answer_to_context(answer=reasoning_text, context=entity_context)
    mapped_entity_merged = merge_mappings(mapped_entity_reasoning, mapped_entity_answer)

    if object_context: # to be used in future cases with comparative objects
        mapped_object_answer = map_answer_to_context(answer=answer_text, context=object_context)
        mapped_object_reasoning = map_answer_to_context(answer=reasoning_text, context=object_context)
        mapped_object_merged = merge_mappings(mapped_object_reasoning, mapped_object_answer)

    
    stage_log = build_stage_log(
        question_info,
        fetched=fetched,
        answer=answer,
        reasoning=reasoning_text,
    )

    stage_log["map_answer_to_context"]["merged_entity_answer"] = mapped_entity_merged

    expected_decision = case["expected_decision"]
    #acceptable_labels = case.get("acceptable_labels", [])
    gold_answer_agent = case.get("gold_answer_agent") or {}
    #gold_answer = gold_answer_agent.get("answer")
    gold_reasoning = gold_answer_agent.get("reasoning")
    gold_mapped_answer = case.get("gold_mapped_answer")

    # Preferred grounding: gold mapping is subset of actual mapping
    preferred_ok = mapping_covers_expected(mapped_entity_merged, gold_mapped_answer)

    # Acceptable grounding: any ontology grounding at all
    grounded_nonempty = any(
        bool(mapped_entity_merged.get(key))
        for key in ["annotations", "superclasses", "types", "properties", "property_values"]
    )
    # Incorrect grounding: no grounding
    empty_grounding = not grounded_nonempty


    if expected_decision == "answer":
        record_check(
            checks=checks,
            name="answer_non_empty",
            passed=bool(answer_text),
            expected="non-empty",
            actual=answer_text,
        )
        record_check(
        checks,
        name="grounding_preferred",
        passed=preferred_ok,
        expected=gold_mapped_answer,
        actual=mapped_entity_merged,
        )
        record_check(
            checks,
            name="grounding_nonempty",
            passed=grounded_nonempty,
            expected="any non-empty grounding",
            actual=mapped_entity_merged,
        )
        record_check(
            checks,
            name="grounding_incorrect",
            passed=not empty_grounding,
            expected="non-empty grounding",
            actual=mapped_entity_merged,
        )

        if gold_reasoning:
            record_check(
                checks=checks,
                name="reasoning_non_empty",
                passed=bool(reasoning_text),
                expected="non-empty",
                actual=reasoning_text,
            )
    elif expected_decision == "abstain":
        explicit_abstain = "can't answer" in answer_text.lower()
        system_abstained = empty_grounding or explicit_abstain

        record_check(
            checks,
            "abstain_expected",
            system_abstained,
            expected="abstain",
            actual=answer_text,
        )

        RESULTS.append(
        {
            "case": {"id": case["id"]},
            "checks": checks,
            "question_info": question_info,
            "fetched": fetched,
            "stage_log": stage_log,
            "generated_question_stage": generated_question_stage,
            "entity_context": entity_context,
            "object_context": object_context,
            "answer": answer_text,
            "reasoning": reasoning_text,
            "mapped_entity_answer": mapped_entity_answer,
            "mapped_reasoning_answer": mapped_entity_reasoning,
            "mapped_entity_merged": mapped_entity_merged,
        }
        )

        failures = [c for c in checks if not c["passed"]]
        assert not failures, f"{len(failures)} abstention checks failed: {', '.join(c['name'] for c in failures)}"

        return



    RESULTS.append(
        {
            "case": {"id": case["id"]},
            "checks": checks,
            "question_info": question_info,
            "fetched": fetched,
            "stage_log": stage_log,
            "generated_question_stage": generated_question_stage,
            "entity_context": entity_context,
            "object_context": object_context,
            "answer": answer_text,
            "reasoning": reasoning_text,
            "gold_answer_agent": gold_answer_agent,
            "mapped_reasoning_answer": mapped_entity_reasoning,
            "mapped_entity_answer": mapped_entity_answer,
            "mapped_entity_merged": mapped_entity_merged,
        }
    )

    failures = [
    c for c in checks
    if c["name"] != "grounding_preferred"
    and c["name"] != "grounding_nonempty"
    and not c["passed"]
    ]

    # Grounding fails ONLY if grounding_incorrect is false
    grounding_failure = any(
        c for c in checks
        if c["name"] == "grounding_incorrect" and not c["passed"]
    )

    assert not failures and not grounding_failure, (
        f"{len(failures)} structural checks failed or grounding incorrect: "
        f"{', '.join(c['name'] for c in failures)}"
)
