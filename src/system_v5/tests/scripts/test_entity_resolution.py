import json
import os
import pytest

from system_v5.backends import load_backend
from system_v5.tools.ttl_handling.load_ttl import load_ttl
from system_v5.tools.ttl_handling.resolve_ttl_path import resolve_ttl_path
from system_v5.tools.inference_module.fetch_relevant_info import fetch_relevant_info, resolve_entity

from agent_loop.agents.inference_module.resolve_entity_agent import ResolveEntityAgent


UNRESOLVABLE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset", "entity_resolution_evaluationDataset.json")
)
REPORT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "reports", "resolution_component_results.json")
)

with open(UNRESOLVABLE_PATH, encoding="utf-8") as f:
    loaded = json.load(f)
CASES = loaded["unresolvable"] if isinstance(loaded, dict) else loaded

RESULTS = []

def record_check(checks, name, passed, expected=None, actual=None):
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "expected": expected,
            "actual": actual,
        }
    )

def build_stage_log(question_info, fetched=None):
    return {
        "input_to_graph": {
            "question_type": question_info.get("question_type"),
            "answer_form": question_info.get("answer_form"),
            "entity": question_info.get("entity", {}).get("value"),
            "relation": question_info.get("relation"),
            "object": question_info.get("object", {}).get("value"),
        },
        "resolve_entity": {
            "resolved_info": fetched if isinstance(fetched, dict) else None,
            "status": "failed" if isinstance(fetched, str) else "present",
        }
    }


@pytest.fixture(scope="session", autouse=True)
def write_report():
    yield
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2)

@pytest.fixture(scope="session")
def ttl_fixture():
    return load_ttl(file_path=resolve_ttl_path())


@pytest.fixture(scope="session")
def backend():
    return load_backend()


@pytest.fixture(scope="session")
def agents(backend):
    return {
        "resolve_entity": ResolveEntityAgent(backend),
    }


@pytest.mark.llm
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_entity_resolution(case, ttl_fixture, agents):
    checks = []

    # Bypass input_to_graph
    question_info = {
        "atomic_question": case.get("atomic_input"),
        "question_type": case.get("question_type"),
        "answer_form": case.get("answer_form"),
        "entity": case.get("entity", {}),
        "relation": case.get("relation"),
        "object": case.get("object", {})
    }

    # Run entity resolution
    fetched = resolve_entity(
        type="entity",
        question_info=question_info,
        graph=ttl_fixture.get("graph", {}),
        resolve_entity_agent=agents["resolve_entity"],
    )

    if "gold_resolve" in case:
        record_check(checks, "resolution_success", isinstance(fetched, dict), "dict with resolved info", type(fetched).__name__)
        actual_resolved_label = fetched.get("label") if isinstance(fetched, dict) else None
        expected_label = case["gold_resolve"]
        record_check(checks, "correct_resolved_entity", actual_resolved_label == expected_label, expected_label, actual_resolved_label)
    else:
        # Must fail resolution
        record_check(checks, "resolution_failure", isinstance(fetched, str), "string error message", type(fetched).__name__)

    RESULTS.append(
        {
            "case": case,
            "checks": checks,
            "question_info": question_info,
            "fetched": fetched,
            "stage_log": build_stage_log(question_info, fetched=fetched),
        }
    )

    failures = [c for c in checks if not c["passed"]]
    assert not failures, f"{len(failures)} checks failed: {', '.join(c['name'] for c in failures)}"
