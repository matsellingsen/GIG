import json
import os
import pytest
from rdflib import Graph
from system_v5.tools.ttl_handling.resolve_ttl_path import resolve_ttl_path
from system_v5.tools.ttl_handling.load_ttl import load_ttl
from system_v5.backends import load_backend
from system_v5.tools.inference_module.fetch_relevant_info import resolve_primary_entity
from system_v5.tools.inference_module.fetch_relevant_info import retrieve_top_candidates
from system_v5.agent_loop.agents.inference_module.resolve_entity_agent import ResolveEntityAgent

# -------------------------------------------------------------------
# Load ontology once for all tests
# -------------------------------------------------------------------
ttl_path = resolve_ttl_path(ttl_path=None)
graph = load_ttl(file_path=ttl_path)

# load backend and agent once for all tests
backend = load_backend(name="phi-npu-openvino")
resolve_agent = ResolveEntityAgent(backend=backend)

# -------------------------------------------------------------------
# Load gold-standard dataset
# -------------------------------------------------------------------
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "real_entity_resolution_gold.json"))
with open(DATASET_PATH) as f:
    GOLD = json.load(f)

RESULTS = []
REPORT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "reports", "real_entity_resolution_results.json")
)


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


# -------------------------------------------------------------------
# Helper: pretty-print candidate debugging info
# -------------------------------------------------------------------
def debug_candidates(entity_value, candidates):
    print("\n--- DEBUG: Candidates for", entity_value, "---")
    for c in candidates:
        print(f"Label: {c['label']}, Score: {c['score']}, URI: {c['uri']}")
    print("--------------------------------------------------\n")


# -------------------------------------------------------------------
# Test: resolve_primary_entity
# -------------------------------------------------------------------
@pytest.mark.parametrize("case", GOLD)
def test_resolve_primary_entity(case):
    """
    Component-level test for entity resolution.
    Ensures that resolve_primary_entity() selects the correct ontology entity.
    """

    entity_value = case["question_info"]["entity"]["value"]

    # Run resolver
    checks = []
    resolved = resolve_primary_entity(
        question_info=case["question_info"],
        graph=graph["graph"],
        resolve_entity_agent=resolve_agent
    )

    record_check(
        checks,
        "resolved_entity",
        resolved is not None,
        expected="found",
        actual="not found" if resolved is None else "found",
    )

    actual_uri = resolved["uri"] if resolved else None
    record_check(
        checks,
        "uri_match",
        actual_uri == case["expected_uri"],
        expected=case["expected_uri"],
        actual=actual_uri,
    )

    if resolved is None or actual_uri != case["expected_uri"]:
        candidates = retrieve_top_candidates(graph["graph"], entity_value, top_n=5)
        debug_candidates(entity_value, candidates)

    RESULTS.append(
        {
            "case": {
                "entity_value": entity_value,
                "expected_uri": case["expected_uri"],
            },
            "checks": checks,
        }
    )
    failures = [c for c in checks if not c["passed"]]
    assert not failures, f"{len(failures)} checks failed: {', '.join(c['name'] for c in failures)}"
