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
    resolved = resolve_primary_entity(
        question_info=case["question_info"],
        graph=graph["graph"],
        resolve_entity_agent=resolve_agent
    )

    # If resolution failed, print candidates for debugging
    if resolved is None:
        candidates = []  # no candidates found
        debug_candidates(entity_value, candidates)
        assert False, f"Entity resolution failed for: {entity_value}"

    # Debug output if mismatch
    if resolved["uri"] != case["expected_uri"]:
        
        candidates = retrieve_top_candidates(graph["graph"], entity_value, top_n=5)
        debug_candidates(entity_value, candidates)

    # Final assertion
    assert resolved["uri"] == case["expected_uri"], (
        f"Resolved entity mismatch for '{entity_value}'.\n"
        f"Expected: {case['expected_uri']}\n"
        f"Got:      {resolved['uri']}"
    )
