import json
import os
import pytest

from system_v5.backends import load_backend
from system_v5.tools.ttl_handling.load_ttl import load_ttl
from system_v5.tools.ttl_handling.resolve_ttl_path import resolve_ttl_path
from system_v5.tools.inference_module.input_to_graph import atomic_to_graph
from system_v5.tools.inference_module.fetch_relevant_info import fetch_relevant_info

from agent_loop.agents.inference_module.extract_question_type_agent import ExtractQuestionTypeAgent
from system_v5.agent_loop.agents.inference_module.resolve_answer_form_agent import ResolveAnswerFormAgent
from agent_loop.agents.inference_module.extract_entity_agent import ExtractEntityAgent
from agent_loop.agents.inference_module.extract_relation_agent import ExtractRelationAgent
from agent_loop.agents.inference_module.extract_object_agent import ExtractObjectAgent
from agent_loop.agents.inference_module.resolve_entity_agent import ResolveEntityAgent


UNRESOLVABLE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset", "unresolvable_simple_questions.json")
)

with open(UNRESOLVABLE_PATH, encoding="utf-8") as f:
    loaded = json.load(f)
CASES = loaded["unresolvable"] if isinstance(loaded, dict) else loaded


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
    }


@pytest.mark.llm
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["atomic_input"])
def test_unresolvable_entities(case, ttl_fixture, agents):
    """
    Tests that:
    - question_type = definition
    - answer_form = value
    - entity extracted correctly
    - relation = be
    - object = null
    - entity resolution fails with noPrimaryEntityFound
    - system abstains
    """

    # Run input_to_graph
    question_info = atomic_to_graph(
        case["atomic_input"],
        extract_question_type_agent=agents["extract_question_type"],
        extract_answer_form_agent=agents["resolve_answer_form"],
        extract_entity_agent=agents["extract_entity"],
        extract_relation_agent=agents["extract_relation"],
        extract_object_agent=agents["extract_object"],
    )

    # Structural checks
    assert question_info["question_type"] == "definition", "Incorrect question type"
    assert question_info["answer_form"] == "value", "Incorrect answer form"

    extracted_entity = question_info.get("entity", {}).get("value")
    assert extracted_entity == case["entity"]["value"], f"Entity extraction failed: {extracted_entity}"

    assert question_info["relation"] == "be", "Relation should be 'be'"
    assert question_info["object"]["value"] == "null", "Object should be null for definition questions"

    # Run entity resolution
    fetched = fetch_relevant_info(
        question_info=question_info,
        ttl=ttl_fixture,
        resolve_entity_agent=agents["resolve_entity"],
    )

    # Must fail resolution
    assert fetched == "noPrimaryEntityFound", f"Expected unresolvable entity, got: {fetched}"
