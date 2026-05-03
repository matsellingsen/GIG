import json
import pytest
import os

from system_v5.tools.inference_module.input_to_graph import atomic_to_graph
from agent_loop.agents.inference_module.extract_triple_agent import ExtractTripleAgent
from agent_loop.agents.inference_module.extract_subject_agent import ExtractSubjectAgent
from agent_loop.agents.inference_module.extract_entity_agent import ExtractEntityAgent
from agent_loop.agents.inference_module.extract_relation_agent import ExtractRelationAgent
from agent_loop.agents.inference_module.extract_question_type_agent import ExtractQuestionTypeAgent
from agent_loop.agents.inference_module.extract_object_agent import ExtractObjectAgent
from agent_loop.agents.inference_module.resolve_answer_form_agent import ResolveAnswerFormAgent

from system_v5.backends import load_backend

# Load dataset once (use relative path so it works on other machines)
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "synthetic_labelled_input.json"))
with open(DATASET_PATH) as f:
    DATASET = json.load(f)

# init backed and agents
backend = load_backend()
extract_entity_agent = ExtractEntityAgent(backend)
extract_relation_agent = ExtractRelationAgent(backend)
extract_question_type_agent = ExtractQuestionTypeAgent(backend)
extract_object_agent = ExtractObjectAgent(backend)
resolve_answer_form_agent = ResolveAnswerFormAgent(backend)

@pytest.mark.parametrize("domain", DATASET.keys())
@pytest.mark.parametrize("tier", ["canonical", "paraphrased", "adversarial"])
@pytest.mark.parametrize("item_index", range(10))
def test_input_to_graph(domain, tier, item_index):
    item = DATASET[domain][tier][item_index]

    atomic_input = item["atomic_input"]
    expected_question_type = item["question_type"]
    expected_answer_form = item["answer_form"]
    expected_entity = item["entity"]["value"]
    expected_relation = item["relation"]
    expected_object = item["object"]["value"]

    result = atomic_to_graph(atomic_input, extract_question_type_agent=extract_question_type_agent, extract_answer_form_agent=resolve_answer_form_agent, extract_entity_agent=extract_entity_agent, extract_relation_agent=extract_relation_agent, extract_object_agent=extract_object_agent)

    # Structure checks
    assert "question_type" in result.keys()
    assert "answer_form" in result.keys()
    assert "entity" in result.keys()
    assert "relation" in result.keys()
    assert "object" in result.keys()

    # Semantic checks
    assert result["question_type"] == item["question_type"]
    assert result["answer_form"] == item["answer_form"]
    assert result["entity"]["value"] == expected_entity #also contains type, but we focus on value for this test
    assert result["relation"] == expected_relation

    if expected_object is not None:
        assert result["object"]["value"] == expected_object # also contains type, but we focus on value for this test
