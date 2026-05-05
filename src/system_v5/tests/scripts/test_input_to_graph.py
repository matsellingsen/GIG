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

SAMPLE_SIZE = int(os.getenv("INPUT_TO_GRAPH_SAMPLE_SIZE", "10"))
TIERS = ["canonical", "paraphrased", "adversarial"]


def build_cases():
    cases = []
    for domain, domain_data in DATASET.items():
        for tier in TIERS:
            tier_items = domain_data.get(tier, [])
            limit = min(SAMPLE_SIZE, len(tier_items))
            for item_index in range(limit):
                cases.append({"domain": domain, "tier": tier, "item_index": item_index})
    return cases


CASES = build_cases()

RESULTS = []
REPORT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "reports", "input_to_graph_results.json")
)

# init backed and agents
backend = load_backend()
extract_entity_agent = ExtractEntityAgent(backend)
extract_relation_agent = ExtractRelationAgent(backend)
extract_question_type_agent = ExtractQuestionTypeAgent(backend)
extract_object_agent = ExtractObjectAgent(backend)
resolve_answer_form_agent = ResolveAnswerFormAgent(backend)


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

@pytest.mark.llm
@pytest.mark.parametrize("case", CASES)
def test_input_to_graph(case):
    domain = case["domain"]
    tier = case["tier"]
    item_index = case["item_index"]
    item = DATASET[domain][tier][item_index]

    atomic_input = item["atomic_input"]
    expected_question_type = item["question_type"]
    expected_answer_form = item["answer_form"]
    expected_entity = item["entity"]["value"]
    expected_relation = item["relation"]
    expected_object = item["object"]["value"]

    checks = []
    case_meta = {
        "domain": domain,
        "tier": tier,
        "item_index": item_index,
        "atomic_input": atomic_input,
    }

    try:
        result = atomic_to_graph(
            atomic_input,
            extract_question_type_agent=extract_question_type_agent,
            extract_answer_form_agent=resolve_answer_form_agent,
            extract_entity_agent=extract_entity_agent,
            extract_relation_agent=extract_relation_agent,
            extract_object_agent=extract_object_agent,
        )
        record_check(checks, "atomic_to_graph_exception", True, expected="no exception", actual="no exception")
    except Exception as exc:
        result = {}
        record_check(checks, "atomic_to_graph_exception", False, expected="no exception", actual=str(exc))

    # Structure checks
    record_check(checks, "has_question_type", "question_type" in result, expected=True, actual="question_type" in result)
    record_check(checks, "has_answer_form", "answer_form" in result, expected=True, actual="answer_form" in result)
    record_check(checks, "has_entity", "entity" in result, expected=True, actual="entity" in result)
    record_check(checks, "has_relation", "relation" in result, expected=True, actual="relation" in result)
    record_check(checks, "has_object", "object" in result, expected=True, actual="object" in result)

    actual_question_type = result.get("question_type")
    actual_answer_form = result.get("answer_form")
    actual_entity = result.get("entity", {}).get("value")
    actual_relation = result.get("relation")
    actual_object = result.get("object", {}).get("value")

    qt_ok = actual_question_type == expected_question_type
    af_ok = actual_answer_form == expected_answer_form
    entity_ok = actual_entity == expected_entity
    relation_ok = actual_relation == expected_relation
    object_ok = actual_object == expected_object if expected_object is not None else None

    # Semantic checks
    record_check(
        checks,
        "question_type_match",
        qt_ok,
        expected=expected_question_type,
        actual=actual_question_type,
    )
    record_check(
        checks,
        "answer_form_match",
        af_ok,
        expected=expected_answer_form,
        actual=actual_answer_form,
    )
    record_check(
        checks,
        "entity_match",
        entity_ok,
        expected=expected_entity,
        actual=actual_entity,
    )
    record_check(
        checks,
        "relation_match",
        relation_ok,
        expected=expected_relation,
        actual=actual_relation,
    )

    if expected_object is not None:
        record_check(
            checks,
            "object_match",
            object_ok,
            expected=expected_object,
            actual=actual_object,
        )

    attribution = {
        "question_type_match": "pass" if qt_ok else "direct",
        "answer_form_match": "pass" if af_ok else "direct",
        "entity_match": "pass" if entity_ok else ("direct" if qt_ok else "cascade"),
        "relation_match": "pass" if relation_ok else ("direct" if (qt_ok and entity_ok) else "cascade"),
    }
    if expected_object is not None:
        attribution["object_match"] = (
            "pass"
            if object_ok
            else ("direct" if (qt_ok and entity_ok and relation_ok) else "cascade")
        )

    RESULTS.append({"case": case_meta, "checks": checks, "attribution": attribution})
    failures = [c for c in checks if not c["passed"]]
    assert not failures, f"{len(failures)} checks failed: {', '.join(c['name'] for c in failures)}"
