import json
import pytest
import os
import difflib
import re

from system_v5.backends import load_backend
from system_v5.tools.inference_module.map_answer_to_context import map_answer_to_context, merge_mappings
from system_v5.tools.ttl_handling.load_ttl import load_ttl
from system_v5.tools.ttl_handling.resolve_ttl_path import resolve_ttl_path
from system_v5.tools.inference_module.input_to_graph import atomic_to_graph
from system_v5.tools.inference_module.fetch_relevant_info import fetch_relevant_info
from system_v5.tools.inference_module.generate_answer import generate_answer

from agent_loop.agents.inference_module.extract_question_type_agent import ExtractQuestionTypeAgent
from system_v5.agent_loop.agents.inference_module.resolve_answer_form_agent import ResolveAnswerFormAgent
from agent_loop.agents.inference_module.extract_entity_agent import ExtractEntityAgent
from agent_loop.agents.inference_module.extract_relation_agent import ExtractRelationAgent
from agent_loop.agents.inference_module.extract_object_agent import ExtractObjectAgent
from agent_loop.agents.inference_module.resolve_entity_agent import ResolveEntityAgent
from agent_loop.agents.inference_module.generate_answer_agent import GenerateAnswerAgent

DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset", "source_grounded_evaluationDataset.json")
)

REPORT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "reports", "GIG_source_dataset_results.json")
)

with open(DATASET_PATH, encoding="utf-8") as f:
    CASES = json.load(f)

RESULTS = []

def _is_fuzzy_match(expected: str, actual: str, ratio_threshold: float = 0.45) -> bool:
    """
    Returns True if 'actual' is a reasonable fuzzy match for 'expected'.
    Mirrors scoring logic from retrieve_top_candidates in inference module.
    """
    if expected is None or actual is None:
        return expected == actual
        
    # Handle the specific abstain mapping scenario accurately instead of relying purely on fuzzy sequence matching
    if expected.lower().strip() == "i can't answer the question":
        return "can't answer" in actual.lower() or "cannot answer" in actual.lower() or "not explicitly stated" in actual.lower()

    exp_norm = expected.lower().strip()
    act_norm = actual.lower().strip()
    if exp_norm.endswith("s"): exp_norm = exp_norm[:-1]
    if act_norm.endswith("s"): act_norm = act_norm[:-1]
    
    exp_tokens = set(re.findall(r'[a-zA-Z]+', exp_norm))
    act_tokens = set(re.findall(r'[a-zA-Z]+', act_norm))
    
    sim_ratio = difflib.SequenceMatcher(None, exp_norm, act_norm).ratio()
    
    act_tokens_filtered = {t for t in act_tokens if len(t) >= 4}
    overlap = len(exp_tokens & act_tokens_filtered) / max(len(exp_tokens), 1)
    substring = 1.0 if any(t in act_tokens_filtered for t in exp_tokens) else 0.0
    
    score = (0.80 * sim_ratio) + (0.15 * overlap) + (0.05 * substring)
    return score >= ratio_threshold

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
def test_source_dataset(case, ttl_fixture, agents):
    checks = []

    # 1. Transform atomic question to graph representation
    question_info = atomic_to_graph(
        case["question"],
        extract_question_type_agent=agents["extract_question_type"],
        extract_answer_form_agent=agents["resolve_answer_form"],
        extract_entity_agent=agents["extract_entity"],
        extract_relation_agent=agents["extract_relation"],
        extract_object_agent=agents["extract_object"],
    )
    question_info["atomic_question"] = case["question"]

    # 2. Fetch context from ontology
    fetched = fetch_relevant_info(
        question_info=question_info,
        ttl=ttl_fixture,
        resolve_entity_agent=agents["resolve_entity"],
    )

    entity_context = fetched.get("entity_context", {}) if isinstance(fetched, dict) else {}
    object_context = fetched.get("object_context", {}) if isinstance(fetched, dict) else {}

    # 3. Generate Answer
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

    # Set up expected / actual comparisons
    expected_qt = case.get("question_type")
    expected_af = case.get("answer_form")
    expected_ans = case.get("gold_answer")
    expected_entity = case.get("expected_entity", {})
    expected_object = case.get("expected_object", {})
    if expected_object == None:
        expected_object = "null"

    # get generated values
    actual_question_type = question_info.get("question_type")
    actual_answer_form = question_info.get("answer_form")
    actual_extracted_entity = question_info.get("entity", {}).get("value")
    #actual_relation = question_info.get("relation")
    actual_extracted_object = question_info.get("object", {}).get("value")
    actual_resolved_entity = fetched.get("resolved_entity").get("label") if isinstance(fetched, dict) else None
    actual_resolved_object = fetched.get("resolved_object") if isinstance(fetched, dict) else None
    
    # Perform structural checks
    record_check(checks, "correct_question_type", actual_question_type == expected_qt, expected=expected_qt, actual=actual_question_type)
    record_check(checks, "correct_answer_form", actual_answer_form == expected_af, expected=expected_af, actual=actual_answer_form)
    record_check(checks, "correct_entity_extraction", actual_extracted_entity == expected_entity, expected=expected_entity, actual=actual_extracted_entity)
    record_check(checks, "correct_object_extraction", actual_extracted_object == expected_object, expected=expected_object, actual=actual_extracted_object)
    record_check(checks, "entity_resolved", actual_resolved_entity is not None, expected=True, actual=actual_resolved_entity)
   
    if expected_object is not "null" and actual_question_type == "comparative": # Comparative quesitons also retrieve context for object and so it must be resolved.
        record_check(checks, "object_resolved", actual_resolved_object is not None, expected=True, actual=actual_resolved_object)
    
    # Perform fuzzy score checking
    is_fuzzy_ok = _is_fuzzy_match(expected_ans, answer_text, ratio_threshold=0.45)
    record_check(checks, "fuzzy_answer_match", is_fuzzy_ok, expected=expected_ans, actual=answer_text)
    is_fuzzy_entity_ok = _is_fuzzy_match(expected_entity, actual_extracted_entity, ratio_threshold=0.45)
    record_check(checks, "fuzzy_entity_match", is_fuzzy_entity_ok, expected=expected_entity, actual=actual_extracted_entity)
    is_fuzzy_object_ok = _is_fuzzy_match(expected_object, actual_extracted_object, ratio_threshold=0.45)
    record_check(checks, "fuzzy_object_match", is_fuzzy_object_ok, expected=expected_object, actual=actual_extracted_object)
     

    RESULTS.append({
        "case": case,
        "question_info": question_info,
        "fetched": fetched,
        "generated_answer": answer_text,
        "reasoning": reasoning_text,
        "merged_entity_mapping": mapped_entity_merged,
        "merged_object_mapping": mapped_object_merged if object_context else None,
        "checks": checks
    })

    failures = [c for c in checks if not c["passed"]]
    assert not failures, f"{len(failures)} checks failed: {', '.join(c['name'] for c in failures)}"
