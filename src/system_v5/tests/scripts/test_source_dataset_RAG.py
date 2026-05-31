import json
import pytest
import os
import difflib
import re

from system_v5.backends import load_backend
from system_v5.tests.baselines.naive_RAG import NaiveRAG

DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset", "source_grounded_evaluationDataset.json")
)

REPORT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "reports", "RAG_source_dataset_results.json")
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
def RAG_model():
    return NaiveRAG()


@pytest.fixture(scope="session", autouse=True)
def write_report():
    yield
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2)

@pytest.mark.llm
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_source_dataset(case, RAG_model):
    checks = []

    
    # 1. Run RAG
    result = RAG_model.run(case["question"])
    answer_text = result.get("answer", "")
    contexts = result.get("contexts", [])

    # record checks
    record_check(checks, "answer_fuzzy_match", _is_fuzzy_match(case["gold_answer"], answer_text), expected=case["gold_answer"], actual=answer_text)

    RESULTS.append({
        "case": case,
        "generated_answer": answer_text,
        "contexts": contexts,
        "checks": checks
    })

    failures = [c for c in checks if not c["passed"]]
    assert not failures, f"{len(failures)} checks failed: {', '.join(c['name'] for c in failures)}"
