import json
from collections import Counter, defaultdict
import os

RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "interpreted_results", "manually_evaluated_source_results.jsonl")
)

OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "interpreted_results", "manual_source_results_complete_interpreted.json")
)


def load_results():
    """Load JSONL results file."""
    results = []
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def _map_questionId_to_category(question_id):
    """Helper function to map question_id to a category based on its number."""
    q_num = int(question_id[1:])  # Extract numeric part, e.g. "q46" -> 46
    if 1 <= q_num <= 7:
        return "definition"
    elif 8 <= q_num <= 14:
        return "taxonomic"
    elif 15 <= q_num <= 21:
        return "capability"
    elif 22 <= q_num <= 28:
        return "property"
    elif 29 <= q_num <= 35:
        return "membership"
    elif 36 <= q_num <= 42:
        return "comparative"
    elif 43 <= q_num <= 49:
        return "quantification"
    else:
        assert False, f"Unexpected question_id format: {question_id}"

def compute_metrics(results):
    """Compute metrics grouped by question and system."""
    
    #1. Group results by system-type and question-type
    by_system_and_qtype = defaultdict(lambda: defaultdict(list))
    for result in results:
        system = result.get("system", "unknown")
        question_type = _map_questionId_to_category(result.get("question_id", "unknown"))
        by_system_and_qtype[system][question_type].append(result)

    # 2. Compute metrics for each system and question type
    metrics = {}
    for system in by_system_and_qtype:
        metrics[system] = {}
        for qtype in by_system_and_qtype[system]:
            qtype_results = by_system_and_qtype[system][qtype]
            total = len(qtype_results)
            correct_gt = sum(1 for r in qtype_results if r.get("classification") == "Correct (Ground Truth)")
            incomplete = sum(1 for r in qtype_results if r.get("classification") == "Incomplete")
            unprecise = sum(1 for r in qtype_results if r.get("classification") == "Unprecise")
            incorrect_false = sum(1 for r in qtype_results if r.get("classification") == "Incorrect (false information)")
            incorrect_vague = sum(1 for r in qtype_results if r.get("classification") == "Incorrect (vague information)")
            incorrect_abstain = sum(1 for r in qtype_results if r.get("classification") == "Incorrect abstain")
            correct_abstain = sum(1 for r in qtype_results if r.get("classification") == "Correct abstain")

            metrics[system][qtype] = {
                "total": total,
                "correct_ground_truth": correct_gt,
                "incomplete": incomplete,
                "unprecise": unprecise,
                "incorrect_false": incorrect_false,
                "incorrect_vague": incorrect_vague,
                "incorrect_abstain": incorrect_abstain,
                "correct_abstain": correct_abstain,
                "correct_gt_pct": round(correct_gt / total * 100, 2) if total > 0 else 0.0,
                "incomplete_pct": round(incomplete / total * 100, 2) if total > 0 else 0.0,
                "unprecise_pct": round(unprecise / total * 100, 2) if total > 0 else 0.0,
                "incorrect_false_pct": round(incorrect_false / total * 100, 2) if total > 0 else 0.0,
                "incorrect_vague_pct": round(incorrect_vague / total * 100, 2) if total > 0 else 0.0,
                "incorrect_abstain_pct": round(incorrect_abstain / total * 100, 2) if total > 0 else 0.0,
                "correct_abstain_pct": round(correct_abstain / total * 100, 2) if total > 0 else 0.0,}
    
    # 3. Optionally, compute overall metrics across all question types for each system
    for system in metrics:
        total = sum(metrics[system][qtype]["total"] for qtype in metrics[system])
        correct_gt = sum(metrics[system][qtype]["correct_ground_truth"] for qtype in metrics[system])
        incomplete = sum(metrics[system][qtype]["incomplete"] for qtype in metrics[system])
        unprecise = sum(metrics[system][qtype]["unprecise"] for qtype in metrics[system])
        incorrect_false = sum(metrics[system][qtype]["incorrect_false"] for qtype in metrics[system])
        incorrect_vague = sum(metrics[system][qtype]["incorrect_vague"] for qtype in metrics[system])
        incorrect_abstain = sum(metrics[system][qtype]["incorrect_abstain"] for qtype in metrics[system])
        correct_abstain = sum(metrics[system][qtype]["correct_abstain"] for qtype in metrics[system])

        metrics[system]["overall"] = {
            "total": total,
            "correct_ground_truth": correct_gt,
            "incomplete": incomplete,
            "unprecise": unprecise,
            "incorrect_false": incorrect_false,
            "incorrect_vague": incorrect_vague,
            "incorrect_abstain": incorrect_abstain,
            "correct_abstain": correct_abstain,
            "correct_gt_pct": round(correct_gt / total * 100, 2) if total > 0 else 0.0,
            "incomplete_pct": round(incomplete / total * 100, 2) if total > 0 else 0.0,
            "unprecise_pct": round(unprecise / total * 100, 2) if total > 0 else 0.0,
            "incorrect_false_pct": round(incorrect_false / total * 100, 2) if total > 0 else 0.0,
            "incorrect_vague_pct": round(incorrect_vague / total * 100, 2) if total > 0 else 0.0,
            "incorrect_abstain_pct": round(incorrect_abstain / total * 100, 2) if total > 0 else 0.0,
            "correct_abstain_pct": round(correct_abstain / total * 100, 2) if total > 0 else 0.0,}
    return metrics


def main():
    results = load_results()
    metrics = compute_metrics(results)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
