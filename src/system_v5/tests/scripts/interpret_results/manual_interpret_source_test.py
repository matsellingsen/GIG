import json
from collections import Counter, defaultdict
import os
import matplotlib.pyplot as plt
import numpy as np

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
        overall_counts = defaultdict(int)
        overall_bases = defaultdict(int)
        
        for qtype in by_system_and_qtype[system]:
            qtype_results = by_system_and_qtype[system][qtype]
            total = len(qtype_results)
            correct_gt = sum(1 for r in qtype_results if r.get("classification") == "Correct (Ground Truth)")
            incomplete = sum(1 for r in qtype_results if r.get("classification") == "Incomplete")
            unprecise = sum(1 for r in qtype_results if r.get("classification") == "Unprecise")
            incorrect_false = sum(1 for r in qtype_results if r.get("classification") == "Incorrect (false information)")
            incorrect_vague = sum(1 for r in qtype_results if r.get("classification") == "Incorrect (vague information)")
            incorrect_abstain = sum(1 for r in qtype_results if r.get("classification") == "Incorrect Abstain")
            correct_abstain = sum(1 for r in qtype_results if r.get("classification") == "Correct Abstain")

            unans_base = sum(1 for r in qtype_results if str(r.get("gold_answer", "")).strip() == "I can't answer the question")
            ans_base = total - unans_base

            counts = {
                "correct_ground_truth": correct_gt,
                "incomplete": incomplete,
                "unprecise": unprecise,
                "incorrect_false": incorrect_false,
                "incorrect_vague": incorrect_vague,
                "incorrect_abstain": incorrect_abstain,
                "correct_abstain": correct_abstain,
            }

            bases = {
                "correct_ground_truth": ans_base,
                "incomplete": ans_base,
                "unprecise": ans_base,
                "incorrect_false": total,
                "incorrect_vague": total,
                "incorrect_abstain": ans_base,
                "correct_abstain": unans_base,
            }

            # Format as fractions
            qtype_metrics = {"total": total}
            for k, v in counts.items():
                qtype_metrics[k] = f"{v}/{bases[k]}"
            
            metrics[system][qtype] = qtype_metrics
            
            overall_bases["total"] += total
            for k, v in counts.items():
                overall_counts[k] += v
                overall_bases[k] += bases[k]

        # Format overall as percentages
        overall_pct = {"total": overall_bases["total"]}
        for k, v in overall_counts.items():
            b = overall_bases[k]
            overall_pct[k] = round((v / b * 100), 2) if b > 0 else 0.0
                
        metrics[system]["overall"] = overall_pct

    return metrics

def generate_bar_chart(metrics):
    """Generate and save a side-by-side bar chart for overall metrics."""
    categories = ['CGT', 'IF', 'I-Abst', 'C-Abst']
    metric_keys = ['correct_ground_truth', 'incorrect_false', 'incorrect_abstain', 'correct_abstain']
    
    gig_scores = [metrics.get('systemA', {}).get('overall', {}).get(k, 0) for k in metric_keys]
    rag_scores = [metrics.get('systemB', {}).get('overall', {}).get(k, 0) for k in metric_keys]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    rects1 = ax.bar(x - width/2, gig_scores, width, label='GIG', color='#1f77b4')
    rects2 = ax.bar(x + width/2, rag_scores, width, label='RAG', color='#ff7f0e')
    
    ax.set_ylabel('Percentage (0-100%)')
    ax.set_title('Overall Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    
    img_path = os.path.join(os.path.dirname(OUTPUT_PATH), "manual_source_results_barchart.png")
    fig.tight_layout()
    plt.savefig(img_path)
    plt.close()
    print(f"Bar chart saved to {img_path}")

def main():
    results = load_results()
    metrics = compute_metrics(results)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics written to {OUTPUT_PATH}")
    generate_bar_chart(metrics)

if __name__ == "__main__":
    main()
