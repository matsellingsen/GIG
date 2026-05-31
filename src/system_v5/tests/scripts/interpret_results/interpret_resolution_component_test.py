import json
import os
from collections import defaultdict

RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "reports_archive", "resolution_component_results.json")
)

OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "interpreted_results", "resolution_component_results_interpreted.json")
)

def load_results():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)

def compute_metrics(results):
    metrics = {
        "unresolvable": {
            "resolution_failure": {"passed": 0, "total": 0, "pct": 0.0}
        },
        "resolvable": {
            "canonical": {
                "resolution_success": {"passed": 0, "total": 0, "pct": 0.0},
                "correct_resolved_entity": {"passed": 0, "total": 0, "pct": 0.0}
            },
            "paraphrased": {
                "resolution_success": {"passed": 0, "total": 0, "pct": 0.0},
                "correct_resolved_entity": {"passed": 0, "total": 0, "pct": 0.0}
            },
            "typo": {
                "resolution_success": {"passed": 0, "total": 0, "pct": 0.0},
                "correct_resolved_entity": {"passed": 0, "total": 0, "pct": 0.0}
            },
            "overall": {
                "resolution_success": {"passed": 0, "total": 0, "pct": 0.0},
                "correct_resolved_entity": {"passed": 0, "total": 0, "pct": 0.0}
            }
        }
    }

    for result in results:
        case = result.get("case", {})
        case_id = case.get("id", "")
        checks = result.get("checks", [])
        
        if case_id.startswith("unresolvable"):
            for check in checks:
                if check["name"] == "resolution_failure":
                    metrics["unresolvable"]["resolution_failure"]["total"] += 1
                    if check.get("passed"):
                        metrics["unresolvable"]["resolution_failure"]["passed"] += 1
        else:
            tier = "unknown"
            if case_id.endswith("_1"):
                tier = "canonical"
            elif case_id.endswith("_2"):
                tier = "paraphrased"
            elif case_id.endswith("_3"):
                tier = "typo"
            
            if tier in metrics["resolvable"]:
                for check in checks:
                    check_name = check["name"]
                    if check_name in metrics["resolvable"][tier]:
                        metrics["resolvable"][tier][check_name]["total"] += 1
                        metrics["resolvable"]["overall"][check_name]["total"] += 1
                        if check.get("passed"):
                            metrics["resolvable"][tier][check_name]["passed"] += 1
                            metrics["resolvable"]["overall"][check_name]["passed"] += 1

    # Format percentages
    unres = metrics["unresolvable"]["resolution_failure"]
    if unres["total"] > 0:
        unres["pct"] = round((unres["passed"] / unres["total"]) * 100, 2)
        
    for tier, checks in metrics["resolvable"].items():
        for chk_name, counts in checks.items():
            if counts["total"] > 0:
                counts["pct"] = round((counts["passed"] / counts["total"]) * 100, 2)
                
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
