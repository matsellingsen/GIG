import json
from collections import Counter, defaultdict
import os

from sympy import re

RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "reports_archive", "input_to_graph_results_complete.json")
)

OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "interpreted_results", "manual_input_to_graph_results_interpreted.json")
)


def load_results():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(results):
    # 1. Divide by domain and then further by tier
    domain_tier_dict = defaultdict(lambda: defaultdict(list))
    for result in results:
        domain = result["case"]["domain"]
        tier = result["case"]["tier"]
        domain_tier_dict[domain][tier].append(result)

    
    # 2. Compute metrics for each domain seperated by tier ("canonical", "paraphrased", "adversarial")

    domain_tier_metrics = defaultdict(lambda: defaultdict(list))
    for domain, tiers in domain_tier_dict.items():
        for tier, tier_results in tiers.items():
            for result in tier_results:
                checks = result["checks"]

            # Compute pass rate for each check
            check_pass_rates = {}
            for check in checks:
                check_name = check["name"]
                passed = check["passed"]
                if check_name not in check_pass_rates:
                    check_pass_rates[check_name] = {"passed": 0, "total": 0}
                check_pass_rates[check_name]["total"] += 1
                if passed:
                    check_pass_rates[check_name]["passed"] += 1
            for check_name, counts in check_pass_rates.items():
                pass_rate = counts["passed"] / counts["total"] if counts["total"] > 0 else 0.0
                domain_tier_metrics[domain][tier].append({
                    "check_name": check_name,
                    "pass_rate": pass_rate,
                    "passed": counts["passed"],
                    "total": counts["total"]
                })

    return domain_tier_metrics


def main():
    results = load_results()
    metrics = compute_metrics(results)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()