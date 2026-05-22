import json
from collections import Counter, defaultdict
import os

from sympy import re

RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "reports_archive", "input_to_graph_results_final.json")
)

OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "interpreted_results", "manual_input_to_graph_results_interpreted_final.json")
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

    
    # 2. Compute metrics for each domain separated by tier ("canonical", "paraphrased", "adversarial")
    #    and include an 'overall' summary for each domain.
    domain_tier_metrics = {}

    for domain, tiers in domain_tier_dict.items():
        domain_tier_metrics[domain] = {}
        
        # Calculate stats per tier
        for tier, tier_results in tiers.items():
            tier_stats = {}
            for result in tier_results:
                for check in result.get("checks", []):
                    check_name = check["name"]
                    if not check_name.startswith("has_"): 
                        if check_name not in tier_stats:
                            tier_stats[check_name] = {"passed": 0, "total": 0}
                        
                        tier_stats[check_name]["total"] += 1
                        if check.get("passed"):
                            tier_stats[check_name]["passed"] += 1
                
            # Compute percentages for this tier
            for check_name, counts in tier_stats.items():
                pct = (counts["passed"] / counts["total"] * 100) if counts["total"] > 0 else 0.0
                counts["pct"] = round(pct, 2)
                
            domain_tier_metrics[domain][tier] = tier_stats
            
        # Calculate 'overall' stats across all 3 tiers for this domain
        overall_stats = {}
        for tier_stats in domain_tier_metrics[domain].values():
            for check_name, counts in tier_stats.items():
                if check_name not in overall_stats:
                    overall_stats[check_name] = {"passed": 0, "total": 0}
                overall_stats[check_name]["passed"] += counts["passed"]
                overall_stats[check_name]["total"] += counts["total"]
                
        # Compute percentages for overall domain stats
        for check_name, counts in overall_stats.items():
            pct = (counts["passed"] / counts["total"] * 100) if counts["total"] > 0 else 0.0
            counts["pct"] = round(pct, 2)
            
        domain_tier_metrics[domain]["overall"] = overall_stats

    

    # Compute aggregate metrics separated by tier
    aggregate_by_tier = {
        "canonical": defaultdict(lambda: {"passed": 0, "total": 0}),
        "paraphrased": defaultdict(lambda: {"passed": 0, "total": 0}),
        "adversarial": defaultdict(lambda: {"passed": 0, "total": 0}),
    }

    for domain, tiers in domain_tier_metrics.items():
        for tier in ["canonical", "paraphrased", "adversarial"]:
            if tier not in tiers:
                continue
            for check_name, counts in tiers[tier].items():
                aggregate_by_tier[tier][check_name]["passed"] += counts["passed"]
                aggregate_by_tier[tier][check_name]["total"] += counts["total"]

    # Compute percentages for each tier
    for tier, stats in aggregate_by_tier.items():
        for check_name, counts in stats.items():
            pct = (counts["passed"] / counts["total"] * 100) if counts["total"] > 0 else 0.0
            counts["pct"] = round(pct, 2)

    domain_tier_metrics["aggregate_by_tier"] = aggregate_by_tier

    # Compute aggregate metrics across all domains and tiers
    aggregate_stats = {}
    for domain, tiers in domain_tier_metrics.items():
        if domain == "aggregate_by_tier":
            continue
        for tier, stats in tiers.items():
            if tier == "overall":
                continue # Skip overall to avoid double-counting
            for check_name, counts in stats.items():
                if check_name not in aggregate_stats:
                    aggregate_stats[check_name] = {"passed": 0, "total": 0}
                aggregate_stats[check_name]["passed"] += counts["passed"]
                aggregate_stats[check_name]["total"] += counts["total"]

    # Compute percentages for aggregate stats
    for check_name, counts in aggregate_stats.items():
        pct = (counts["passed"] / counts["total"] * 100) if counts["total"] > 0 else 0.0
        counts["pct"] = round(pct, 2)
    domain_tier_metrics["aggregate"] = aggregate_stats
    
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