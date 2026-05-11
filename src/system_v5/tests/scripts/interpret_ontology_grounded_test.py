import json
from collections import Counter, defaultdict
import os

RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "reports_archive", "ontology_grounded_results_complete.json")
)

OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "interpreted_results", "ontology_grounded_results_complete_interpreted.json")
)


def load_results():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def safe_get(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


def compute_metrics(results):
    metrics = {}

    total = len(results)

    # ---------------------------------------------------------
    # 1. Extraction Accuracy (per field + combined)
    # ---------------------------------------------------------
    field_map = {
        "question_type": "correct_question_type",
        "answer_form": "correct_answer_form",
        "entity": "correct_extracted_entity",
        "relation": "correct_relation_extraction",
        "object": "correct_object_extraction",
    }

    correct_counts = Counter()

    for r in results:
        checks = r["checks"]
        for field, check_name in field_map.items():
            if any(c["name"] == check_name and c["passed"] for c in checks):
                correct_counts[field] += 1

    combined_correct = 0
    for r in results:
        checks = r["checks"]
        if all(
            any(c["name"] == check_name and c["passed"] for c in checks)
            for check_name in field_map.values()
        ):
            combined_correct += 1

    metrics["extraction_accuracy"] = {
        field: correct_counts[field] / total for field in field_map.keys()
    }
    metrics["extraction_accuracy"]["combined"] = combined_correct / total

    # ---------------------------------------------------------
    # 2. Entity Resolution Metrics
    # ---------------------------------------------------------
    resolution_counts = Counter()

    for r in results:
        fetched = r.get("fetched")
        if fetched == "noPrimaryEntityFound":
            resolution_counts["noPrimaryEntityFound"] += 1
        elif fetched == "noComparativeObjectFound":
            resolution_counts["noComparativeObjectFound"] += 1
        else:
            resolution_counts["resolved"] += 1

    metrics["entity_resolution"] = {
        "resolved": resolution_counts["resolved"] / total,
        "noPrimaryEntityFound": resolution_counts["noPrimaryEntityFound"] / total,
        "noComparativeObjectFound": resolution_counts["noComparativeObjectFound"] / total,
    }

    # ---------------------------------------------------------
    # 3. Answer Generation Metrics
    # ---------------------------------------------------------
    answerable = [r for r in results if "_unanswerable" not in r["case"]["id"]]#safe_get(r, "case", "name") == "answer_not_abstain"]
    unanswerable = [r for r in results if "_unanswerable" in r["case"]["id"]]#safe_get(r, "case", "name") == "abstain_expected"]

    def is_explicit_abstain(text):
        return text is None or "can't answer" in (text or "").lower()

    metrics["answer_generation"] = {
        "answerable_nonempty_rate": sum(bool(r.get("answer")) for r in answerable) / max(len(answerable), 1),
        "unanswerable_correct_abstain_rate": sum(is_explicit_abstain(r.get("answer")) for r in unanswerable) / max(len(unanswerable), 1),
        #"hallucination_rate": sum(bool(r.get("answer")) for r in unanswerable) / max(len(unanswerable), 1),
        "avg_answer_length": sum(len(r.get("answer") or "") for r in results) / total,
    }

    # ---------------------------------------------------------
    # 4. Grounding Metrics (answerable cases only)
    # ---------------------------------------------------------
    grounding_pref = 0
    grounding_nonempty = 0
    answerable_count = len(answerable)

    for r in answerable:
        checks = r["checks"]
        if any(c["name"] == "grounding_preferred" and c["passed"] for c in checks):
            grounding_pref += 1
        if any(c["name"] == "grounding_nonempty" and c["passed"] for c in checks):
            grounding_nonempty += 1
        

    if answerable_count > 0:
        preferred_rate = grounding_pref / answerable_count
        nonempty_rate = grounding_nonempty / answerable_count
    else:
        preferred_rate = 0.0
        nonempty_rate = 0.0

    metrics["grounding"] = {
        "preferred_rate": preferred_rate,
        "nonempty_rate": nonempty_rate,
        "empty_rate": 1.0 - nonempty_rate,
    }

    # ---------------------------------------------------------
    # 5. Pipeline Stability (cascading failures on answerable cases)
    # ---------------------------------------------------------
    cascade_failures = 0

    for r in answerable:
        checks = r["checks"]
        structural_fail = any(
            c["name"] in field_map.values() and not c["passed"] for c in checks
        )
        grounding_failure = any(
            c["name"] == "grounding_incorrect" and not c["passed"] for c in checks
        )
        if structural_fail or grounding_failure:
            cascade_failures += 1

    metrics["pipeline_stability"] = {
        "cascade_failure_rate": cascade_failures / max(answerable_count, 1)
    }

    # ---------------------------------------------------------
    # 6. Expected Decision Compliance
    # ---------------------------------------------------------
    compliance = 0

    for r in results:
        expected = safe_get(r, "case", "expected_decision")
        checks = r["checks"]
        if expected == "answer":
            if any(c["name"] == "answer_not_abstain" and c["passed"] for c in checks):
                compliance += 1
        elif expected == "abstain":
            if any(c["name"] == "abstain_expected" and c["passed"] for c in checks):
                compliance += 1

    metrics["expected_decision_compliance"] = {
        "compliance_rate": compliance / total
    }

    # ---------------------------------------------------------
    # 7. Per Question Type Metrics
    # ---------------------------------------------------------
    qtypes = defaultdict(list)
    for r in results:
        qt = safe_get(r, "question_info", "question_type")
        qtypes[qt].append(r)

    metrics["per_question_type"] = {}
    for qt, group in qtypes.items():
        group_total = len(group)
        # combined extraction accuracy per question type
        combined_ok = 0
        grounding_nonempty_qt = 0

        for r in group:
            checks = r["checks"]
            if all(
                any(c["name"] == check_name and c["passed"] for c in checks)
                for check_name in field_map.values()
            ):
                combined_ok += 1
            if any(c["name"] == "grounding_nonempty" and c["passed"] for c in checks):
                grounding_nonempty_qt += 1

        metrics["per_question_type"][qt] = {
            "count": group_total,
            "extraction_accuracy": combined_ok / group_total,
            "grounding_nonempty_rate": grounding_nonempty_qt / group_total,
        }

    # ---------------------------------------------------------
    # 8. Per Answer Form Metrics
    # ---------------------------------------------------------
    aforms = defaultdict(list)
    for r in results:
        af = safe_get(r, "question_info", "answer_form")
        aforms[af].append(r)

    metrics["per_answer_form"] = {}
    for af, group in aforms.items():
        group_total = len(group)
        combined_ok = 0
        grounding_nonempty_af = 0

        for r in group:
            checks = r["checks"]
            if all(
                any(c["name"] == check_name and c["passed"] for c in checks)
                for check_name in field_map.values()
            ):
                combined_ok += 1
            if any(c["name"] == "grounding_nonempty" and c["passed"] for c in checks):
                grounding_nonempty_af += 1

        metrics["per_answer_form"][af] = {
            "count": group_total,
            "extraction_accuracy": combined_ok / group_total,
            "grounding_nonempty_rate": grounding_nonempty_af / group_total,
        }

    # ---------------------------------------------------------
    # 9. Ontology Coverage Metrics
    # ---------------------------------------------------------
    seen_classes = set()
    seen_annotations = set()

    for r in results:
        ctx = r.get("entity_context") or {}
        seen_classes.update(ctx.get("types", []))
        # superclasses is a dict: {entity_label: [superclasses]}
        for _, supers in (ctx.get("superclasses") or {}).items():
            seen_classes.update(supers)
        # annotations is a dict: {key: value}
        seen_annotations.update((ctx.get("annotations") or {}).keys())

    metrics["ontology_coverage"] = {
        "unique_classes": len(seen_classes),
        "unique_annotations": len(seen_annotations),
    }

    # ---------------------------------------------------------
    # 10. Latency Metrics (still placeholder)
    # ---------------------------------------------------------
    metrics["latency"] = {
        "available": False,
        "note": "No timestamps found in result structure."
    }

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