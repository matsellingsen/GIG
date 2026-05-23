import json
from collections import Counter, defaultdict
import os

from sympy import re

RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "reports_archive", "ontology_grounded_standardized_results.json")
)

OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "interpreted_results", "manual_ontology_grounded_results_complete_interpreted.json")
)


def load_results():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)

def _category_from_case_id(case_id: str):
    if case_id.startswith("present_def_"):
        return "definition"
    if case_id.startswith("present_cap_"):
        return "capability"
    if case_id.startswith("present_prop_"):
        return "property"
    if case_id.startswith("present_mem_"):
        return "membership"
    if case_id.startswith("present_cmp_"):
        return "comparative"
    if case_id.startswith("present_qty_"):
        return "quantification"
    if case_id.startswith("present_tax_"):
        return "taxonomic"
    return None
def _flatten_grounding_to_tokens(grounding) -> set:
    tokens = set()

    if not grounding:
        return tokens

    if isinstance(grounding, dict):
        for value in grounding.values():
            tokens |= _flatten_grounding_to_tokens(value)
    elif isinstance(grounding, list):
        for item in grounding:
            tokens |= _flatten_grounding_to_tokens(item)
    else:
        tokens.add(str(grounding).lower())

    return tokens

def _extract_gold_grounding(result: dict) -> dict:
    """Extract gold grounding from checks array (grounding_preferred check)."""
    checks = result.get("checks", [])
    for check in checks:
        if check.get("name") == "grounding_preferred":
            return check.get("expected", {})
    return {}

def _extract_gold_side_grounding(gold_grounding: dict, side: str) -> dict:
    if not isinstance(gold_grounding, dict):
        return {}

    if side in gold_grounding and isinstance(gold_grounding[side], dict):
        return gold_grounding[side]

    return gold_grounding

def _is_predicted_subset_of_context(predicted, context) -> bool:
    if not predicted or not context:
        return False
    pred_tokens = _flatten_grounding_to_tokens(predicted)
    ctx_tokens = _flatten_grounding_to_tokens(context)
    return bool(pred_tokens) and pred_tokens.issubset(ctx_tokens)

def _calculate_grounding_overlap(gold_grounding: dict, predicted_grounding: dict) -> bool:
    if not gold_grounding or not predicted_grounding:
        return False

    gold_tokens = _flatten_grounding_to_tokens(gold_grounding)
    predicted_tokens = _flatten_grounding_to_tokens(predicted_grounding)

    return len(gold_tokens & predicted_tokens) > 0

def _check_entity_resolved(type: str, result: dict) -> bool:
    """Check if entity was successfully resolved to an ontology node."""
    fetched = result.get("fetched", {})
    
    # Check for error strings
    if isinstance(fetched, str):
        if type == "entity":
            return fetched != "noPrimaryEntityFound"
        elif type == "object":
            return fetched != "noComparativeObjectFound"
    
    # Check if resolved_entity exists and is valid
    resolved_entity = fetched.get(f"resolved_{type}")
    return resolved_entity is not None and resolved_entity != ""

def _category_metrics(cat, cat_results):
    """ Compute metrics for a given category."""
    # 1. divide into answerable vs unanswerable
    answerable = [r for r in cat_results if "_unanswerable" not in r["case"]["id"]]#safe_get(r, "case", "name") == "answer_not_abstain"]
    unanswerable = [r for r in cat_results if "_unanswerable" in r["case"]["id"]]
    answerable_metrics = {
        "total": len(answerable),
        "object_expected": len(answerable) if cat == "comparative" else 0,
        "correct_question_type": 0,
        "correct_answer_form": 0,
        "correct_entity": 0,
        "correct_relation": 0,
        "correct_object": 0,
        "entity_resolved": 0,
        "correct_entity_resolution": 0,  # Grounding overlap indicates correct node
        "object_resolved": 0,
        "correct_object_resolution": 0, # Grounding overlap indicates correct node
        "preferred_grounding_match": 0,  # Direct check if preferred grounding matches expected
        "acceptable_grounding": 0, # Acceptable if it matches expected or shares tokens with expected
        "unacceptable_grounding": 0, # Unacceptable if it doesn't match expected and shares no tokens and resolved entity is not correct
        "no_mapping_performed": 0, # Unresolved entity/object skipped mapping
    }
    
    unanswerable_metrics = {
        "total": len(unanswerable),
        "object_expected": len(unanswerable) if cat == "comparative" else 0,
        "correctly_abstained": 0,
        "correct_question_type": 0,
        "correct_answer_form": 0,
        "correct_entity": 0,
        "correct_relation": 0,
        "correct_object": 0,
        "entity_resolved": 0,
        "correct_entity_resolution": 0,
        "object_resolved": 0,
        "correct_object_resolution": 0,
        "preferred_grounding_match": 0,
        "acceptable_grounding": 0,
        "unacceptable_grounding": 0,
        "no_mapping_performed": 0,
    }
    # 2. For anwerable cases, main question is "Does the system ground the answer to the ontology?"
    for ans_case in answerable:
        checks = ans_case["checks"]

        # ------------------------------------------------
        # Extraction correctness checks (question type, answer form, entity, relation, object)
        # ------------------------------------------------
        #check if correct question type is extracted
        correct_question_type = any(c["name"] == "correct_question_type" and c["passed"] for c in checks)
        if correct_question_type:
            answerable_metrics["correct_question_type"] += 1

        # check if correct answer form is extracted
        correct_answer_form = any(c["name"] == "correct_answer_form" and c["passed"] for c in checks)
        if correct_answer_form:
            answerable_metrics["correct_answer_form"] += 1

        # check if correct entity is extracted
        correct_entity = any(c["name"] == "correct_extracted_entity" and c["passed"] for c in checks)
        if correct_entity:
            answerable_metrics["correct_entity"] += 1

        # check if correct relation is extracted
        correct_relation = any(c["name"] == "correct_relation_extraction" and c["passed"] for c in checks)
        if correct_relation:
            answerable_metrics["correct_relation"] += 1

        # check if correct object is extracted
        correct_object = any(c["name"] == "correct_object_extraction" and c["passed"] for c in checks)
        if correct_object:
            answerable_metrics["correct_object"] += 1
        
        # check if entity is resolved
        is_resolved_entity = _check_entity_resolved("entity", ans_case)
        if is_resolved_entity:
            answerable_metrics["entity_resolved"] += 1
        else:
            answerable_metrics["no_mapping_performed"] += 1
            continue
        
        # Check if resolved entity is grounded to correct entity in ontology
        correct_resolved_entity = any(c["name"] == "correct_entity_resolution" and c["passed"] for c in checks)
        if correct_resolved_entity:
            answerable_metrics["correct_entity_resolution"] += 1

        # ------------------------------------------------
        # Grounding checks
        # ------------------------------------------------
            
        # Check if resolved to correct node via grounding overlap
        #gold_grounding = _extract_gold_grounding(ans_case)
        predicted_entity_grounding = ans_case.get("mapped_entity_merged", {})
        predicted_object_grounding = ans_case.get("mapped_object_merged", {})


        print("CATEGORY:", cat)
        pred_entity_ctx = ans_case.get("entity_context", {})
        pred_object_ctx = ans_case.get("object_context", {})

        if cat == "comparative":
            print(f"Case {ans_case['case']['id']}")
            # add object resolution checks
            is_resolved_object = _check_entity_resolved("object", ans_case) #reuse entity resolution check since it checks for both 
            if is_resolved_object:
                answerable_metrics["object_resolved"] += 1
            else:
                print(f"Case {ans_case['case']['id']} failed object resolution, skipping grounding checks for this case.")
                answerable_metrics["no_mapping_performed"] += 1
                continue
            
            correct_resolved_object = any(c["name"] == "correct_object_resolution" and c["passed"] for c in checks)
            if correct_resolved_object:
                answerable_metrics["correct_object_resolution"] += 1

            # Check if preferred grounding matches expected (direct match check)
            preferred_grounding = any(c["name"] == "grounding_preferred" and c["passed"] for c in ans_case["checks"])
            if preferred_grounding:
                answerable_metrics["preferred_grounding_match"] += 1
            
            # add grounding checks for both entity and object
            entity_ok = _is_predicted_subset_of_context(predicted_entity_grounding, pred_entity_ctx)
            object_ok = _is_predicted_subset_of_context(predicted_object_grounding, pred_object_ctx)

            predicted_subset_ok = (entity_ok and correct_resolved_entity) and (object_ok and correct_resolved_object)
            acceptable_grounding = preferred_grounding or predicted_subset_ok
            unacceptable_grounding = not acceptable_grounding

            if acceptable_grounding:
                answerable_metrics["acceptable_grounding"] += 1
            if unacceptable_grounding:
                print(f"Case {ans_case['case']['id']} has unacceptable grounding.")
                answerable_metrics["unacceptable_grounding"] += 1

        else:
            # Check if preferred grounding matches expected (direct match check)
            preferred_grounding = any(c["name"] == "grounding_preferred" and c["passed"] for c in ans_case["checks"])
            if preferred_grounding:
                answerable_metrics["preferred_grounding_match"] += 1

            # Add grounding checks for entity only (since non-comparative questions typically only have entity grounding)            
            entity_ok = _is_predicted_subset_of_context(predicted_entity_grounding, pred_entity_ctx)
            acceptable_grounding = preferred_grounding or (entity_ok and correct_resolved_entity)
            unacceptable_grounding = not acceptable_grounding
            if acceptable_grounding:
                answerable_metrics["acceptable_grounding"] += 1
            if unacceptable_grounding:
                print(f"Case {ans_case['case']['id']} has unacceptable grounding.")
                answerable_metrics["unacceptable_grounding"] += 1
            

    # 3. For unanswerable cases the main question is "Does the system correctly abstain?" and "If not, does it ground to the correct entity or a related entity in the ontology, or does it fail to ground at all?"
    for unans_case in unanswerable:
        checks = unans_case.get("checks", [])


        # extraction correctness
        if any(c.get("name") == "correct_question_type" and c.get("passed") for c in checks):
            unanswerable_metrics["correct_question_type"] += 1
        if any(c.get("name") == "correct_answer_form" and c.get("passed") for c in checks):
            unanswerable_metrics["correct_answer_form"] += 1
        if any(c.get("name") == "correct_extracted_entity" and c.get("passed") for c in checks):
            unanswerable_metrics["correct_entity"] += 1
        if any(c.get("name") == "correct_relation_extraction" and c.get("passed") for c in checks):
            unanswerable_metrics["correct_relation"] += 1
        if any(c.get("name") == "correct_object_extraction" and c.get("passed") for c in checks):
            unanswerable_metrics["correct_object"] += 1

       # abstain correctness

        # resolution and grounding
        is_resolved_entity = _check_entity_resolved("entity", unans_case)
        if is_resolved_entity:
            unanswerable_metrics["entity_resolved"] += 1

        correct_resolved_entity = any(c.get("name") == "correct_entity_resolution" and c.get("passed") for c in checks)
        if correct_resolved_entity:
            unanswerable_metrics["correct_entity_resolution"] += 1

        
        # abstain correctness
        abstain_ok = any(c.get("name") == "abstain_expected" and c.get("passed") for c in checks)
        if abstain_ok:
            print(f"Case {unans_case['case']['id']} correctly abstained.")
            unanswerable_metrics["correctly_abstained"] += 1
            unanswerable_metrics["no_mapping_performed"] += 1
            continue # if system correctly abstains, we don't need to evaluate grounding since it shouldn't have made a statement about the entity/ontology.
            #break # if system correctly abstains, we don't need to evaluate grounding since it shouldn't have made a statement about the entity/ontology.
        
        if not abstain_ok and is_resolved_entity:

            # preferred grounding
            preferred = any(c.get("name") == "grounding_preferred" and c.get("passed") for c in checks)
            if preferred:
                unanswerable_metrics["preferred_grounding_match"] += 1

            # acceptable grounding: predicted subset of context
            predicted_entity_grounding = unans_case.get("mapped_entity_merged", {})
            predicted_object_grounding = unans_case.get("mapped_object_merged", {})
            pred_entity_ctx = unans_case.get("entity_context", {})
            pred_object_ctx = unans_case.get("object_context", {})

            # For comparative/unanswerable we treat entity+object, else only entity
            cat = _category_from_case_id(unans_case.get("case", {}).get("id", ""))
            acceptable = False
            if cat == "comparative":
                is_object_resolved = _check_entity_resolved("object", unans_case)
                if is_object_resolved:
                    unanswerable_metrics["object_resolved"] += 1
                    # comparative may have object resolution
                    correct_resolved_object = any(c.get("name") == "correct_object_resolution" and c.get("passed") for c in checks)
                    if correct_resolved_object:
                        unanswerable_metrics["correct_object_resolution"] += 1

                    entity_ok = _is_predicted_subset_of_context(predicted_entity_grounding, pred_entity_ctx)
                    object_ok = _is_predicted_subset_of_context(predicted_object_grounding, pred_object_ctx)
                    # Unanswerable cases often don't output 'correct_entity_resolution' or 'correct_object_resolution' checks.
                    # As long as subsets match the fetched contexts, the grounding itself is acceptable.
                    predicted_subset_ok = entity_ok and object_ok
                    acceptable = preferred or predicted_subset_ok
                else:
                    print(f"Case {unans_case['case']['id']} failed object resolution, skipping grounding checks for this case.")
                    unanswerable_metrics["no_mapping_performed"] += 1
                    continue
            else:
                entity_ok = _is_predicted_subset_of_context(predicted_entity_grounding, pred_entity_ctx)
                acceptable = preferred or entity_ok

            if acceptable:
                unanswerable_metrics["acceptable_grounding"] += 1
            else:
                print(f"Case {unans_case['case']['id']} has unacceptable grounding.")
                unanswerable_metrics["unacceptable_grounding"] += 1
        elif not abstain_ok and not is_resolved_entity:
            unanswerable_metrics["no_mapping_performed"] += 1

    return {
        "answerable": answerable_metrics,
        "unanswerable": unanswerable_metrics,
    }


    # 3. For unanswerable cases the main question is "Does the system correctly abstain?" 
def compute_metrics(results):

    # split results into categories based on question type
    categories = {
        "definition": [],
        "taxonomic": [],
        "comparative": [],
        "property": [],
        "membership": [],
        "quantification": [],
        "capability": [],}
       
    for r in results:
        case_id = r.get("case", {}).get("id", "")
        gold_cat = _category_from_case_id(case_id)
        categories[gold_cat].append(r)
    
    # For each cateogry, compute appropriate metrics
    category_metrics = {}
    for cat, cat_results in categories.items():
        category_metrics[cat] = _category_metrics(cat, cat_results)
    
    # also compute overall metrics across all categories
    overall_metrics = {"answerable": defaultdict(int), "unanswerable": defaultdict(int)}
    for cat, cat_metrics in category_metrics.items():
        #print(f"Metrics for category {cat}: {cat_metrics}")
        answerable_metrics = cat_metrics["answerable"]
        unanswerable_metrics = cat_metrics["unanswerable"] if cat != "definition" else {} # definition doesn't have unanswerable cases
        for metric, value in answerable_metrics.items():
            overall_metrics["answerable"][metric] += value
        for metric, value in unanswerable_metrics.items():
            overall_metrics["unanswerable"][metric] += value

    # Convert counts to percentages with smart bases
    def to_pct(stats):
        if not stats: return {}
        total = stats.get("total", 0)
        if total == 0: return {k: 0.0 for k in stats if k not in ["total", "object_expected"]} | {"total": 0}
        
        res = {"total": total}
        for k, v in stats.items():
            if k in ["total", "object_expected"]: continue
            base = total
            if k == "correct_object_resolution":
                base = stats.get("object_resolved", 0)
            elif k == "correct_entity_resolution":
                base = stats.get("entity_resolved", 0)
            elif k == "object_resolved":
                base = stats.get("object_expected", 0)
            
            res[k] = round((v / base * 100), 2) if base > 0 else 0.0
        return res

    def to_fraction(stats):
        if not stats: return {}
        total = stats.get("total", 0)
        if total == 0: return {k: "0/0" for k in stats if k not in ["total", "object_expected"]} | {"total": 0}
        
        res = {"total": total}
        for k, v in stats.items():
            if k in ["total", "object_expected"]: continue
            base = total
            if k == "correct_object_resolution":
                base = stats.get("object_resolved", 0)
            elif k == "correct_entity_resolution":
                base = stats.get("entity_resolved", 0)
            elif k == "object_resolved":
                base = stats.get("object_expected", 0)
            
            res[k] = f"{v}/{base}"
        return res

    for cat in category_metrics:
        if "answerable" in category_metrics[cat]:
            category_metrics[cat]["answerable"] = to_fraction(category_metrics[cat]["answerable"])
        if "unanswerable" in category_metrics[cat]:
            category_metrics[cat]["unanswerable"] = to_fraction(category_metrics[cat]["unanswerable"])

    overall_pct = {
        "answerable": to_pct(overall_metrics["answerable"]),
        "unanswerable": to_pct(overall_metrics["unanswerable"])
    }
    category_metrics["overall"] = overall_pct

    return category_metrics

def main():
    results = load_results()
    metrics = compute_metrics(results)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()