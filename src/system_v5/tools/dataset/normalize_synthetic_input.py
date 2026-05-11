"""Normalize synthetic labelled input to match system question types and answer forms.

This script reads the synthetic_labelled_input.json file, enforces the allowed
question types and answer forms, and writes a normalized file next to the
original named synthetic_labelled_input_normalized.json.

Usage: python normalize_synthetic_input.py
"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "tests" / "dataset" / "synthetic_labelled_input.json"
OUT_PATH = ROOT / "tests" / "dataset" / "synthetic_labelled_input_normalized.json"

ALLOWED_TYPES = [
    "definition",
    "taxonomic",
    "capability",
    "property",
    "membership",
    "comparative",
    "quantification",
    "unknown",
]

ALLOWED_ANSWER_FORMS = {
    "definition": ["value"],
    "taxonomic": ["list", "assertion"],
    "property": ["value", "list", "assertion"],
    "membership": ["list"],
    "capability": ["assertion", "value"],
    "comparative": ["assertion", "value"],
    "quantification": ["value"],
    "unknown": ["assertion", "value", "list"],
}


def normalize_item(item: dict) -> dict:
    # normalize question_type
    qtype = item.get("question_type")
    if isinstance(qtype, str):
        qtype = qtype.strip().lower()
    else:
        qtype = "unknown"
    if qtype not in ALLOWED_TYPES:
        qtype = "unknown"
    item["question_type"] = qtype

    # normalize answer_form
    af = item.get("answer_form")
    if isinstance(af, str):
        af = af.strip().lower()
    else:
        af = None
    allowed = ALLOWED_ANSWER_FORMS.get(qtype, ALLOWED_ANSWER_FORMS["unknown"])
    if af not in allowed:
        # choose first allowed as canonical
        item["answer_form"] = allowed[0]
    else:
        item["answer_form"] = af

    # ensure entity/object structure exists
    ent = item.get("entity") or {}
    if "value" not in ent:
        ent["value"] = None
    item["entity"] = ent

    obj = item.get("object") or {}
    if "value" not in obj:
        obj["value"] = None
    item["object"] = obj

    return item


def normalize(data: dict) -> dict:
    out = {}
    for domain, splits in data.items():
        out[domain] = {}
        for split_name, examples in splits.items():
            new_examples = []
            for ex in examples:
                # operate on a shallow copy to avoid mutating original structure
                exc = dict(ex)
                normalized = normalize_item(exc)
                new_examples.append(normalized)
            out[domain][split_name] = new_examples
    return out


def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Input file not found: {IN_PATH}")
    data = json.loads(IN_PATH.read_text(encoding='utf-8'))
    normalized = normalize(data)
    OUT_PATH.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Wrote normalized dataset to {OUT_PATH}")


if __name__ == "__main__":
    main()
