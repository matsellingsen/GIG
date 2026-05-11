import json
from pathlib import Path

PATH = Path(r"c:\Users\matse\gig\src\system_v5\tests\dataset\synthetic_labelled_input_normalized.json")

RELATION_MAP = {
    "definition": "be",
    "taxonomic": "be subtype of",
    "property": "have property",
    "membership": "have member",
    "capability": "have capability",
    "comparative": "compare",
    "quantification": "count",
    "existential": "exist",
    "unknown": "unknown",
}

NULL_QTYPES = {"definition", "unknown", "quantification"}
NULL_AFORMS = {"list"}
NULL_COMBINATIONS = {("capability", "value")}


def normalize_entry(e):
    qtype = e.get("question_type", "unknown")
    aform = e.get("answer_form", "unknown")

    # enforce canonical relation
    e["relation"] = RELATION_MAP.get(qtype, "unknown")

    # apply null-object rules
    if (qtype in NULL_QTYPES) or (aform in NULL_AFORMS) or ((qtype, aform) in NULL_COMBINATIONS):
        e["object"] = {"value": None, "type": None}
    else:
        obj = e.get("object", {}) or {}
        if "value" not in obj:
            obj["value"] = None
        if "type" not in obj:
            obj["type"] = None
        e["object"] = obj


if not PATH.exists():
    raise SystemExit(f"Input file not found: {PATH}")

with PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

for domain, splits in data.items():
    for split_name, items in splits.items():
        for entry in items:
            normalize_entry(entry)

with PATH.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Updated {PATH}")
