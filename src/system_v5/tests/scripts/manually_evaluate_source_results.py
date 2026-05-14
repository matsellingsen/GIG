import streamlit as st
import json
import random
from pathlib import Path
import os

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
GIG_DATA = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "reports_archive", "GIG_source_dataset_results.json")
)
RAG_DATA = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "reports_archive", "RAG_source_dataset_results.json")
)
REPORT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "interpreted_results", "manually_evaluated_source_results.jsonl")
)
CLASS_OPTIONS = [
    "Correct (Ground Truth)",
    "Correct (Context)",
    "Acceptable",
    "Unprecise",
    "Incorrect",
    "Incorrect Abstain",
    "Correct Abstain",
]

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
def _format_ontology_context(ctx: dict) -> str:
    lines = []

    # Header
    lines.append(f"Entity: {ctx.get('label', 'Unknown')}")
    lines.append("")

    # Types
    types = ctx.get("types", [])
    if types:
        lines.append("Types:")
        for t in types:
            lines.append(f"  - {t}")
        lines.append("")

    # Superclasses
    superclasses = ctx.get("superclasses", {})
    if superclasses:
        lines.append("Superclasses:")
        for t, supers in superclasses.items():
            if supers:
                supers_str = ", ".join(supers)
            else:
                supers_str = "(none)"
            lines.append(f"  - {t}: {supers_str}")
        lines.append("")

    # Class descriptions
    class_desc = ctx.get("class_descriptions", {})
    if class_desc:
        lines.append("Class Descriptions:")
        for cls, info in class_desc.items():
            desc = info.get("description")
            if desc:  # skip null descriptions
                lines.append(f"  - {cls}: {desc}")
        lines.append("")

    return "\n".join(lines)

def _format_mapping(mapping):
    """
    Convert the merged_entity_mapping object into a clean, human-readable
    text block suitable for display in the annotation UI.
    """

    lines = []

    # Types
    types = mapping.get("types", [])
    if types:
        lines.append("Types:")
        for t in types:
            lines.append(f"  - {t}")
        lines.append("")

    # Superclasses
    superclasses = mapping.get("superclasses", [])
    if superclasses:
        lines.append("Superclasses:")
        for s in superclasses:
            lines.append(f"  - {s}")
        lines.append("")

    # Properties
    props = mapping.get("properties", [])
    if props:
        lines.append("Properties:")
        for p in props:
            lines.append(f"  - {p}")
        lines.append("")

    # Property values
    prop_vals = mapping.get("property_values", {})
    if prop_vals:
        lines.append("Property Values:")
        for prop, val in prop_vals.items():
            lines.append(f"  - {prop}: {val}")
        lines.append("")

    # Equivalent classes
    eq = mapping.get("equivalent_classes", [])
    if eq:
        lines.append("Equivalent Classes:")
        for e in eq:
            lines.append(f"  - {e}")
        lines.append("")

    # Annotations
    ann = mapping.get("annotations", [])
    if ann:
        lines.append("Annotations:")
        for a in ann:
            lines.append(f"  - {a}")
        lines.append("")

    # Members
    members = mapping.get("members", [])
    if members:
        lines.append("Members:")
        for m in members:
            lines.append(f"  - {m}")
        lines.append("")

    # Chunk IDs
    chunks = mapping.get("chunk_id", [])
    if chunks:
        lines.append("Chunk IDs:")
        for c in chunks:
            lines.append(f"  - {c}")
        lines.append("")

    return "\n".join(lines)

def load_RAG_data(path):
    """
    Load RAG results from a JSON file and normalize them into the format
    expected by the Streamlit evaluation tool.

    Expected input structure:
    [
      {
        "case": {...},
        "generated_answer": "...",
        "contexts": [...],
        "checks": [...]
      },
      ...
    ]
    """

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized = []

    for item in raw:
        case = item["case"]

        qid = case["id"]
        question = case["question"]
        answer = item["generated_answer"]
        gold_answer = case.get("gold_answer", "")

        # Join contexts into one scrollable block
        context = "\n\n---\n\n".join(item.get("contexts", []))

        normalized.append({
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "systemB": {
                "answer": answer,
                "context": context
            }
        })

    return normalized

def load_GIG_data(path):
    """
    Load GIG results from a JSON file and normalize them into the format
    expected by the Streamlit evaluation tool.
    """

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized = []

    for item in raw:
        case = item["case"]

        qid = case["id"]
        question = case["question"]
        gold_answer = case.get("gold_answer", "")

        answer = item.get("generated_answer", "")
        reasoning = item.get("reasoning", "")

        fetched = item.get("fetched", {})
        if type(fetched) is dict:
            fetched_context_entity = fetched.get("entity_context", "")
            fetched_context_object = fetched.get("object_context", "")

        else:
            fetched_context_entity = ""
            fetched_context_object = ""
        
        entity_mapping = item.get("merged_entity_mapping", {})
        object_mapping = item.get("merged_object_mapping", {})

        if fetched_context_entity:
            entity_context_formatted = _format_ontology_context(fetched_context_entity)
        if fetched_context_object:
            object_context_formatted = _format_ontology_context(fetched_context_object)
        
        if entity_mapping:
            mapped_entity_formatted = _format_mapping(entity_mapping)
        if object_mapping:
            mapped_object_formatted = _format_mapping(object_mapping)

        normalized.append({
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "systemA": {
                "answer": answer,
                "reasoning": reasoning,
                "entity_context": entity_context_formatted if fetched_context_entity else "",
                "object_context": object_context_formatted if fetched_context_object else "",
                "mapped_answer_entity": mapped_entity_formatted if entity_mapping else "",
                "mapped_answer_object": mapped_object_formatted if object_mapping else ""
            }
        })

    return normalized

def merge_datasets(gig_data, rag_data):
    """
    Merge ontology-system data (systemA) and RAG data (systemB)
    into a unified list of question objects ready for annotation.

    Expected output format:
    {
        "id": "...",
        "question": "...",
        "systemA": {...},
        "systemB": {...}
    }
    """

    # Index both datasets by question ID
    A_index = {item["id"]: item for item in gig_data}
    B_index = {item["id"]: item for item in rag_data}

    merged = []

    for qid, A_item in A_index.items():
        if qid not in B_index:
            raise ValueError(f"Missing RAG entry for question ID: {qid}")

        B_item = B_index[qid]

        merged.append({
            "id": qid,
            "question": A_item.get("question"),  # both systems share the same question text
            "gold_answer": A_item.get("gold_answer", ""),
            "systemA": A_item.get("systemA"),
            "systemB": B_item.get("systemB")
        })

    return merged

questions = merge_datasets(load_GIG_data(GIG_DATA), load_RAG_data(RAG_DATA))
# ---------------------------------------------------------
# STATE MANAGEMENT
# ---------------------------------------------------------

if "index" not in st.session_state:
    st.session_state.index = 0

if "system_order" not in st.session_state:
    st.session_state.system_order = {}

# ---------------------------------------------------------
# SAVE ANNOTATION
# ---------------------------------------------------------
def save_annotation(record):
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# ---------------------------------------------------------
# UI LOGIC
# ---------------------------------------------------------

st.title("QA System Evaluation Tool")

if st.session_state.index >= len(questions):
    st.success("All questions have been annotated.")
    st.stop()

q = questions[st.session_state.index]

# Randomize system order per question
if st.session_state.index not in st.session_state.system_order:
    st.session_state.system_order[st.session_state.index] = random.choice(["systemA", "systemB"])

current_system = st.session_state.system_order[st.session_state.index]

st.header(f"Question {st.session_state.index + 1}/{len(questions)}")
st.write(q["question"])

# ---------------------------------------------------------
# DISPLAY SYSTEM A (your ontology system)
# ---------------------------------------------------------

if current_system == "systemA":
    st.subheader("System Output")

    st.markdown("### Final Answer")
    st.text_area("", value=q["systemA"]["answer"])

    st.markdown("### Reasoning")
    st.text_area("", value=q["systemA"]["reasoning"], height=200)

    st.markdown("### Entity Context")
    st.code(q["systemA"]["entity_context"])

    st.markdown("### Object Context")
    st.code(q["systemA"]["object_context"])

    st.markdown("### Mapped Entity Answer")
    st.code(q["systemA"]["mapped_answer_entity"])

    st.markdown("### Mapped Object Answer")
    st.code(q["systemA"]["mapped_answer_object"])

    

    

    chosen = st.radio("Classification", CLASS_OPTIONS, key=f"classA_{st.session_state.index}")

    if st.button("Save & Next"):
        save_annotation({
            "question_id": q["id"],
            "system": "systemA",
            "classification": chosen,
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "answer": q["systemA"]["answer"],
            "reasoning": q["systemA"]["reasoning"],
            "entity_context": q["systemA"]["entity_context"],
            "object_context": q["systemA"]["object_context"],
            "mapped_answer_entity": q["systemA"]["mapped_answer_entity"],
            "mapped_answer_object": q["systemA"]["mapped_answer_object"],

        })
        st.session_state.index += 1
        st.rerun()

# ---------------------------------------------------------
# DISPLAY SYSTEM B (RAG)
# ---------------------------------------------------------

else:
    st.subheader("System Output")

    st.markdown("### Final Answer")
    st.text_area("", value=q["systemB"]["answer"], height=200)

    st.markdown("### Retrieved Context")
    st.code(q["systemB"]["context"])


    chosen = st.radio("Classification", CLASS_OPTIONS, key=f"classB_{st.session_state.index}")

    if st.button("Save & Next"):
        save_annotation({
            "question_id": q["id"],
            "system": "systemB",
            "classification": chosen,
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "answer": q["systemB"]["answer"],
            "context": q["systemB"]["context"],
        })
        st.session_state.index += 1
        st.rerun()
