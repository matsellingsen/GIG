import json
import re
from collections import defaultdict

with open(r"C:\Users\matse\gig\src\system_v5\intermediate_results\phi-npu-openvino_ontology_20260406_232546_classResolved.json", 'r', encoding='utf-8') as f:
    ontology = json.load(f)

exact_clusters = defaultdict(set)
for inst in ontology.get("ontology_abox", []):
    if inst.get("type") == "ClassAssertion":
        iid = inst.get("individual", "")
        # exact match is norm + class
        norm = re.sub(r'[\W_]+', '', iid).lower()
        exact_clusters[(norm, inst.get("class"))].add(iid)

for (norm, cls), ids in exact_clusters.items():
    if len(ids) > 1:
        print(f"Norm: {norm}, Class: {cls}, IDs: {ids}")
