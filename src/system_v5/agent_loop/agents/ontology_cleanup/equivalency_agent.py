from ..ontology_construction.base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class EquivalencyAgent(BaseConstructionAgent):
    def __init__(self, backend):
        # Load the corresponding system prompt
        system_prompt = load_prompt(r"c:\Users\matse\gig\src\system_v5\prompts\system\agents\equivalency-agent.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, cluster: list) -> tuple:
        if not cluster:
            return [], "Skipped: Empty cluster."

        if len(cluster) == 1:
            auto_resolve = [{
                "original_id": cluster[0]["id"],
                "action": "keep_distinct",
                "target_id": None
            }]
            return auto_resolve, "Skipped LLM: Cluster only contained 1 item."

        # Format input for the model
        classes_block = "\n".join([f"- {c['id']}: {c['description']}" for c in cluster])
        
        user_msg = (
            f"### Clustered Classes\n{classes_block}\n\n"
            "### Goal\nIdentify which classes are SEMANTICALLY SYNONYMOUS with each other. "
            "For each class, determine if it is a primary canonical concept ('keep_distinct') "
            "or a functional equivalent of another class in this list ('equivalent'). "
            "If 'equivalent', provide the target_id it is equivalent to."
        )

        original_ids = [c["id"] for c in cluster]

        # JSON schema for OpenVINO GenAI structured generation
        # Using an object with explicit keys mathematically forces the SLM to generate every single class ID
        resolution_schema = {
            "type": "object",
            "properties": {
                "resolved_classes": {
                    "type": "object",
                    "properties": {
                        orig_id: {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["keep_distinct", "equivalent"]},
                                "target_id": {"type": ["string", "null"]}
                            },
                            "required": ["action", "target_id"],
                            "additionalProperties": False
                        } for orig_id in original_ids
                    },
                    "required": original_ids,
                    "additionalProperties": False
                }
            },
            "required": ["resolved_classes"],
            "additionalProperties": False
        }

        # Query the NPU Backend
        raw_extractions, prompt_used = self.generate_with_schema(user_msg, resolution_schema)
        
        # Parse the outputs safely
        resolutions = []
        if raw_extractions and "resolved_classes" in raw_extractions:
            parsed_ids = set()
            for orig_id, res in raw_extractions["resolved_classes"].items():
                resolutions.append({
                    "original_id": orig_id,
                    "action": res.get("action", "keep_distinct"),
                    "target_id": res.get("target_id")
                })
                parsed_ids.add(orig_id)
            
            # Universal Python Safety Net: If the LLM somehow omitted a class, auto-keep it
            for orig_id in original_ids:
                if orig_id not in parsed_ids:
                    resolutions.append({
                        "original_id": orig_id,
                        "action": "keep_distinct",
                        "target_id": None
                    })
            
        return resolutions, prompt_used
