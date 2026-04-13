from ..ontology_construction.base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class AuditorAgent(BaseConstructionAgent):
    def __init__(self, backend):
        # Load the corresponding system prompt
        system_prompt = load_prompt(r"c:\Users\matse\gig\src\system_v5\prompts\system\agents\auditor-agent.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, cluster: list) -> tuple:
        if not cluster:
            return [], "Skipped: Empty cluster."

        # Even for 1 class we might delete it if it's completely hallucinated/broken
        # So we do not skip 1-item clusters strictly in Python.

        # Format input for the model
        classes_block = "\n".join([f"- {c['id']}: {c['description']}" for c in cluster])
        
        user_msg = (
            f"### Candidate Classes\n{classes_block}\n\n"
            "### Goal\nIdentify which of these classes are hallucinated noise, highly-localized artifacts, "
            "or meaningless abstractions that corrupt an overarching OWL 2 Ontology. "
            "For each class, mark it as 'delete' if it must be purged, or 'keep_distinct' if it's a valid TBox Class."
            "Target IDs are not required for this step."
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
                                "action": {"type": "string", "enum": ["keep_distinct", "delete"]}
                            },
                            "required": ["action"],
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
                    "target_id": None
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
