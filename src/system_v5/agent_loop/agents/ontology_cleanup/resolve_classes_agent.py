from ..ontology_construction.base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class ResolveClassesAgent(BaseConstructionAgent):
    def __init__(self, backend):
        # Load the corresponding system prompt
        system_prompt = load_prompt(r"c:\Users\matse\gig\src\system_v5\prompts\system\agents\resolve-classes.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, cluster: list) -> tuple:
        if not cluster:
            return [], "Skipped: Empty cluster."

        # Optional: Skip 1-item clusters strictly in Python to save NPU time
        if len(cluster) == 1:
            auto_resolve = [{
                "original_id": cluster[0]["id"],
                "canonical_id": cluster[0]["id"],
                "canonical_description": cluster[0]["description"]
            }]
            return auto_resolve, "Skipped LLM: Cluster only contained 1 item."

        # Format input for the model
        classes_block = "\n".join([f"- {c['id']}: {c['description']}" for c in cluster])
        
        user_msg = (
            f"### Clustered Classes\n{classes_block}\n\n"
            "### Goal\nIdentify which classes are exact synonyms and merge them by assigning the same Canonical Class. "
            "Keep distinct concepts as separate Canonical Classes. Map EVERY original class to its designated Canonical Class."
        )

        # JSON schema heavily constrained for the OpenVINO GenAI structured generation
        resolution_schema = {
            "type": "object",
            "properties": {
                "resolved_classes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "original_id": {"type": "string"},
                            "canonical_id": {"type": "string"},
                            "canonical_description": {"type": "string"}
                        },
                        "required": ["original_id", "canonical_id", "canonical_description"],
                        "additionalProperties": False
                    }
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
            resolutions = raw_extractions["resolved_classes"]
            
        return resolutions, prompt_used
