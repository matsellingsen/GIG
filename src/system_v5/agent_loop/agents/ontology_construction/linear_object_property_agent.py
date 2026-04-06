from .base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class LinearObjectPropertyAgent(BaseConstructionAgent):
    def __init__(self, backend):
        system_prompt = load_prompt(r"c:\Users\matse\gig\src\system_v5\prompts\system\agents\linear-object-property.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, local_existing_classes: list) -> tuple:
        if not local_existing_classes: # failsafe-guard
            return [], "Skipped: No local classes to map."
            
        local_class_names = [c["class"] for c in local_existing_classes]

        # Load base axioms (to provide valid property verbs if needed)
        base_axioms = self.seed_axioms.copy()
        valid_properties = [a["type"] for a in base_axioms if a["type"] != "subClassOf"]
        
        object_property_schema = {
            "type": "array",
            "uniqueItems": True,  
            "items": {
                "type": "object",
                "properties": {
                    "domain": {"enum": local_class_names if local_class_names else [""]},
                    "property": {"enum": valid_properties},
                    "range": {"enum": local_class_names if local_class_names else [""]}
                },
                "required": ["domain", "property", "range"],
                "additionalProperties": False
            }
        }

        local_classes_block = "\n".join([f"- {c['class']}: {c['desc']}" for c in local_existing_classes])
        base_axioms_block = "\n".join([f"- {a['type']}: {a['description']}" for a in base_axioms if a['type'] != "subClassOf"])
        
        user_msg = f"""
                    ### Local Classes  
                    {local_classes_block}

                    ### Valid Properties
                    {base_axioms_block}

                    ### Source Text
                    {chunk_text}

                    ### Goal
                    Define the lateral relationships (ObjectProperties) between the provided Local Classes based on the Source Text."""

        return self.generate_with_schema(user_msg, object_property_schema)
