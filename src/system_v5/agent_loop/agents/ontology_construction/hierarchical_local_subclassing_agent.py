from .base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class HierarchicalLocalSubclassingAgent(BaseConstructionAgent):
    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\hierarchical-local-subclassing.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, local_existing_classes: list) -> tuple:
        if len(local_existing_classes) < 2:
            return [], "Skipped: Not enough local classes to form lateral hierarchy."
        
        # 1. Prepare Enums
        local_class_names = [c["class"] for c in local_existing_classes]
        
        # 2. Define Schema with strictly enforced limits
        hierarchical_axiom_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "SubClassOf"},
                    
                    # HARD RESTRICTION: Only local classes can be subclasses
                    "subclass": {
                        "enum": local_class_names 
                    },
                    
                    # HARD RESTRICTION: Superclass MUST also be a Local Class
                    "superclass": {
                        "enum": local_class_names
                    }
                },
                "required": ["type", "subclass", "superclass"],
                "additionalProperties": False
            }
        }
        
        # 3. Format Prompt
        local_classes_block = "\n".join([f"- {c['class']}: {c['desc']}" for c in local_existing_classes])
        user_msg = f"""
                    ### Local Classes  
                    {local_classes_block}

                    ### Goal
                    Identify hierarchical superclass-subclass relationships strictly BETWEEN the Local Classes."""

        return self.generate_with_schema(user_msg, hierarchical_axiom_schema)
