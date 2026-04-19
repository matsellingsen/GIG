from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt

class HierarchicalBaseGroundingAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\hierarchical-base-grounding.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, local_existing_classes: list) -> tuple:
        if not local_existing_classes: 
            return [], "Skipped: No local classes to map."
        
        # 1. Load the base ontology classes
        base_ontology_classes = self.seed_classes.copy() 
        
        # 2. Prepare Enums
        base_class_names = list(self.seed_classes.keys())
        local_class_names = [c["class"] for c in local_existing_classes]
        num_local_classes = len(local_existing_classes)
        
        # 3. Define Schema with strictly enforced limits
        hierarchical_axiom_schema = {
            "type": "array",
            "minItems": num_local_classes,
            "maxItems": num_local_classes,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "SubClassOf"},
                    
                    # HARD RESTRICTION: Only local (new) classes can be subclasses
                    "subclass": {
                        "enum": local_class_names if local_class_names else [""] 
                    },
                    
                    # HARD RESTRICTION: Superclass MUST be a Base Class
                    "superclass": {
                        "enum": base_class_names if base_class_names else [""]
                    }
                },
                "required": ["type", "subclass", "superclass"],
                "additionalProperties": False
            }
        }
        
        # 4. Format Prompt
        base_ontology_classes_block = "\n".join([f"- {k}: {v}" for k, v in base_ontology_classes.items()])
        local_classes_block = "\n".join([f"- {c['class']}: {c['desc']}" for c in local_existing_classes])
        user_msg = f"""
                    ### Base Ontology Classes
                    {base_ontology_classes_block}

                    ### Local Classes  
                    {local_classes_block}

                    ### Goal
                    Identify hierarchical superclass-subclass relationships connecting the Local Classes to the established Base Ontology Classes."""

        return self.generate_with_schema(user_msg, hierarchical_axiom_schema)
