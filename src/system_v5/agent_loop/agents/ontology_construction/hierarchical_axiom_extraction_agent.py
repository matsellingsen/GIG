from .base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class HierarchicalAxiomExtractionAgent(BaseConstructionAgent):
    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\hierarchical-axiom-extraction.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, local_existing_classes: list) -> tuple:
        if not local_existing_classes: # failsafe-guard: If there are no local classes, we cannot extract hierarchical axioms, so we skip this step.
            return [], "Skipped: No local classes to map."
        
        # 1. Load the base ontology classes
        base_ontology_classes = self.seed_classes.copy() # Start with seeds name->desc
        
        # 2. Prepare Enums
        base_class_names = list(self.seed_classes.keys())
        local_class_names = [c["class"] for c in local_existing_classes]
        
        # Superclass can be anything (Base or Local)
        all_class_names = base_class_names + local_class_names 

        # 2. Define Schema
        hierarchical_axiom_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "SubClassOf"},
                    
                    # HARD RESTRICTION: Only local (new) classes can be subclasses
                    "subclass": {
                        "enum": local_class_names if local_class_names else [""] 
                    },
                    
                    # Superclass can be anything (linking Local to Local, or Local to Base)
                    "superclass": {
                        "enum": all_class_names if all_class_names else [""]
                    }
                },
                "required": ["type", "subclass", "superclass"],
                "additionalProperties": False
            }
        }
        

        # 3. Format Prompt
        base_ontology_classes_block = "\n".join([f"- {k}: {v}" for k, v in base_ontology_classes.items()])
        local_classes_block = "\n".join([f"- {c['class']}: {c['desc']}" for c in local_existing_classes])
        user_msg = f"""
                    ### Base Ontology Classes
                    {base_ontology_classes_block}

                    ### Local Classes  
                    {local_classes_block}

                    ### Source Text
                    {chunk_text}

                    ### Goal
                    Establish hierarchical relationships by identifying superclass-subclass relationships between the Local Classes and the Base Ontology Classes."""

        return self.generate_with_schema(user_msg, hierarchical_axiom_schema)