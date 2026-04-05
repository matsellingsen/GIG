from .base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class LinearAxiomExtractionAgent(BaseConstructionAgent):
    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\linear-axiom-extraction.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, local_existing_classes: list) -> tuple:
        if not local_existing_classes: # failsafe-guard: If there are no local classes, skip to save compute
            return [], "Skipped: No local classes to map."
            
        # 1. Prepare Enums (Strictly Local Classes)
        local_class_names = [c["class"] for c in local_existing_classes]

        # 2. Define Schema (Strictly linear properties only, restricted to local classes)
        linear_axiom_schema = {
            "type": "array",
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {"enum": ["ObjectPropertyDomain", "ObjectPropertyRange", "DataPropertyDomain"]},
                            "property": {"type": "string"},
                            "class": {"enum": local_class_names if local_class_names else [""]}
                        },
                        "required": ["type", "property", "class"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "InverseObjectProperties"},
                            "property": {"type": "string"},
                            "inverseProperty": {"type": "string"}
                        },
                        "required": ["type", "property", "inverseProperty"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "DataPropertyRange"},
                            "property": {"type": "string"},
                            "datatype": {"enum": ["xsd:string", "xsd:integer", "xsd:decimal", "xsd:boolean", "xsd:date"]}
                        },
                        "required": ["type", "property", "datatype"],
                        "additionalProperties": False
                    }
                ]
            }
        }

        # 3. Format Prompt
        local_classes_block = "\n".join([f"- {c['class']}: {c['desc']}" for c in local_existing_classes])
        user_msg = f"""
                    ### Local Classes  
                    {local_classes_block}

                    ### Source Text
                    {chunk_text}

                    ### Goal
                    Define the lateral relationships (ObjectProperties) and attributes (DataProperties) for the provided Local Classes based on the Source Text."""

        return self.generate_with_schema(user_msg, linear_axiom_schema)