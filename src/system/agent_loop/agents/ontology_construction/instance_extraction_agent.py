from .base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class InstanceExtractionAgent(BaseConstructionAgent):
    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system\\prompts\\system\\agents_v3\\instance-extraction.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, local_classes: list, local_axioms: list) -> list:
        # 1. Merge Classes (Seed + Local) for Type Constraints
        # Just getting the names is enough for the Enum
        valid_class_names = list(self.seed_classes.keys()) + [c["class"] for c in local_classes]
        # De-duplicate
        valid_class_names = list(set(valid_class_names))
        
        if not valid_class_names:
            return []

        # 2. Extract Properties from Axioms
        found_properties = set()
        for ax in local_axioms:
            if "property" in ax:
                found_properties.add(ax["property"])
        
        property_names = list(found_properties)

        # 3. Define Schema
        instance_schema = {
            "type": "array",
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "DeclareIndividual"},
                            "id": {"type": "string"}
                        },
                        "required": ["type", "id"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "ClassAssertion"},
                            "individual": {"type": "string"},
                            "class": {"enum": valid_class_names}
                        },
                        "required": ["type", "individual", "class"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "ObjectPropertyAssertion"},
                            "subject": {"type": "string"},
                            "property": {"enum": property_names}, 
                            "object": {"type": "string"}
                        },
                        "required": ["type", "subject", "property", "object"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "DataPropertyAssertion"},
                            "subject": {"type": "string"},
                            "property": {"enum": property_names},
                            "value": {"type": "string"},
                            "datatype": {"enum": ["xsd:string", "xsd:integer", "xsd:decimal", "xsd:boolean"]}
                        },
                        "required": ["type", "subject", "property", "value", "datatype"],
                        "additionalProperties": False
                    }
                ]
            }
        }

        # 4. Context for Prompt
        schema_summary = f"Valid Classes: {', '.join(valid_class_names)}\nValid Properties: {', '.join(property_names)}"
        
        user_msg = f"""
                    ### Schema
                    {schema_summary}

                    ### Source Text
                    {chunk_text}
|
                    ### Instruction
                    Extract specific instances and facts from the Source Text that fit the schema."""

        return self.generate_with_schema(user_msg, instance_schema)