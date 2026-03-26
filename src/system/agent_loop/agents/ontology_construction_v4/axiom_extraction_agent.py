from .base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class AxiomExtractionAgent(BaseConstructionAgent):
    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system\\prompts\\system\\agents_v3\\axiom-extraction.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, local_existing_classes: list) -> list:
        # 1. Merge Local + Seed Classes for the Enum
        # This ensures we can always link to 'Person' or 'Event' even if not explicitly found in this chunk.
        
        # Create a dictionary to merge by name and avoid duplicates
        merged_classes = self.seed_classes.copy() # Start with seeds name->desc
        
        # Update with local findings (local findings take precedence description-wise if needed, or we keep seeds)
        for c in local_existing_classes:
            merged_classes[c["class"]] = c["desc"]
            
        class_names = list(merged_classes.keys())

        # 2. Define Schema
        axiom_schema = {
            "type": "array",
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "SubClassOf"},
                            "subclass": {"enum": class_names},
                            "superclass": {"enum": class_names}
                        },
                        "required": ["type", "subclass", "superclass"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "EquivalentClasses"},
                            "classes": {
                                "type": "array",
                                "items": {"enum": class_names},
                                "minItems": 2,
                                "maxItems": 2
                            }
                        },
                        "required": ["type", "classes"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "DisjointClasses"},
                            "classes": {
                                "type": "array",
                                "items": {"enum": class_names},
                                "minItems": 2,
                                "maxItems": 2
                            }
                        },
                        "required": ["type", "classes"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"enum": ["ObjectPropertyDomain", "ObjectPropertyRange", "DataPropertyDomain"]},
                            "property": {"type": "string"},
                            "class": {"enum": class_names}
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
        classes_block = "\n".join([f"- {k}: {v}" for k, v in merged_classes.items()])
        user_msg = f"""
                    ### Classes
                    {classes_block}

                    ### Text
                    {chunk_text}

                    ### Instruction
                    Identify logical relationships (axioms) and properties from the text for the classes above."""

        return self.generate_with_schema(user_msg, axiom_schema)