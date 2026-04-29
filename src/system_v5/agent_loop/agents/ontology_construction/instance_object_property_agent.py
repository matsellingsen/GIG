from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt

class InstanceObjectPropertyAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt(r"c:\Users\matse\gig\src\system_v5\prompts\system\agents\instance-object-property.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, declared_individuals: list, object_axioms: list) -> tuple:
        if not declared_individuals or not object_axioms:
            return [], "Skipped: Missing individuals or axioms."

        # Extract strict enums to prevent hallucination (deterministic)
        individual_ids = sorted({str(ind["id"]) for ind in declared_individuals if "id" in ind})
        # We only look at object axioms (where range is a Class, not a datatype)
        valid_obj_props = sorted({str(ax["property"]) for ax in object_axioms if "datatype" not in ax})

        if not individual_ids or not valid_obj_props:
             return [], "Skipped: No valid individuals or object properties to link."

        object_property_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"enum": individual_ids},
                    "property": {"enum": valid_obj_props},
                    "object": {"enum": individual_ids}
                },
                "required": ["subject", "property", "object"],
                "additionalProperties": False
            }
        }

        individuals_block = "\n".join([f"- {ind['id']} (Class: {ind.get('class', 'Unknown')})" for ind in declared_individuals])
        axioms_block = "\n".join([f"- {ax['domain']} -> {ax['property']} -> {ax['range']}" for ax in object_axioms if "datatype" not in ax])
        
        user_msg = f"""
                    ### Declared Individuals
                    {individuals_block}

                    ### Valid Object Axioms
                    {axioms_block}

                    ### Source Text
                    {chunk_text}

                    ### Goal
                    Identify relationships between the provided Individuals based on the Original Text and Valid Object Axioms."""

        raw_extractions, prompt_used = self.generate_with_schema(user_msg, object_property_schema)
        
        # Transform into ABox format
        abox_assertions = []
        for ext in (raw_extractions or []):
            if "subject" in ext and "property" in ext and "object" in ext:
                abox_assertions.append({
                    "type": "ObjectPropertyAssertion",
                    "subject": ext["subject"],
                    "property": ext["property"],
                    "object": ext["object"]
                })
                
        return abox_assertions, prompt_used
