from .base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class InstanceDataPropertyAgent(BaseConstructionAgent):
    def __init__(self, backend):
        system_prompt = load_prompt(r"c:\Users\matse\gig\src\system_v5\prompts\system\agents\instance-data-property.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, declared_individuals: list, data_axioms: list) -> tuple:
        if not declared_individuals or not data_axioms:
            return [], "Skipped: Missing individuals or data axioms."

        # Extract strict enums to prevent hallucination
        individual_ids = list(set([ind["id"] for ind in declared_individuals if "id" in ind]))
        # We only look at data axioms (where datatype exists)
        valid_data_props = list(set([ax["property"] for ax in data_axioms if "datatype" in ax]))

        if not individual_ids or not valid_data_props:
             return [], "Skipped: No valid individuals or data properties to link."

        data_property_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"enum": individual_ids},
                    "property": {"enum": valid_data_props},
                    "value": {"type": "string"},
                    "datatype": {"enum": ["xsd:string", "xsd:integer", "xsd:decimal", "xsd:boolean", "xsd:date"]}
                },
                "required": ["subject", "property", "value", "datatype"],
                "additionalProperties": False
            }
        }

        individuals_block = "\n".join([f"- {ind['id']} (Class: {ind.get('class', 'Unknown')})" for ind in declared_individuals])
        axioms_block = "\n".join([f"- {ax['domain']} -> {ax['property']} -> {ax['datatype']}" for ax in data_axioms if "datatype" in ax])
        
        user_msg = f"""
                    ### Declared Individuals
                    {individuals_block}

                    ### Valid Data Axioms
                    {axioms_block}

                    ### Source Text
                    {chunk_text}

                    ### Goal
                    Extract specific data values and attributes for the given Individuals based on the Original Text and Valid Data Axioms."""

        raw_extractions, prompt_used = self.generate_with_schema(user_msg, data_property_schema)
        
        # Transform into ABox format
        abox_assertions = []
        for ext in (raw_extractions or []):
            if "subject" in ext and "property" in ext and "value" in ext and "datatype" in ext:
                abox_assertions.append({
                    "type": "DataPropertyAssertion",
                    "subject": ext["subject"],
                    "property": ext["property"],
                    "value": ext["value"],
                    "datatype": ext["datatype"]
                })
                
        return abox_assertions, prompt_used
