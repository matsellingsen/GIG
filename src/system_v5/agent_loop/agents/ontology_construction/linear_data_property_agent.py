from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt

class LinearDataPropertyAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt(r"c:\Users\matse\gig\src\system_v5\prompts\system\agents\linear-data-property.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, local_existing_classes: list) -> tuple:
        if not local_existing_classes: # failsafe-guard
            return [], "Skipped: No local classes to map."
            
        local_class_names = sorted({str(c.get("class")) for c in local_existing_classes})

        data_property_schema = {
            "type": "array",
            "uniqueItems": True,  
            "items": {
                "type": "object",
                "properties": {
                    "domain": {"enum": local_class_names if local_class_names else [""]},
                    "property": {"type": "string", "pattern": "^[a-z][a-zA-Z0-9]*$"},
                    "datatype": {"enum": ["xsd:string", "xsd:integer", "xsd:decimal", "xsd:boolean", "xsd:date"]}
                },
                "required": ["domain", "property", "datatype"],
                "additionalProperties": False
            }
        }

        local_classes_block = "\n".join([f"- {c['class']}: {c['desc']}" for c in local_existing_classes])
        
        user_msg = f"""
                    ### Local Classes  
                    {local_classes_block}

                    ### Source Text
                    {chunk_text}

                    ### Goal
                    Define the attributes or data values (DataProperties) for the provided Local Classes based on the Source Text."""

        return self.generate_with_schema(user_msg, data_property_schema)
