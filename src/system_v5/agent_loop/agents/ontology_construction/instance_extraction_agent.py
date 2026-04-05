from .base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class InstanceExtractionAgent(BaseConstructionAgent):
    def __init__(self, backend):
        # Ensure path is correct relative to workspace
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\instance-extraction.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, local_classes: list, local_axioms: list) -> tuple:
        # 1. Prepare Enums
        local_class_names = [c["class"] for c in local_classes]
        valid_class_names = local_class_names if local_class_names else None # If no local classes, we cannot extract instances, so we set to None to trigger safety check later.
        
        if not valid_class_names:
            return [], "Skipped: No valid classes for instantiation."

        # 2. Extract Properties (from linear axioms)
        all_props = set()
        for ax in local_axioms:
            if "property" in ax:
                all_props.add(ax["property"])
            
        property_list = sorted(list(all_props)) or ["relatedTo", "hasValue"]

        # 3. Define Hierarchical Schema
        instance_schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "class": {"enum": valid_class_names},
                            "relations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "property": {"enum": property_list},
                                        "targetId": {"type": "string"}
                                    },
                                    "required": ["property", "targetId"],
                                    "additionalProperties": False
                                }
                            },
                            "attributes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "property": {"enum": property_list}, 
                                        "value": {"type": "string"},
                                        "datatype": {"enum": ["xsd:string", "xsd:int"]}
                                    },
                                    "required": ["property", "value"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["id", "class"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["entities"],
            "additionalProperties": False
        }

        # 4. Prompt Context (Separated for SLM clarity)
        user_msg = f"""

### Valid Classes
{', '.join(local_class_names) if local_class_names else 'None'}

### Valid Properties
{', '.join(property_list)}

### Source Text
{chunk_text}

### Goal
Extract specific instances from the Source Text and map them exclusively to the provided Valid Classes and Valid Properties."""
        
        # 5. Execute
        data, full_prompt = self.generate_with_schema(user_msg, instance_schema)
        
        # 6. Safety Check before Flattening
        if not data:
            return [], full_prompt
            
        # Handle case where model outputs list directly despite schema asking for dict
        if isinstance(data, list):
            data = {"entities": data}

        return self._flatten_to_axioms(data), full_prompt
    
    def _flatten_to_axioms(self, hierarchical_data: dict) -> list:
        # ... (keep your existing flatten logic exactly the same) 
        axioms = []
        if not hierarchical_data or "entities" not in hierarchical_data:
            return []

        for entity in hierarchical_data["entities"]:
            # 1. Declaration
            axioms.append({"type": "DeclareIndividual", "id": entity["id"]})
            # 2. Class Assertion
            axioms.append({"type": "ClassAssertion", "individual": entity["id"], "class": entity["class"]})
            
            # 3. Object Properties (Relations)
            if "relations" in entity:
                for rel in entity["relations"]:
                    axioms.append({
                        "type": "ObjectPropertyAssertion",
                        "subject": entity["id"],
                        "property": rel["property"],
                        "object": rel["targetId"]
                    })
            
            # 4. Data Properties (Attributes)
            if "attributes" in entity:
                for attr in entity["attributes"]:
                    axioms.append({
                        "type": "DataPropertyAssertion",
                        "subject": entity["id"],
                        "property": attr["property"],
                        "value": attr["value"],
                        "datatype": attr.get("datatype", "xsd:string")
                    })
        return axioms