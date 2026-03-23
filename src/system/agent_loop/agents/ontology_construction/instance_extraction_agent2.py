from .base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class InstanceExtractionAgent2(BaseConstructionAgent):
    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system\\prompts\\system\\agents_v3\\instance-extraction2.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, local_classes: list, local_axioms: list) -> list:
        # 1. Merge Classes (Seed + Local) for Type Constraints
        valid_class_names = list(self.seed_classes.keys()) + [c["class"] for c in local_classes]
        valid_class_names = list(set(valid_class_names))
        
        if not valid_class_names:
            return []

        # 2. Extract Properties from Axioms
        found_properties = set()
        for ax in local_axioms:
            if "property" in ax:
                found_properties.add(ax["property"])
        property_names = list(found_properties) or ["relatedTo"]

        # 3. Define Hierarchical Schema
        # This structure forces "One Entity -> One Class" physically
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
                                        "property": {"enum": property_names},
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
                                        "property": {"enum": property_names},
                                        "value": {"type": "string"},
                                        "datatype": {"enum": ["xsd:string", "xsd:int", "xsd:float", "xsd:boolean", "xsd:date"]}
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

        # 4. Prompt Context
        schema_summary = f"Valid Classes: {', '.join(valid_class_names)}\nValid Properties: {', '.join(property_names)}"
        
        user_msg = f"""
### Schema
{schema_summary}

### Text
{chunk_text}

### Instruction
Extract unique entities. For each entity, assign ONE class and list its relationships."""

        # 5. Extract and Flatten
        result_obj = self.generate_with_schema(user_msg, instance_schema)
        
        # Determine if result_obj is the dict or [dict, prompt] tuple
        # BaseConstructionAgent returns (json_data, prompt)
        if isinstance(result_obj, tuple):
             data = result_obj[0]
        else:
             data = result_obj

        return self._flatten_to_axioms(data)

    def _flatten_to_axioms(self, hierarchical_data: dict) -> list:
        """Converts the hierarchical JSON back to flat OWL axioms."""
        axioms = []
        if not hierarchical_data or "entities" not in hierarchical_data:
            return []

        for entity in hierarchical_data["entities"]:
            # 1. Declaration
            axioms.append({"type": "DeclareIndividual", "id": entity["id"]})
            # 2. Class Assertion
            axioms.append({"type": "ClassAssertion", "individual": entity["id"], "class": entity["class"]})
            
            # 3. Object Properties
            if "relations" in entity:
                for rel in entity["relations"]:
                    axioms.append({
                        "type": "ObjectPropertyAssertion",
                        "subject": entity["id"],
                        "property": rel["property"],
                        "object": rel["targetId"]
                    })
            
            # 4. Data Properties
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