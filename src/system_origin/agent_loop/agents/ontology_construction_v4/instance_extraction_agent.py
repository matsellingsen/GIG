from .base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class InstanceExtractionAgent(BaseConstructionAgent):
    def __init__(self, backend):
        # Ensure path is correct relative to workspace
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system\\prompts\\system\\agents_v4\\instance-extraction2.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, local_classes: list, local_axioms: list) -> list:
        # 1. Merge Classes
        valid_class_names = list(self.seed_classes.keys()) + [c["class"] for c in local_classes]
        valid_class_names = sorted(list(set(valid_class_names))) # Sort for deterministic schema
        
        if not valid_class_names:
            return []

        # 2. Split Properties into Object/Data (Heuristic attempt)
        # In a perfect world, local_axioms tells us the type.
        # If not, we allow all properties in both slots, but having the split is cleaner if possible.
        object_props = set()
        data_props = set()
        
        # We assume local_axioms might have a "type" field like "ObjectPropertyAssertion" implies existence? 
        # Actually usually axiom extractor just outputs {"superclass":...} or {"domain":...}.
        # For now, we collect ALL properties found in any slot.
        all_props = set()
        for ax in local_axioms:
            # Check widely for property strings
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
                                        # You can restrict datatype properties here if you used a separate list
                                        "property": {"enum": property_list}, 
                                        "value": {"type": "string"},
                                        # Simplified datatypes to avoid confusion
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

        # 4. Prompt Context
        schema_summary = f"Valid Classes: {', '.join(valid_class_names)}\nValid Properties: {', '.join(property_list)}"
        
        user_msg = f"""
### Schema Context
{schema_summary}

### Source Text
{chunk_text}

### Instruction
Extract unique entities. For each entity, assign ONE class and list its relationships."""
        
        # 5. Execute
        data, full_prompt = self.generate_with_schema(user_msg, instance_schema)
        
        # 6. Safety Check before Flattening
        if not data:
            return [], full_prompt
            
        # Handle case where model outputs list directly despite schema asking for dict
        if isinstance(data, list):
            # Wrap it so flatten logic works
            data = {"entities": data}

        return self._flatten_to_axioms(data), full_prompt
    
    def _flatten_to_axioms(self, hierarchical_data: dict) -> list:
        """
        Converts the hierarchical JSON (Entities -> Relations) back to flat OWL axioms (Triples).
        """
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