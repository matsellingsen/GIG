from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt
import json
import os

class MapToContextAgent(BaseOntologyAgent):
    def __init__(self, backend):
        self.system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\""inference_module\\map-to-context.txt")
        super().__init__(backend=backend, system_prompt=None)

    def run(self, answer, full_context):

        # Build user message
        user_msg = self._build_user_message(answer, full_context)

        # Build schema
        mapping_schema = self._build_schema(full_context)

        # Generate structured mapping
        return self.generate_with_schema(user_msg, mapping_schema)

    # ---------------------------------------------------------
    # Build user message
    # ---------------------------------------------------------
    def _build_user_message(self, answer, full_context):
        context_str = json.dumps(full_context, indent=2)

        return f"""
        ### Task
        Map the answer back to the ontology context.

        ### Answer
        {answer}

        ### Full Ontology Context
        {context_str}
        """

    # ---------------------------------------------------------
    # Build grammar-safe JSON schema (Option A)
    # ---------------------------------------------------------
    def _build_schema(self, full_context):

        def array_field(allowed_values):
            if not allowed_values:
                allowed_values = ["none"]

            # Normalize to strings and sort for deterministic enums
            normalized = sorted({str(v) for v in allowed_values})
            return {
                "type": "array",
                "items": {"enum": normalized},
                "uniqueItems": True
            }

        schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }

        # -----------------------------
        # 1. types
        # -----------------------------
        if "types" in full_context:
            schema["properties"]["types_reasoning"] = {"type": "string"}
            schema["properties"]["types"] = array_field(full_context["types"])
            schema["required"].extend(["types_reasoning", "types"])

        # -----------------------------
        # 2. superclasses
        # -----------------------------
        if "superclasses" in full_context:
            allowed = list(full_context["superclasses"].keys())
            schema["properties"]["superclasses"] = array_field(allowed)
            #schema["properties"]["superclasses_justification"] = {"type": "string"}
            schema["required"].extend(["superclasses_reasoning", "superclasses"])

        # -----------------------------
        # 3. equivalent_classes
        # -----------------------------
        if "equivalent_classes" in full_context:
            schema["properties"]["equivalent_classes"] = array_field(
                full_context["equivalent_classes"]
            )
            schema["properties"]["equivalent_classes_justification"] = {"type": "string"}
            schema["required"].extend(["equivalent_classes_reasoning", "equivalent_classes"])

        # -----------------------------
        # 4. properties_by_type → property names only
        # -----------------------------
        if "properties_by_type" in full_context:
            allowed_props = set()
            for t, tdata in full_context["properties_by_type"].items():
                for item in tdata["outgoing_object_properties"]:
                    allowed_props.add(item["property"])
                for item in tdata["outgoing_data_properties"]:
                    allowed_props.add(item["property"])
                for item in tdata["incoming_object_properties"]:
                    allowed_props.add(item["property"])
                for item in tdata["incoming_data_properties"]:
                    allowed_props.add(item["property"])

            allowed_props = sorted(allowed_props)

            schema["properties"]["properties"] = array_field(allowed_props)
            schema["properties"]["properties_reasoning"] = {"type": "string"}
            schema["required"].extend(["properties_reasoning", "properties"])

        # -----------------------------
        # 5. class_descriptions
        # -----------------------------
        if "class_descriptions" in full_context:
            allowed = list(full_context["class_descriptions"].keys())
            schema["properties"]["class_descriptions"] = array_field(allowed)
            schema["properties"]["class_descriptions_reasoning"] = {"type": "string"}
            schema["required"].extend(["class_descriptions_reasoning", "class_descriptions"])

        # -----------------------------
        # 6. object_property_descriptions
        # -----------------------------
        if "object_property_descriptions" in full_context:
            allowed = list(full_context["object_property_descriptions"].keys())
            schema["properties"]["object_property_descriptions"] = array_field(allowed)
            schema["properties"]["object_property_descriptions_reasoning"] = {"type": "string"}
            schema["required"].extend([
                "object_property_descriptions_reasoning",
                "object_property_descriptions"
            ])

        return schema
