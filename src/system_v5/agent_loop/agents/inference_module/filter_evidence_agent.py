from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt
import json
import os

class FilterEvidenceAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt(
            "C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\filter-evidence.txt"
        )
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, question_info, all_info):
        
        # Build user message
        user_msg = self._build_user_message(
            question_info=question_info,
            full_context=all_info
        )

        # Build JSON schema dynamically based on which components exist
        evidence_schema = self._build_schema(all_info)

        return self.generate_with_schema(user_msg, evidence_schema)

    # ---------------------------------------------------------
    # Build user message
    # ---------------------------------------------------------
    def _build_user_message(self, question_info, full_context):
        atomic_question = question_info.get("atomic_question", "unknown")
        question_info = {k: v for k, v in question_info.items() if k != "atomic_question"}  # drop atomic question from question info since it's already included separately in the prompt

        qinfo_str = json.dumps(question_info, indent=2)
        context_str = json.dumps(full_context, indent=2)

        return f"""
        ### Task
        Select only the minimal ontology facts required to answer the question.

        ### Atomic Question
        {atomic_question}

        ### Question Information
        {qinfo_str}

        ### Full Ontology Context
        {context_str}
        """

        # ---------------------------------------------------------
    # Build STRICT JSON schema using KEY-SELECTION ONLY
    # ---------------------------------------------------------
    def _build_schema(self, full_context):

        schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }

        def wrap_component(allowed_keys):
            # Normalize and sort allowed keys for deterministic enums
            normalized = sorted({str(k) for k in allowed_keys}) if allowed_keys else ["none"]
            return {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "array",
                        "items": {"enum": normalized},
                        "uniqueItems": True
                    },
                    "justification": {"type": "string"}
                },
                "required": ["value", "justification"],
                "additionalProperties": False
            }

        # -----------------------------------------------------
        # 1. types (list of strings) — safe to use directly
        # -----------------------------------------------------
        if "types" in full_context:
            allowed = full_context["types"]
            schema["properties"]["types"] = wrap_component(allowed)

        # -----------------------------------------------------
        # 2. superclasses (dict) → flatten to keys
        # -----------------------------------------------------
        if "superclasses" in full_context:
            allowed = list(full_context["superclasses"].keys())
            schema["properties"]["superclasses"] = wrap_component(allowed)

        # -----------------------------------------------------
        # 3. equivalent_classes (list of strings)
        # -----------------------------------------------------
        if "equivalent_classes" in full_context:
            allowed = full_context["equivalent_classes"]
            schema["properties"]["equivalent_classes"] = wrap_component(allowed)

        # -----------------------------------------------------
        # 4. properties_by_type (nested dicts) → flatten to keys
        # -----------------------------------------------------
        if "properties_by_type" in full_context:

            allowed_keys = []

            for t, tdata in full_context["properties_by_type"].items():

                # outgoing object properties
                for i, _ in enumerate(tdata["outgoing_object_properties"]):
                    allowed_keys.append(f"{t}.outgoing_object_properties.{i}")

                # outgoing data properties
                for i, _ in enumerate(tdata["outgoing_data_properties"]):
                    allowed_keys.append(f"{t}.outgoing_data_properties.{i}")

                # incoming object properties
                for i, _ in enumerate(tdata["incoming_object_properties"]):
                    allowed_keys.append(f"{t}.incoming_object_properties.{i}")

                # incoming data properties
                for i, _ in enumerate(tdata["incoming_data_properties"]):
                    allowed_keys.append(f"{t}.incoming_data_properties.{i}")

            schema["properties"]["properties_by_type"] = wrap_component(allowed_keys)

        # -----------------------------------------------------
        # 5. annotations (dict) → flatten to keys
        # -----------------------------------------------------
        if "annotations" in full_context:
            allowed = list(full_context["annotations"].keys())
            schema["properties"]["annotations"] = wrap_component(allowed)

        # -----------------------------------------------------
        # 6. class_descriptions (dict of dicts) → flatten to keys
        # -----------------------------------------------------
        if "class_descriptions" in full_context:
            allowed = list(full_context["class_descriptions"].keys())
            schema["properties"]["class_descriptions"] = wrap_component(allowed)

        # -----------------------------------------------------
        # 7. object_property_descriptions (dict of dicts) → keys
        # -----------------------------------------------------
        if "object_property_descriptions" in full_context:
            allowed = list(full_context["object_property_descriptions"].keys())
            schema["properties"]["object_property_descriptions"] = wrap_component(allowed)

        # -----------------------------------------------------
        # 8. provenance (list of strings)
        # -----------------------------------------------------
        if "provenance" in full_context:
            allowed = full_context["provenance"]
            schema["properties"]["provenance"] = wrap_component(allowed)

        return schema

