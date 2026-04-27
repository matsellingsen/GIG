from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt
import json


class ResolveEntityAgent(BaseOntologyAgent):

    def __init__(self, backend):
        system_prompt = load_prompt(
            "C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\resolve-entity.txt"
        )
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(
        self,
        question_info: dict = None,
        candidates: list = None,
    ) -> tuple:

        # -------------------------
        # Build allowed label set
        # -------------------------
        candidate_labels = []
        if isinstance(candidates, list):
            for c in candidates:
                if isinstance(c, dict):
                    label = c.get("label") or c.get("entity") or c.get("name")
                else:
                    label = str(c)
                if label:
                    candidate_labels.append(label)

        # Deduplicate while preserving order
        seen = set()
        unique_labels = []
        for l in candidate_labels:
            if l not in seen:
                seen.add(l)
                unique_labels.append(l)

        # Always allow "unknown"
        allowed = unique_labels + ["unknown"] if unique_labels else ["unknown"]

        # Schema: model must pick one label + reasoning
        schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},# "maxLength": 300},
                "selected_label": {"type": "string", "enum": allowed}
                
            },
            #"required": ["selected_label"],# "reasoning"],
            "required": ["reasoning", "selected_label"],
            "additionalProperties": False
        }

        # -------------------------
        # Build candidate context section
        # -------------------------
        candidates_str = json.dumps(candidates, indent=2) if candidates else "None"

        # -------------------------
        # Extract question_info context
        # -------------------------
        if question_info:
            extracted_entity = question_info.get("entity", {})
            extracted_relation = question_info.get("relation")
            extracted_object = question_info.get("object", {})
            question_type = question_info.get("question_type")

            extracted_entity_str = json.dumps(extracted_entity, indent=2)
            extracted_object_str = json.dumps(extracted_object, indent=2)

            question_context_str = json.dumps({
                "question_type": question_type,
                "relation": extracted_relation
            }, indent=2)
        else:
            extracted_entity_str = "None"
            extracted_object_str = "None"
            question_context_str = "None"

        # -------------------------
        # Build user message
        # -------------------------
        user_msg = f"""
            ### Goal
            Select the single best matching ontology entity label from the candidate list.
            If none fit, choose "unknown".

            ### Extracted Entity
            {extracted_entity_str}

            ### Question Context
            {question_context_str}

            ### Extracted Object (if any)
            {extracted_object_str}

            Candidate Entities and context:
            {candidates_str}

            """

        # -------------------------
        # Execute model with schema
        # -------------------------
        return self.generate_with_schema(user_msg, schema)