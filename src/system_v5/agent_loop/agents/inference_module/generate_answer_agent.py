from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt
import json


class GenerateAnswerAgent(BaseOntologyAgent):
    """
    Unified Answer‑Generation Agent.
    Uses a single system prompt and a dynamically constructed user message
    containing:
        - atomic question
        - question_type
        - answer_form
        - extracted triplet
        - ontology mapping
        - relevant ontology context
    """

    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\generate-answer.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, question_info, entity_context, object_context) -> dict:
        user_msg = self._build_user_message(question_info, entity_context, object_context)

        answer_schema = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A concise explanation of the reasoning process that led to the answer, referencing the relevant ontology context."},
                "answer": {
                    "type": "string",
                    "description": "The generated answer to the atomic question."}
            },
            "required": ["reasoning", "answer"],
            "additionalProperties": False
        }

        return self.generate_with_schema(user_msg, answer_schema)

    # ---------------------------------------------------------
    # Unified dynamic user message builder
    # ---------------------------------------------------------
    def _build_user_message(self, question_info, entity_context, object_context):

        atomic_question = question_info.get("atomic_question", "unknown")
        question_type = question_info.get("question_type", "unknown")
        answer_form = question_info.get("answer_form", "value")

        extracted_entity = question_info.get("entity", "unknown")
        relation = question_info.get("relation", "unknown")
        obj = question_info.get("object", "unknown")

        resolved_entity = entity_context.get("entity", {}) if isinstance(entity_context, dict) else {}
        resolved_label = resolved_entity.get("label")
        resolved_uri = resolved_entity.get("uri")
        entity_value = resolved_label or resolved_uri or extracted_entity

        if object_context is not None:
            resolved_object = object_context.get("object", {}) if isinstance(object_context, dict) else {}
            resolved_object_label = resolved_object.get("label")
            resolved_object_uri = resolved_object.get("uri")
            object_value = resolved_object_label or resolved_object_uri or obj

        entity_context_str = json.dumps(entity_context, indent=2) if entity_context else "None"
        object_context_str = json.dumps(object_context, indent=2) if object_context else "None"

        rules_str = self._build_interpretation_rules(
            question_type=question_type,
            answer_form=answer_form,
            relation=relation,
            obj=obj,
        )
        user_msg = f"""
        ### Goal
        Generate an ontology‑grounded answer to the atomic question.

        ### Atomic Question
        {atomic_question}

        ### Classification Signals
        - question_type: {question_type}
        - answer_form: {answer_form}

        ### Extracted Triplet
        - entity: {extracted_entity}
        - relation: {relation}
        - object: {obj}

        ### Extracted Entity Alias
        {entity_value}

        ### Ontology information connected to the Entity
        {entity_context_str}
        """

        if object_context is not None:
            user_msg += f"""        
        ### Extracted Object Alias
        {object_value}

        ### Ontology information connected to the Object
        {object_context_str}
        """

        return user_msg

    def _build_interpretation_rules(self, question_type, answer_form, relation, obj):
        relation_value = relation if isinstance(relation, str) else str(relation)
        object_value = obj.get("value") if isinstance(obj, dict) else obj
        object_value = "unknown" if object_value in (None, "null") else object_value

        rules = []
        rules.append(f"- question_type: {question_type}, answer_form: {answer_form}.")
        rules.append(
            "- Normalize labels by lowercasing, removing spaces/underscores/hyphens, "
            "splitting camelCase, and stripping leading 'has'/'is'."
        )

        if question_type == "property":
            if relation_value == "have property":
                rules.append(
                    f"- The target property name is the triplet object: '{object_value}'."
                    "The target property name should correspond to a predicate label in the context (after normalization)."
                )
            else:
                rules.append(
                    f"- The target property name is the triplet relation: '{relation_value}'."
                )
            if answer_form == "assertion":
                rules.append(
                    "- An assertion asks whether the entity has that property."
                )
            else:
                rules.append(
                    "- A value/list asks for the value(s) of that property."
                )

        elif question_type == "membership":
            rules.append(
                "- Membership questions ask about inclusion or part-whole relations."
            )

        elif question_type == "taxonomic":
            if relation_value == "be instance of":
                rules.append("- The target relation is instance-of (class membership).")
            else:
                rules.append(
                    "- The target relation is subclass/superclass membership."
                )

            if answer_form == "assertion":
                rules.append(
                    "- An assertion asks whether the membership relation holds."
                )
            else:
                rules.append("- A value/list asks for the relevant class name(s).")

        elif question_type == "definition":
            pass
            rules.append(
                "- A definition asks what class the entity belongs to and what properties it has."
            )

        else:
            rules.append(
                "- Interpret the triplet to identify the target attribute or relation."
        )

        return "\n".join(rules)
