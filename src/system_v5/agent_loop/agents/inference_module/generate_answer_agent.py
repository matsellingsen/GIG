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

    def run(self, question_info, relevant_info):
        user_msg = self._build_user_message(question_info, relevant_info)

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
    def _build_user_message(self, question_info, relevant_info):

        atomic_question = question_info.get("atomic_question", "unknown")
        question_type = question_info.get("question_type", "unknown")
        answer_form = question_info.get("answer_form", "value")

        entity = question_info.get("entity", "unknown")
        relation = question_info.get("relation", "unknown")
        obj = question_info.get("object", "unknown")

        context_str = json.dumps(relevant_info, indent=2) if relevant_info else "None"

        return f"""
        ### Goal
        Generate an ontology‑grounded answer to the atomic question.

        ### Atomic Question
        {atomic_question}

        ### Classification Signals
        - question_type: {question_type}
        - answer_form: {answer_form}

        ### Extracted Triplet
        - entity: {entity}
        - relation: {relation}
        - object: {obj}

        ### Relevant Ontology Context
        {context_str}
        """
