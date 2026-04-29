from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt


class ValidateAnswerAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt(
            "C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\validate-answer.txt"
        )
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(
        self,
        atomic_question: str = None,
        question_type: str = None,
        extracted_triplet: dict = None,
        generated_answer: str = None,
        mapped_answer_triplets: list = None,
    ) -> tuple:
        """
        Validate whether the generated answer fully answers the question,
        based on the extracted triplet and the mapped answer triplets.
        """

        # Normalize inputs
        qtype = None
        if isinstance(question_type, dict):
            qtype = question_type.get("question_type", "unknown")
        else:
            qtype = str(question_type) if question_type else "unknown"

        atomic_question_str = atomic_question or ""
        extracted_triplet_str = str(extracted_triplet) if extracted_triplet is not None else "None"
        generated_answer_str = generated_answer or ""
        mapped_answer_triplets_str = (
            str(mapped_answer_triplets) if mapped_answer_triplets is not None else "[]"
        )

        # JSON schema: reasoning (string), decision (boolean)
        schema = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Short explanation of whether the answer fully answers the question.",
                },
                "decision": {
                    "type": "boolean",
                    "description": "True if the answer fully answers the question, otherwise False.",
                },
            },
            "required": ["reasoning", "decision"],
            "additionalProperties": False,
        }

        # Build user message
        user_msg = f"""
        ### Goal
        Decide whether the generated answer fully answers the atomic question,
        given the question type, the extracted triplet, and the mapped answer triplets.

        ### Atomic question
        {atomic_question_str}

        ### Question type
        {qtype}

        ### Extracted triplet
        {extracted_triplet_str}

        ### Generated answer
        {generated_answer_str}

        ### Mapped answer triplets
        {mapped_answer_triplets_str}
        """

        return self.generate_with_schema(user_msg, schema)
