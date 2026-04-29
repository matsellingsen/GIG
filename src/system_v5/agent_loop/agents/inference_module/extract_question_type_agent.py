from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt


class ExtractQuestionTypeAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt(
            "C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\extract-question-type.txt"
        )
        self.question_types_dict = {
            "definition": "Questions asking what or who an entity is.",  
            "taxonomic": "Questions about class membership or subclass relations.",
            "capability": "Questions about what an entity does or can do.",
            "property": "Questions about qualities, attributes, or literal values.",
            "membership": "Questions about parts, members, or included items.",
            "comparative": "Questions comparing two entities on a property.",
            "quantification": "Questions about counts or quantities.",
            "existential": "Questions about existence or presence.",
            "unknown": "Fallback for malformed, ambiguous, or unsupported inputs."
        }
        question_types_list = list(self.question_types_dict.keys())
        json_schema = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Short explanation of the reasoning behind the question type classification.",
                },
                "question_type": {
                    "type": "string",
                    "enum": question_types_list,
                    "description": "The classified question type.",
                },
                #"confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["reasoning", "question_type"],# "confidence"],
            "additionalProperties": False,
        }

        super().__init__(backend=backend, system_prompt=system_prompt, json_schema=json_schema)

    def run(self, chunk_text: str) -> tuple:
        question_types_and_descriptions = "\n".join([f"- `{qt}`: {desc}" for qt, desc in self.question_types_dict.items()])

        user_msg = f"""
                ### Goal
                Classify the Atomic Input into exactly one of the allowed question types.

                ### Allowed Question Types
                {question_types_and_descriptions}

                ### Atomic Input
                {chunk_text}
                """

        return self.generate_with_schema(user_msg)
