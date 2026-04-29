from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt


class ResolveAnswerFormAgent(BaseOntologyAgent):


    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\resolve-answer-form.txt"
        )

        self.answer_forms_list = ["assertion", "value", "list"]
        json_schema = {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Short explanation of the reasoning behind the answer form classification.",
                    },
                    "answer_form": {
                        "type": "string",
                        "enum": self.answer_forms_list,
                        "description": "The classified answer form.",
                    },
                    #"confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["reasoning", "answer_form"],# "confidence"],
                "additionalProperties": False,
            }
        super().__init__(system_prompt=system_prompt, backend=backend, json_schema=json_schema)


    def run(self, chunk_text: str) -> tuple:
        answer_forms = "\n".join([f"- {form}" for form in self.answer_forms_list])
        user_msg = f"""
                ### Goal
                Classify the Atomic Input into exactly one of the allowed answer forms.

                ### Atomic Input
                {chunk_text}

                ### Allowed Answer Forms
                {answer_forms}
                """

        return self.generate_with_schema(user_msg)
