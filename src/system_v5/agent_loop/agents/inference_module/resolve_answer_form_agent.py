from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt


class ResolveAnswerFormAgent(BaseOntologyAgent):


    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\resolve-answer-form.txt"
        )
        self.all_answer_forms_list = ["assertion", "value", "list"]
        super().__init__(system_prompt=system_prompt, backend=backend)

    def resolve_answer_form(self,question_type: str) -> list:
            if question_type == "definition":
                return ["value"]
            elif question_type == "taxonomic":
                return ["value", "list"]
            elif question_type == "property":
                return ["value", "list"]
            elif question_type == "membership":
                return ["list"]
            elif question_type == "capability":
                 return ["assertion", "value"]
            elif question_type == "comparative":
                return ["assertion", "value"]
            elif question_type == "quantification":
                return ["list", "value"]
            elif question_type == "existential":
                return ["assertion"]
            else:
                return self.all_answer_forms_list
            
    def run(self, chunk_text: str, question_type: str) -> tuple:
        answer_forms_list = self.resolve_answer_form(question_type)
        answer_forms = "\n".join([f"- {form}" for form in answer_forms_list])
        json_schema = {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Short explanation of the reasoning behind the answer form classification.",
                    },
                    "answer_form": {
                        "type": "string",
                        "enum": answer_forms_list,
                        "description": "The classified answer form.",
                    },
                    #"confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["reasoning", "answer_form"],# "confidence"],
                "additionalProperties": False,
            }
        
        user_msg = f"""
                ### Goal
                Classify the Atomic Input into exactly one of the allowed answer forms.

                ### Atomic Input
                {chunk_text}

                ### Allowed Answer Forms
                {answer_forms}
                """

        return self.generate_with_schema(user_msg, json_schema)
