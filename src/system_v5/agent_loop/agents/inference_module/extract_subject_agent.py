from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt


class ExtractSubjectAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\extract-subject.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, parsed_input: list = None) -> tuple:
        # If parsed_input tokens are provided, constrain the subject string to those tokens when possible
        if parsed_input:
            subject_schema = {"type": "string", "enum": parsed_input + ["unknown"]}
        else:
            subject_schema = {"type": "string"}

        user_msg = f"""
                ### Atomic Question/Statement
                {chunk_text}

                ### Goal
                Return the most appropriate subject explicitly present in the Atomic Question/Statement. If no appropriate subject exists, use "unknown".
                """

        return self.generate_with_schema(user_msg, subject_schema)
