from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt

class ExtractTripleAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\extract-triple.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str) -> tuple:
        triple_schema = {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"}
            },
            "required": ["subject", "predicate", "object"],
            "additionalProperties": False
        }

        user_msg = f"""
                    ### Source Text
                    {chunk_text}

                    ### Goal
                    Extract a single factual triple (subject-predicate-object) from the provided Source Text.""" 

        return self.generate_with_schema(user_msg, triple_schema)