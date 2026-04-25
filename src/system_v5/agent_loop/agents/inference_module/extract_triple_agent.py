from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt
from tools.base_ontology.load_base_ontology import load_axioms

class ExtractTripleAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\extract-triple.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, parsed_input: list = None) -> tuple:
        # Load base-axioms and build an enum for the predicate label when available
        axioms = []
        try:
            axioms = load_axioms()
        except Exception:
            axioms = []

        axiom_labels = [a.get("type") for a in axioms if isinstance(a, dict) and a.get("type")]
        axiom_descriptions = {a.get("type"): a.get("description", "") for a in axioms if isinstance(a, dict) and a.get("type")}
        # allow explicit 'unknown' sentinel when no base-axiom applies
        if axiom_labels:
            predicate_prop = {"type": "string", "enum": axiom_labels + ["unknown"]}
        else:
            predicate_prop = {"type": "string"}
        
        if parsed_input:
            subject_prop = {"type": "string", "enum": parsed_input + ["unknown"]}
            object_prop = {"type": "string", "enum": parsed_input + ["unknown"]}
        else:
            subject_prop = {"type": "string"}
            object_prop = {"type": "string"}

        triple_schema = {
            "type": "object",
            "properties": {
                "subject": subject_prop,
                "predicate": predicate_prop,
                "object": object_prop,
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "uri": {"type": "string"},
                            "label": {"type": "string"},
                            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["uri", "label", "score"],
                        "additionalProperties": False
                    },
                    "minItems": 1,
                    "maxItems": 5
                },
                "spans": {
                    "type": "object",
                    "properties": {
                        "subject_span": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "start": {"type": "integer"},
                                "end": {"type": "integer"}
                            }
                        },
                        "predicate_span": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "start": {"type": "integer"},
                                "end": {"type": "integer"}
                            }
                        },
                        "object_span": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "start": {"type": "integer"},
                                "end": {"type": "integer"}
                            }
                        }
                    },
                    "additionalProperties": False
                }
            },
            "required": ["subject", "predicate", "object", "confidence"],
            "additionalProperties": False
        }

        user_msg = f"""
                    ### Base Ontology Axioms
                    {axiom_descriptions}
        
                    ### Atomic Question/Statement
                    {chunk_text}

                    ### Goal
                    Extract a single factual triple (subject-predicate-object) from the provided Atomic Question/Statement.""" 

        return self.generate_with_schema(user_msg, triple_schema)