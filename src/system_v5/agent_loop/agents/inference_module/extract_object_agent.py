from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt

class ExtractObjectAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt(
            "C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\extract-object.txt"
        )
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, parsed_input: list = None,
        entity_candidate: dict = None, relation_candidate: dict = None,
        question_classification: dict = None) -> tuple:

        # Optionally constrain object to parsed tokens when provided
        # but allow question-type-driven overrides below.
        if parsed_input:
            base_object_prop = {"type": "string", "enum": parsed_input + ["unknown"]}
        else:
            base_object_prop = {"type": "string"}

        # Determine question type (accept dict or plain string)
        qtype = None
        if question_classification:
            if isinstance(question_classification, dict):
                qtype = question_classification.get("question_type")
            else:
                qtype = str(question_classification)

        # Hard mapping: allowed object_type values per question type
        allowed_by_qtype = {
            "definition": ["unknown"],
            "taxonomic": ["class"],
            "capability": ["individual", "class"],
            "property": ["literal"],
            "membership": ["class", "individual"],
            "comparative": ["individual", "class"],
            "quantification": ["class"],
            "existential": ["class", "individual"],
            "unknown": ["class", "individual", "literal", "unknown"]
        }

        # Build object property schema constrained by question type when present
        if qtype in allowed_by_qtype:
            object_type_prop = {"type": "string", "enum": allowed_by_qtype[qtype]}
        else:
            object_type_prop = {"type": "string", "enum": ["class", "individual", "literal", "unknown"]}

        # For some qtypes, also hard-limit the object text itself (e.g., definition->unknown)
        if qtype == "definition":
            object_prop = {"type": "string", "enum": ["unknown"]}
        else:
            object_prop = base_object_prop

        schema = {
            "type": "object",
            "properties": {
                "object": object_prop,
                "object_type": object_type_prop
            },
            "required": ["object", "object_type"],
            "additionalProperties": False
        }

        # Entity context (optional)
        entity_info = "None"
        if isinstance(entity_candidate, dict):
            entity_info = f"entity: {entity_candidate.get('entity')} (type: {entity_candidate.get('entity_type')})"

        # Relation context (optional)
        relation_info = "None"
        if isinstance(relation_candidate, dict):
            relation_info = f"relation: {relation_candidate.get('relation')}"

        user_msg = f"""
        ### Atomic Input
        {chunk_text}

        ### Entity Context
        {entity_info}

        ### Relation Context
        {relation_info}

        ### Question Type
        {question_classification}

        ### Goal
        Extract the object (the target/value of the relation) and classify it as 
        `class`, `individual`, `literal`, or `unknown`. Use the question type to 
        determine where to look for the object and how to type it.

        Return a JSON object matching the schema exactly.
        """

        return self.generate_with_schema(user_msg, schema)