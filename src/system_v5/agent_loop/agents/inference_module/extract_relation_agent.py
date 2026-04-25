from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt

class ExtractRelationAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt(
            "C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\extract-relation.txt"
        )
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, parsed_input: list = None,
            entity_candidate: dict = None, question_classification: str = None) -> tuple:

        # Canonical relation mapping by question type
        question_type_to_relations = {
            "definition": ["be"],
            "taxonomic": ["be subtype of"],
            "capability": ["measure", "detect", "classify", "produce", "use", "unknown"],
            "property": ["have property"],
            "membership": ["include", "contain"],
            "comparative": ["compare", "have property"],
            "quantification": ["count"],
            "existential": ["exist"],
            "unknown": ["unknown"]
        }

        # Override allowed relations if type is known
        if question_classification["question_type"] in question_type_to_relations.keys():
            allowed = question_type_to_relations[question_classification["question_type"]]
            relation_prop = {"type": "string", "enum": allowed}

        # Final schema (relation_type removed from ontology mapping responsibility)
        schema = {
            "type": "object",
            "properties": {
                "relation": relation_prop,
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            },
            "required": ["relation", "confidence"],
            "additionalProperties": False
        }

        # Entity context (light disambiguation only)
        entity_info = "None"
        if isinstance(entity_candidate, dict):
            entity_info = f"entity: {entity_candidate.get('entity')} (type: {entity_candidate.get('entity_type')})"

        # Prompt to the model
        user_msg = f"""
        ### Atomic Input
        {chunk_text}

        ### Entity Context
        {entity_info}

        ### Question Type
        {question_classification}

        ### Goal
        Extract the canonical natural-language predicate that expresses the relation 
        between the entity and the object. Use the question type to restrict or 
        determine the allowed predicates. If multiple allowed predicates exist, 
        choose the one that best matches the sentence. If uncertain, choose "unknown".

        Return a JSON object matching the schema exactly.
        """

        return self.generate_with_schema(user_msg, schema)