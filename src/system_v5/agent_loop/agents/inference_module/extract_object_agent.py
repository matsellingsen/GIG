from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt

class ExtractObjectAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt(
            "C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\extract-object.txt"
        )
        super().__init__(backend=backend, system_prompt=system_prompt)

        # Few-shot examples per question type
        self.qtype_to_examples = {
            "definition": [
                {
                    "input": "What is a protocol?",
                    "object": "unknown",
                    "object_type": "unknown"
                },
                {
                    "input": "Who is Alice?",
                    "object": "unknown",
                    "object_type": "unknown"
                }
            ],
            "taxonomic": [
                {
                    "input": "Is a sedan a type of vehicle?",
                    "object": "vehicle",
                    "object_type": "class"
                },
                {
                    "input": "Is a notebook a kind of document?",
                    "object": "document",
                    "object_type": "class"
                }
            ],
            "capability": [
                {
                    "input": "Can a scanner read text?",
                    "object": "text",
                    "object_type": "class"
                },
                {
                    "input": "Does a filter detect noise?",
                    "object": "noise",
                    "object_type": "class"
                }
            ],
            "property": [
                {
                    "input": "How tall is the structure?",
                    "object": "tall",
                    "object_type": "literal"
                },
                {
                    "input": "What is the size of the container?",
                    "object": "size",
                    "object_type": "literal"
                }
            ],
            "membership": [
                {
                    "input": "What items are included in the package?",
                    "object": "items",
                    "object_type": "class"
                },
                {
                    "input": "What parts does the system contain?",
                    "object": "parts",
                    "object_type": "class"
                },
                {
                    "input": "What elements are included in the device?",
                    "object": "elements",
                    "object_type": "class"
                }
            ],
            "comparative": [
                {
                    "input": "Is the box bigger than the crate?",
                    "object": "crate",
                    "object_type": "class"
                },
                {
                    "input": "Is Alice faster than Daniel?",
                    "object": "Daniel",
                    "object_type": "individual"
                }
            ],
            "quantification": [
                {
                    "input": "How many items does the box have?",
                    "object": "items",
                    "object_type": "class"
                },
                {
                    "input": "How many entries belong to the list?",
                    "object": "entries",
                    "object_type": "class"
                }
            ],
            "existential": [
                {
                    "input": "Does the file exist?",
                    "object": "file",
                    "object_type": "class"
                },
                {
                    "input": "Is Alice present?",
                    "object": "Alice",
                    "object_type": "individual"
                }
            ],
            "unknown": [
                {
                    "input": "Explain the situation.",
                    "object": "unknown",
                    "object_type": "unknown"
                }
            ]
        }

    def _format_examples(self, qtype):
        blocks = []
        for ex in self.qtype_to_examples.get(qtype, []):
            block = (
                f'Input: "{ex["input"]}"\n\n'
                "Output:\n"
                "{\n"
                f'  "object": "{ex["object"]}",\n'
                f'  "object_type": "{ex["object_type"]}"\n'
                "}"
            )
            blocks.append(block)
        return "\n\n".join(blocks)

    def run(self, chunk_text: str, parsed_input: list = None,
            entity_candidate: dict = None, relation_candidate: dict = None,
            question_classification: dict = None) -> tuple:

        # Determine question type
        qtype = None
        if isinstance(question_classification, dict):
            qtype = question_classification.get("question_type", "unknown")
        else:
            qtype = str(question_classification) if question_classification else "unknown"

        # Build schema
        if parsed_input:
            unique = [str(x) for x in dict.fromkeys(parsed_input)]
            if "unknown" not in unique:
                unique.append("unknown")
            base_object_prop = {"type": "string", "enum": unique}
        else:
            base_object_prop = {"type": "string"}

        allowed_by_qtype = {
            "definition": ["null"],
            "taxonomic": ["class", "individual"],
            "capability": ["individual", "class"],
            "property": ["literal"],
            "membership": ["class", "individual"],
            "comparative": ["individual", "class"],
            "quantification": ["class"],
            "existential": ["class", "individual"],
            "unknown": ["class", "individual", "literal", "null"]
        }

        object_type_prop = {
            "type": "string",
            "enum": allowed_by_qtype.get(qtype, ["class", "individual", "literal", "null"])
        }

        if qtype == "definition": #SKIP GENERATION (only 1 valid option so generation is trivial)
            object_prop = {"type": "string", "enum": ["null"]}
            json_output = {"object": "null", "object_type": "null"}
            return json_output, None # deterministic output (only "null" is valid), so skip the agent call and return directly
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

        # Build user message
        examples_str = self._format_examples(qtype)

        entity_info = (
            f'entity: {entity_candidate.get("entity")} (type: {entity_candidate.get("entity_type")})'
            if isinstance(entity_candidate, dict) else "None"
        )

        relation_info = (
            f'relation: {relation_candidate.get("relation")}'
            if isinstance(relation_candidate, dict) else "None"
        )

        user_msg = f"""
        ### Goal
        Extract the object (the target/value of the relation) and classify it as
        `class`, `individual`, `literal`, or `null`. Use the Question Type and the
        Entity/Relation context to determine where to look for the object and how to type it.

        ### Examples for This Question Type
        {examples_str}

        ### Atomic Input
        {chunk_text}

        ### Entity Context
        {entity_info}

        ### Relation Context
        {relation_info}

        ### Question Type
        {question_classification}
        """

        return self.generate_with_schema(user_msg, schema)