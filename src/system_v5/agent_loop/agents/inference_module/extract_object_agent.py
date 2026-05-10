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
            question_classification: dict = None, answer_form: dict = None) -> tuple:
   
        # Determine question type
        qtype = None
        if isinstance(question_classification, dict):
            qtype = question_classification.get("question_type", "unknown")
        else:
            qtype = str(question_classification) if question_classification else "unknown"

        # Determine answer form
        aform = None
        if isinstance(answer_form, dict):
            aform = answer_form.get("answer_form", "unknown")
        else:
            aform = str(answer_form) if answer_form else "unknown"

        #Override system prompt conditionally if question type is comparative, since this requires a different reasoning approach and the base prompt is not sufficient to guide the model to the correct answer space.
        if qtype == "comparative":
            print("COMPARATIVE QUESTION DETECTED - SWITCHING TO COMPARATIVE SYSTEM PROMPT AND EXAMPLES")
            self.system_prompt = load_prompt(
                "C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\extract-object-comparative.txt"
            )
        # Build schema
        if parsed_input:
            unique = [str(x) for x in dict.fromkeys(parsed_input)]
            if "unknown" not in unique:
                unique.append("unknown")
            base_object_prop = {"type": "string", "enum": unique}
        else:
            base_object_prop = {"type": "string"}

        deterministic_combinations = [("capability", "value")]
        determinisitic_qtypes = ["definition", "unknown", "quantification"] # For these question types, the object is either not needed or not extractable, so we can skip generation and return "null" directly.
        deterministic_aforms = ["list"]
        if ((qtype in determinisitic_qtypes) or
            (aform in deterministic_aforms) or
            ((qtype, aform) in deterministic_combinations)): #SKIP GENERATION (object is not needed or not extractable for these cases, so return "null" directly)
            object_prop = {"type": "string", "enum": ["null"]}
            json_output = {"reasoning": "Deterministic output", "object": "null"}
            return json_output, None # deterministic output (only "null" is valid), so skip the agent call and return directly
        
        else:
            object_prop = base_object_prop

        schema = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Short explanation of the reasoning behind the object extraction.",
                    },
                "object": object_prop,
            },
            "required": ["reasoning", "object"],
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
        Extract the object (the target/value of the relation).
        Use the Question Type and the Extracted Entity and Relation to determine where to look for the object and how to type it.

        ### Question Type
        {question_classification}

        ### Extracted Entity
        {entity_info}

        ### Extracted Relation
        {relation_info}

        ### Atomic Input
        {chunk_text}

        """

        return self.generate_with_schema(user_msg, schema)