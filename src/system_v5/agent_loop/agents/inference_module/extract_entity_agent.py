from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt


class ExtractEntityAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\extract-entity.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)
        self.question_type_to_entity_type_examples = {
            "definition": [
                { "input": "What is a protocol?", "anchor_entity": "protocol", "entity_type": "class" },
                { "input": "Who is Alice?", "anchor_entity": "Alice", "entity_type": "individual" },
                { "input": "What is a framework?", "anchor_entity": "framework", "entity_type": "class" }
            ],

            "taxonomic": [
                { "input": "Is a sedan a type of vehicle?", "anchor_entity": "sedan", "entity_type": "class" },
                { "input": "Is a notebook a kind of document?", "anchor_entity": "notebook", "entity_type": "class" },
                { "input": "Is a laptop classified as equipment?", "anchor_entity": "laptop", "entity_type": "class" }
            ],

            "capability": [
                { "input": "Can a scanner read text?", "anchor_entity": "scanner", "entity_type": "class" },
                { "input": "Does a filter detect noise?", "anchor_entity": "filter", "entity_type": "class" },
                { "input": "How does a system generate reports?", "anchor_entity": "system", "entity_type": "class" }
            ],

            "property": [
                { "input": "How tall is the structure?", "anchor_entity": "structure", "entity_type": "class" },
                { "input": "How tall is Alice?", "anchor_entity": "Alice", "entity_type": "individual" },
                { "input": "What is the size of the container?", "anchor_entity": "container", "entity_type": "class" }
            ],

            "membership": [
                { "input": "What items are included in the package?", "anchor_entity": "package", "entity_type": "class" },
                { "input": "What parts does the system contain?", "anchor_entity": "system", "entity_type": "class" },
                { "input": "What components are found in the kit?", "anchor_entity": "kit", "entity_type": "class" }
            ],

            "comparative": [
                { "input": "Is the box bigger than the crate?", "anchor_entity": "box", "entity_type": "class" },
                { "input": "Is Alice faster than Daniel?", "anchor_entity": "Alice", "entity_type": "individual" },
                { "input": "Is the system more efficient than the module?", "anchor_entity": "system", "entity_type": "class" }
            ],

            "quantification": [
                { "input": "How many items does the box have?", "anchor_entity": "box", "entity_type": "class" },
                { "input": "Are there any files in the folder?", "anchor_entity": "folder", "entity_type": "class" },
                { "input": "How many entries belong to the list?", "anchor_entity": "list", "entity_type": "class" }
            ],

            "existential": [
                { "input": "Does the file exist?", "anchor_entity": "file", "entity_type": "class" },
                { "input": "Is Alice present?", "anchor_entity": "Alice", "entity_type": "individual" },
                { "input": "Is there a document?", "anchor_entity": "document", "entity_type": "class" }
            ],

            "unknown": [
                { "input": "What does this mean?", "anchor_entity": "unknown", "entity_type": "unknown" },
                { "input": "Explain the situation.", "anchor_entity": "unknown", "entity_type": "unknown" },
                { "input": "Help me understand this.", "anchor_entity": "unknown", "entity_type": "unknown" }
            ]
            }
    
    def _format_examples(self, examples_dict, question_type):
        """Return all examples for a given question type formatted as Input/Output blocks."""
        blocks = []
        for ex in examples_dict.get(question_type, []):
            block = (
                f'Input: "{ex["input"]}"\n\n'
                "Output:\n"
                "{\n"
                f'  "entity": "{ex["anchor_entity"]}",\n'
                f'  "entity_type": "{ex["entity_type"]}"\n'
                "}"
            )
            blocks.append(block)
        return "\n\n".join(blocks)
    
    def run(self, chunk_text: str, parsed_input: list = None, question_classification: dict = None) -> tuple:
        # Optionally constrain entity to parsed tokens when provided
        if parsed_input:
            entity_prop = {"type": "string", "enum": parsed_input + ["unknown"]}
        else:
            entity_prop = {"type": "string"}

        schema = {
            "type": "object",
            "properties": {
                "entity": entity_prop,
                "entity_type": {"type": "string", "enum": ["class", "individual", "unknown"]}
            },
            "required": ["entity", "entity_type"],
            "additionalProperties": False
        }

        # Provide question-classification context to bias entity typing when available
        qc_info = "None"
        if question_classification and isinstance(question_classification, dict):
            qtype = question_classification.get("question_type", "unknown")
            conf = question_classification.get("confidence")
            qc_info = f"question_type: {qtype}" + (f" (confidence: {conf})" if conf is not None else "")

            # Also provide examples of entity types commonly associated with the question type when possible
            if qtype in self.question_type_to_entity_type_examples.keys():
                qtype_examples_str = self._format_examples(self.question_type_to_entity_type_examples, qtype)
        user_msg = f"""
                    ### Atomic Input
                    {chunk_text}

                    ### Question Classification
                    {qc_info}

                    ### Examples for This Question Type
                    {qtype_examples_str}

                    ### Instructions
                    Using the guidelines and examples above:
                    1. Identify the primary entity the question is about.
                    2. Extract the full noun phrase exactly as written.
                    3. Classify it as `class`, `individual`, or `unknown`.

                    ### Goal
                    Identify the primary entity the sentence is about and classify it as one of: `class`, `individual`, or `unknown`.
                    Use the Question Classification information (if provided) to bias whether the entity is more likely a class or an individual.
                    Return a JSON object matching the provided schema exactly.
                    """
        #print("Running ExtractEntityAgent with user message:")
        #print(user_msg)

        return self.generate_with_schema(user_msg, schema)
