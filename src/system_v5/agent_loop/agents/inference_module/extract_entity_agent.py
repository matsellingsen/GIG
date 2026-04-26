from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt


class ExtractEntityAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\extract-entity.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)
        self.question_type_to_entity_type_examples = {
            "definition": [
                { "input": "What is a protocol?", "primary_entity": "protocol", "entity_type": "class" },
                { "input": "Who is Alice?", "primary_entity": "Alice", "entity_type": "individual" },
                { "input": "What is a framework?", "primary_entity": "framework", "entity_type": "class" }
            ],

            "taxonomic": [
                { "input": "Is a sedan a type of vehicle?", "primary_entity": "sedan", "entity_type": "class" },
                { "input": "Is a notebook a kind of document?", "primary_entity": "notebook", "entity_type": "class" },
                { "input": "Is a laptop classified as equipment?", "primary_entity": "laptop", "entity_type": "class" }
            ],

            "capability": [
                { "input": "Can a scanner read text?", "primary_entity": "scanner", "entity_type": "class" },
                { "input": "Does a filter detect noise?", "primary_entity": "filter", "entity_type": "class" },
                { "input": "How does a system generate reports?", "primary_entity": "system", "entity_type": "class" }
            ],

            "property": [
                { "input": "How tall is the structure?", "primary_entity": "structure", "entity_type": "class" },
                { "input": "How tall is Alice?", "primary_entity": "Alice", "entity_type": "individual" },
                { "input": "What is the size of the container?", "primary_entity": "container", "entity_type": "class" }
            ],

            "membership": [{"input": "What items are included in the package?", "primary_entity": "package", "entity_type": "class"},
                           {"input": "What parts does the system contain?", "primary_entity": "system", "entity_type": "class"},
                           {"input": "What files are stored in Data Archive System?", "primary_entity": "Data Archive System", "entity_type": "individual"}
            ],


            "comparative": [
                { "input": "Is the box bigger than the crate?", "primary_entity": "box", "entity_type": "class" },
                { "input": "Is Alice faster than Daniel?", "primary_entity": "Alice", "entity_type": "individual" },
                { "input": "Is the system more efficient than the module?", "primary_entity": "system", "entity_type": "class" }
            ],

            "quantification": [
                { "input": "How many items does the box have?", "primary_entity": "box", "entity_type": "class" },
                { "input": "Are there any files in the folder?", "primary_entity": "folder", "entity_type": "class" },
                { "input": "How many entries belong to the list?", "primary_entity": "list", "entity_type": "class" }
            ],

            "existential": [
                { "input": "Does the file exist?", "primary_entity": "file", "entity_type": "class" },
                { "input": "Is Alice present?", "primary_entity": "Alice", "entity_type": "individual" },
                { "input": "Is there a document?", "primary_entity": "document", "entity_type": "class" }
            ],

            "unknown": [
                { "input": "What does this mean?", "primary_entity": "unknown", "entity_type": "unknown" },
                { "input": "Explain the situation.", "primary_entity": "unknown", "entity_type": "unknown" },
                { "input": "Help me understand this.", "primary_entity": "unknown", "entity_type": "unknown" }
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
                f'  "entity": "{ex["primary_entity"]}",\n'
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
                    ### Goal
                    Identify the Primary Entity that functions as the subject or container in the relation expressed by the sentence. and classify it as 
                    `class`, `individual`, or `unknown`.

                    ### Examples for This Question Type
                    {qtype_examples_str}

                    ### Atomic Input
                    {chunk_text}

                    ### Question type
                    {qc_info}
                    """
        #print("Running ExtractEntityAgent with user message:")
        #print(user_msg)

        return self.generate_with_schema(user_msg, schema)
