import json

from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt


class ExtractEntityAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\extract-entity.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)
        self.question_type_to_entity_type_examples = {
            "definition": [
                {
                "input": "What is a protocol?",
                "reasoning": "The question directly asks for the definition of 'protocol'.",
                "primary_entity": "protocol"
                },
                {
                "input": "What does the term framework mean?",
                "reasoning": "The question seeks the meaning of the term 'framework'.",
                "primary_entity": "framework"
                },
                {
                "input": "Could you explain what a data model refers to?",
                "reasoning": "This paraphrased question still asks for the definition of 'data model'.",
                "primary_entity": "data model"
                },
                {
                "input": "How would you describe a schema in simple terms?",
                "reasoning": "The question requests a description of the concept 'schema'.",
                "primary_entity": "schema"
                },
                {
                "input": "I'm not sure what a repository actually is—can you clarify?",
                "reasoning": "Despite the conversational phrasing, the question asks for the definition of 'repository'.",
                "primary_entity": "repository"
                },
                {
                "input": "Before we continue, what exactly is an algorithm supposed to be?",
                "reasoning": "The entity being defined is 'algorithm', even though the question is embedded in a longer sentence.",
                "primary_entity": "algorithm"
                }
            ],

            "taxonomic": [
                {
                "input": "Is a sedan a type of vehicle?",
                "reasoning": "Taxonomic questions classify the first entity; here it is 'sedan'.",
                "primary_entity": "sedan"
                },
                {
                "input": "Is a notebook considered a document?",
                "reasoning": "The entity being classified is 'notebook'.",
                "primary_entity": "notebook"
                },
                {
                "input": "Would you classify a tablet as a mobile device?",
                "reasoning": "The question asks whether 'tablet' belongs to a broader class.",
                "primary_entity": "tablet"
                },
                {
                "input": "Does a router fall under networking equipment?",
                "reasoning": "The classification target is 'router'.",
                "primary_entity": "router"
                },
                {
                "input": "People often say a hatchback is basically a car—does that hold here?",
                "reasoning": "Despite the informal phrasing, the classification target is still 'hatchback'.",
                "primary_entity": "hatchback"
                },
                {
                "input": "If we treat equipment broadly, would a laptop count as part of it?",
                "reasoning": "The question checks whether 'laptop' fits into a class, making it the primary entity.",
                "primary_entity": "laptop"
                }
            ],

            "capability": [
                {
                "input": "Can a scanner read text?",
                "reasoning": "Capability questions target the acting entity; here it is 'scanner'.",
                "primary_entity": "scanner"
                },
                {
                "input": "Does a filter detect noise?",
                "reasoning": "The question concerns the abilities of 'filter'.",
                "primary_entity": "filter"
                },
                {
                "input": "How does a system generate reports?",
                "reasoning": "Value‑form capability questions ask how the entity performs an action.",
                "primary_entity": "system"
                },
                {
                "input": "In what way does the engine compute results?",
                "reasoning": "The question asks how 'engine' performs a capability.",
                "primary_entity": "engine"
                },
                {
                "input": "Is the module actually capable of processing audio, or is that a misconception?",
                "reasoning": "Despite the adversarial phrasing, the capability target is 'module'.",
                "primary_entity": "module"
                },
                {
                "input": "When people say the application can export data, are they right?",
                "reasoning": "The question still asks whether 'application' has a capability.",
                "primary_entity": "application"
                }
            ],

            "property": [
                {
                "input": "How tall is the structure?",
                "reasoning": "Property questions ask about an attribute of the subject.",
                "primary_entity": "structure"
                },
                {
                "input": "What is the size of the container?",
                "reasoning": "The question concerns a property of 'container'.",
                "primary_entity": "container"
                },
                {
                "input": "How heavy is the device?",
                "reasoning": "The question asks for a property of 'device'.",
                "primary_entity": "device"
                },
                {
                "input": "What is the temperature of the solution?",
                "reasoning": "The question asks for a property of 'solution'.",
                "primary_entity": "solution"
                },
                {
                "input": "I'm trying to figure out the length of the cable—do you know it?",
                "reasoning": "Despite the conversational phrasing, the property target is 'cable'.",
                "primary_entity": "cable"
                },
                {
                "input": "Before we proceed, what's the width of that panel we discussed earlier?",
                "reasoning": "The question asks for a property of 'panel', even though it's embedded in context.",
                "primary_entity": "panel"
                }
            ],

            "membership": [
                {
                "input": "What items are included in the package?",
                "reasoning": "Membership questions ask for members of a collection; here 'package'.",
                "primary_entity": "package"
                },
                {
                "input": "What parts does the system contain?",
                "reasoning": "The question asks for the components of 'system'.",
                "primary_entity": "system"
                },
                {
                "input": "Which files are stored in the Data Archive System?",
                "reasoning": "The question asks for members of 'Data Archive System'.",
                "primary_entity": "Data Archive System"
                },
                {
                "input": "What modules belong to the platform?",
                "reasoning": "The question asks for the members of 'platform'.",
                "primary_entity": "platform"
                },
                {
                "input": "Could you list the sensors that are part of the monitoring unit?",
                "reasoning": "The question asks for the members of 'monitoring unit'.",
                "primary_entity": "monitoring unit"
                },
                {
                "input": "I'm reviewing the repository—what datasets does it actually include?",
                "reasoning": "Despite the conversational phrasing, the collection is 'repository'.",
                "primary_entity": "repository"
                }
            ],

            "comparative": [
                {
                "input": "Is the box bigger than the crate?",
                "reasoning": "Comparative questions extract the first entity being compared.",
                "primary_entity": "box"
                },
                {
                "input": "Is Alice faster than Daniel?",
                "reasoning": "The comparison centers on the first entity, 'Alice'.",
                "primary_entity": "Alice"
                },
                {
                "input": "How much heavier is the engine compared to the motor?",
                "reasoning": "Value‑form comparative questions still extract the first entity.",
                "primary_entity": "engine"
                },
                {
                "input": "Which component is larger: the panel or the frame?",
                "reasoning": "The first entity in the comparison is extracted.",
                "primary_entity": "panel"
                },
                {
                "input": "Between the new model and the old one, is the new model more reliable?",
                "reasoning": "The comparison is framed around the first entity.",
                "primary_entity": "new model"
                },
                {
                "input": "If we compare the processor with the accelerator, is the processor generally faster?",
                "reasoning": "Despite the embedded phrasing, the first compared entity is 'processor'.",
                "primary_entity": "processor"
                }
            ],

            "quantification": [
                {
                "input": "How many items does the box have?",
                "reasoning": "Quantification questions ask about the number of members belonging to the primary entity.",
                "primary_entity": "box"
                },
                {
                "input": "Are there any files in the folder?",
                "reasoning": "Assertion‑form quantification still targets the container entity.",
                "primary_entity": "folder"
                },
                {
                "input": "How many entries belong to the list?",
                "reasoning": "The question asks for the number of members of 'list'.",
                "primary_entity": "list"
                },
                {
                "input": "Does the dataset contain any records?",
                "reasoning": "The question asks whether 'dataset' contains members.",
                "primary_entity": "dataset"
                },
                {
                "input": "Roughly how many modules are included in the platform?",
                "reasoning": "The question asks for the count of members of 'platform'.",
                "primary_entity": "platform"
                },
                {
                "input": "I'm checking the monitoring unit—are multiple sensors part of it?",
                "reasoning": "Despite the conversational phrasing, the entity being quantified is 'monitoring unit'.",
                "primary_entity": "monitoring unit"
                }
            ],

            "existential": [
                {
                "input": "Does the file exist?",
                "reasoning": "Existential questions ask whether the subject exists.",
                "primary_entity": "file"
                },
                {
                "input": "Is Alice present?",
                "reasoning": "The question concerns the existence or presence of 'Alice'.",
                "primary_entity": "Alice"
                },
                {
                "input": "Is there a document?",
                "reasoning": "The entity whose existence is questioned is 'document'.",
                "primary_entity": "document"
                },
                {
                "input": "Does a backup copy exist?",
                "reasoning": "The question asks whether 'backup copy' exists.",
                "primary_entity": "backup copy"
                },
                {
                "input": "Is any sensor active right now?",
                "reasoning": "The question asks about the existence of an active sensor.",
                "primary_entity": "sensor"
                },
                {
                "input": "Before we continue, is there even a log file available?",
                "reasoning": "Despite the embedded phrasing, the existence target is 'log file'.",
                "primary_entity": "log file"
                }
            ],

            "unknown": [
                {
                "input": "What does this mean?",
                "reasoning": "No explicit entity is referenced.",
                "primary_entity": "unknown"
                },
                {
                "input": "Explain the situation.",
                "reasoning": "The question contains no identifiable entity.",
                "primary_entity": "unknown"
                },
                {
                "input": "Help me understand this.",
                "reasoning": "There is no extractable entity.",
                "primary_entity": "unknown"
                },
                {
                "input": "Can you clarify this?",
                "reasoning": "The question lacks any entity reference.",
                "primary_entity": "unknown"
                },
                {
                "input": "What is going on here anyway?",
                "reasoning": "No entity is mentioned.",
                "primary_entity": "unknown"
                },
                {
                "input": "I don't get this at all.",
                "reasoning": "The input contains no entity.",
                "primary_entity": "unknown"
                }
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
                f'  "reasoning": "{ex["reasoning"]}",\n'
                f'  "entity": "{ex["primary_entity"]}",\n'
                "}"
            )
            blocks.append(block)
        return "\n\n".join(blocks)
    
    def run(self, chunk_text: str, parsed_input: list = None, question_classification: dict = None) -> tuple:
        # Optionally constrain entity to parsed tokens when provided
        if parsed_input:
            unique = [str(x) for x in dict.fromkeys(parsed_input)]
            if "unknown" not in unique:
                unique.append("unknown")
            entity_prop = {"type": "string", "enum": unique}
        else:
            entity_prop = {"type": "string"}

        schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Short explanation of the reasoning behind the entity extraction and classification."},
                "primary_entity": entity_prop,
                #"entity_type": {"type": "string", "enum": ["class", "individual", "unknown"]}
            },
            "required": ["reasoning", "primary_entity"],#, "entity_type"],
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
                qtype_examples_str = json.dumps(self.question_type_to_entity_type_examples[qtype], indent=2)
                #qtype_examples_str = self._format_examples(self.question_type_to_entity_type_examples, qtype)

        user_msg = f"""
                    ### Goal
                    Identify the Primary Entity that functions as the subject or container in the relation expressed by the sentence.
                    ### Question type
                    {qc_info}

                    ### Examples for question type "{qtype}"
                    {qtype_examples_str}

                    ### Atomic Input
                    {chunk_text}
                    """

        return self.generate_with_schema(user_msg, schema)
