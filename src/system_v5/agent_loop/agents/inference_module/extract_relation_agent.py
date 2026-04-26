from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt

class ExtractRelationAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt(
            "C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\extract-relation.txt"
        )
        super().__init__(backend=backend, system_prompt=system_prompt)
        self.question_type_to_relations = {
            "definition": {
                "be": {
                "description": "Indicates that the question asks for the identity, definition, or nature of the entity.",
                "examples": [
                    {
                    "input": "What is a protocol?",
                    "primary_entity": "protocol",
                    "entity_type": "class",
                    "relation": "be"
                    },
                    {
                    "input": "Who is Alice?",
                    "primary_entity": "Alice",
                    "entity_type": "individual",
                    "relation": "be"
                    }
                ]
                }
            },

            "taxonomic": {
                "be subtype of": {
                "description": "Indicates a hierarchical class relationship where the entity is a subclass or type of another class.",
                "examples": [
                    {
                    "input": "Is a sedan a type of vehicle?",
                    "primary_entity": "sedan",
                    "entity_type": "class",
                    "relation": "be subtype of"
                    },
                    {
                    "input": "Is a notebook a kind of document?",
                    "primary_entity": "notebook",
                    "entity_type": "class",
                    "relation": "be subtype of"
                    }
                ]
                }
            },

            "capability": {
                "measure": {
                "description": "Indicates that the entity performs measurement on another entity.",
                "examples": [
                    {
                    "input": "Can a sensor measure temperature?",
                    "primary_entity": "sensor",
                    "entity_type": "class",
                    "relation": "measure"
                    },
                    {
                    "input": "Does the tool measure pressure?",
                    "primary_entity": "tool",
                    "entity_type": "class",
                    "relation": "measure"
                    }
                ]
                },
                "detect": {
                "description": "Indicates that the entity identifies, senses, or detects another entity.",
                "examples": [
                    {
                    "input": "Can the device detect motion?",
                    "primary_entity": "device",
                    "entity_type": "class",
                    "relation": "detect"
                    },
                    {
                    "input": "Does the system detect errors?",
                    "primary_entity": "system",
                    "entity_type": "class",
                    "relation": "detect"
                    }
                ]
                },
                "classify": {
                "description": "Indicates that the entity categorizes or assigns types to another entity.",
                "examples": [
                    {
                    "input": "Can the software classify documents?",
                    "primary_entity": "software",
                    "entity_type": "class",
                    "relation": "classify"
                    },
                    {
                    "input": "Does the module classify inputs?",
                    "primary_entity": "module",
                    "entity_type": "class",
                    "relation": "classify"
                    }
                ]
                },
                "produce": {
                "description": "Indicates that the entity generates or outputs another entity.",
                "examples": [
                    {
                    "input": "Does the machine produce heat?",
                    "primary_entity": "machine",
                    "entity_type": "class",
                    "relation": "produce"
                    },
                    {
                    "input": "Can the system produce reports?",
                    "primary_entity": "system",
                    "entity_type": "class",
                    "relation": "produce"
                    }
                ]
                },
                "use": {
                "description": "Indicates that the entity utilizes or relies on another entity.",
                "examples": [
                    {
                    "input": "Does the tool use batteries?",
                    "primary_entity": "tool",
                    "entity_type": "class",
                    "relation": "use"
                    },
                    {
                    "input": "Does the program use external data?",
                    "primary_entity": "program",
                    "entity_type": "class",
                    "relation": "use"
                    }
                ]
                },
                "unknown": {
                "description": "Indicates that the capability relation is unclear or cannot be determined from the question.",
                "examples": [
                    {
                    "input": "How does the system work?",
                    "primary_entity": "system",
                    "entity_type": "class",
                    "relation": "unknown"
                    },
                    {
                    "input": "What does the device do?",
                    "primary_entity": "device",
                    "entity_type": "class",
                    "relation": "unknown"
                    }
                ]
                }
            },

            "property": {
                "have property": {
                "description": "Indicates that the entity possesses a property, attribute, or characteristic.",
                "examples": [
                    {
                    "input": "How tall is the structure?",
                    "primary_entity": "structure",
                    "entity_type": "class",
                    "relation": "have property"
                    },
                    {
                    "input": "What is the color of the badge?",
                    "primary_entity": "badge",
                    "entity_type": "class",
                    "relation": "have property"
                    }
                ]
                }
            },

            "membership": {
                "has member": {
                "description": "Indicates that the entity contains, includes, or has items/members as part of it.",
                "examples": [
                    {
                    "input": "What items are included in the package?",
                    "primary_entity": "package",
                    "entity_type": "class",
                    "relation": "has member"
                    },
                    {
                    "input": "What components are found in the kit?",
                    "primary_entity": "kit",
                    "entity_type": "class",
                    "relation": "has member"
                    }
                ]
                },
                "is member of": {
                "description": "Indicates that the entity belongs to, is included in, or is part of another entity.",
                "examples": [
                    {
                    "input": "Which group is this item part of?",
                    "primary_entity": "item",
                    "entity_type": "class",
                    "relation": "is member of"
                    },
                    {
                    "input": "Which collection does this document belong to?",
                    "primary_entity": "document",
                    "entity_type": "class",
                    "relation": "is member of"
                    }
                ]
                }
            },

            "comparative": {
                "compare": {
                "description": "Indicates that the question compares two entities along some dimension.",
                "examples": [
                    {
                    "input": "Is the box bigger than the crate?",
                    "primary_entity": "box",
                    "entity_type": "class",
                    "relation": "compare"
                    },
                    {
                    "input": "Is Alice faster than Daniel?",
                    "primary_entity": "Alice",
                    "entity_type": "individual",
                    "relation": "compare"
                    }
                ]
                },
                "have property": {
                "description": "Indicates that the comparison is based on a property or attribute of the entity.",
                "examples": [
                    {
                    "input": "Which is heavier, the package or the container?",
                    "primary_entity": "package",
                    "entity_type": "class",
                    "relation": "have property"
                    },
                    {
                    "input": "Which is larger, the folder or the binder?",
                    "primary_entity": "folder",
                    "entity_type": "class",
                    "relation": "have property"
                    }
                ]
                }
            },

            "quantification": {
                "count": {
                "description": "Indicates that the question asks for the number or quantity of items related to the entity.",
                "examples": [
                    {
                    "input": "How many items does the box have?",
                    "primary_entity": "box",
                    "entity_type": "class",
                    "relation": "count"
                    },
                    {
                    "input": "How many entries belong to the list?",
                    "primary_entity": "list",
                    "entity_type": "class",
                    "relation": "count"
                    }
                ]
                }
            },

            "existential": {
                "exist": {
                "description": "Indicates that the question asks whether the entity exists or is present.",
                "examples": [
                    {
                    "input": "Does the file exist?",
                    "primary_entity": "file",
                    "entity_type": "class",
                    "relation": "exist"
                    },
                    {
                    "input": "Is Alice present?",
                    "primary_entity": "Alice",
                    "entity_type": "individual",
                    "relation": "exist"
                    }
                ]
                }
            },

            "unknown": {
                "unknown": {
                "description": "Indicates that the relation cannot be determined from the question.",
                "examples": [
                    {
                    "input": "What does this mean?",
                    "primary_entity": "unknown",
                    "entity_type": "unknown",
                    "relation": "unknown"
                    },
                    {
                    "input": "Explain the situation.",
                    "primary_entity": "unknown",
                    "entity_type": "unknown",
                    "relation": "unknown"
                    }
                ]
                }
            }
            }

    def _prepare_relation_descriptions_and_examples(self, question_type_to_relations, question_type):
        """
        Given the canonical relation mapping and a question type,
        return two formatted strings:
        - relation_descriptions_str
        - relation_examples_str (few-shot examples matching inference format)
        """
        allowed = question_type_to_relations.get(question_type, {})

        # Format descriptions
        relation_descriptions_str = "\n".join(
            f"- `{rel}`: {info['description']}"
            for rel, info in allowed.items()
        )

        # Format examples in the exact inference format
        example_blocks = []
        for rel, info in allowed.items():
            for ex in info["examples"]:
                block = (
                    f'Input: "{ex["input"]}"\n'
                    f'Primary Entity: "{ex["primary_entity"]}"\n'
                    f'Primary Entity Type: "{ex["entity_type"]}"\n'
                    "Output:\n"
                    "{\n"
                    f'  "relation": "{rel}"\n'
                    "}"
                )
                example_blocks.append(block)

        relation_examples_str = "\n\n".join(example_blocks)

        return relation_descriptions_str, relation_examples_str

    
    def run(self, chunk_text: str, parsed_input: list = None,
            entity_candidate: dict = None, question_classification: str = None) -> tuple:

        question_type_to_relations = self.question_type_to_relations


        # Override allowed relations if type is known
        if question_classification["question_type"] in question_type_to_relations.keys():
            allowed = question_type_to_relations[question_classification["question_type"]]
            list_of_allowed = list(allowed.keys())
            relation_prop = {"type": "string", "enum": list_of_allowed}
        
        if len(list_of_allowed) < 2: # choice is deterministic, no need to prompt the model.
            json_output = {"relation": list_of_allowed[0]} #, "confidence": 1.0}
            return json_output, None  # No choice, return the single allowed relation immediately

        # Final schema (relation_type removed from ontology mapping responsibility)
        schema = {
            "type": "object",
            "properties": {
                "relation": relation_prop,
                #"confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            },
            "required": ["relation"],# "confidence"],
            "additionalProperties": False
        }

        # Entity context (light disambiguation only)
        entity_info = "None"
        if isinstance(entity_candidate, dict):
            entity_info = f"entity: {entity_candidate.get('entity')} (type: {entity_candidate.get('entity_type')})"

        # relation descriptions and examples for the allowed relations based on question type
        relation_descriptions_str, relation_examples_str = self._prepare_relation_descriptions_and_examples(question_type_to_relations, question_classification["question_type"])

        # Prompt to the model
        user_msg = f"""
        ### Goal
        Determine which Allowed Relation best describes the semantic relationship expressed in the Atomic Input, given the Primary Entity and its Entity Type. 

        ### Allowed Relations and Their Descriptions
        {relation_descriptions_str}

        ### Examples
        {relation_examples_str}

        ### Atomic Input
        {chunk_text}

        ### Entity Context
        {entity_info}

        ### Question Type
        {question_classification}
        """

        return self.generate_with_schema(user_msg, schema)