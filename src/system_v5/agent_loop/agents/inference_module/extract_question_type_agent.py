import json

from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt


class ExtractQuestionTypeAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt(
            "C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\extract-question-type.txt"
        )
        self.question_types_examples = {
        "definition": [
            {
                "input": "What is an Object?",
                "output": {
                    "reasoning": "The question asks for the identity or nature of the entity.",
                    "question_type": "definition"
                }
            },
            {
                "input": "What is a Device?",
                "output": {
                    "reasoning": "The question seeks a general description of the entity.",
                    "question_type": "definition"
                }
            }
        ],
        "taxonomic": [
            {
                "input": "Is an Object a type of Entity?",
                "output": {
                    "reasoning": "The question asks whether one entity belongs to a category.",
                    "question_type": "taxonomic"
                }
            },
            {
                "input": "What type of thing is a Device?",
                "output": {
                    "reasoning": "The question asks for the category the entity belongs to.",
                    "question_type": "taxonomic"
                }
            }
        ],
        "capability": [
            {
                "input": "What does a Device do?",
                "output": {
                    "reasoning": "The question asks about the function or action of the entity.",
                    "question_type": "capability"
                }
            },
            {
                "input": "How does an Object perform its task?",
                "output": {
                    "reasoning": "The question asks about a process the entity participates in.",
                    "question_type": "capability"
                }
            }
        ],
        "property": [
            {
                "input": "What is the status of this Item?",
                "output": {
                    "reasoning": "The question asks for a specific attribute value.",
                    "question_type": "property"
                }
            },
            {
                "input": "What is the label of the Object?",
                "output": {
                    "reasoning": "The question requests a literal property of the entity.",
                    "question_type": "property"
                }
            }
        ],
        "membership": [
            {
                "input": "What components does the System contain?",
                "output": {
                    "reasoning": "The question asks for the parts contained within the entity.",
                    "question_type": "membership"
                }
            },
            {
                "input": "Which elements are included in the Collection?",
                "output": {
                    "reasoning": "The question asks for members of a collection.",
                    "question_type": "membership"
                }
            }
        ],
        "comparative": [
            {
                "input": "Is Object A larger than Object B?",
                "output": {
                    "reasoning": "The question compares two entities along a dimension.",
                    "question_type": "comparative"
                }
            },
            {
                "input": "Which is faster, the Device or the Tool?",
                "output": {
                    "reasoning": "The question requests a comparison between two entities.",
                    "question_type": "comparative"
                }
            }
        ],
        "quantification": [
            {
                "input": "How many Items are in the System?",
                "output": {
                    "reasoning": "The question asks for a count of items.",
                    "question_type": "quantification"
                }
            },
            {
                "input": "What is the number of Devices in the collection?",
                "output": {
                    "reasoning": "The question requests a quantity.",
                    "question_type": "quantification"
                }
            }
        ],
        
        #"existential": [
        #    {
        #        "input": "Does this Object exist?",
        #        "output": {
        #            "reasoning": "The question asks whether an entity exists.",
        #            "question_type": "existential"
        #        }
        #    },
        #    {
        #        "input": "Are there any Items present?",
        #        "output": {
        #            "reasoning": "The question asks whether instances are present.",
        #            "question_type": "existential"
        #        }
        #    }
        #],
        "unknown": [
            {
                "input": "Explain this concept.",
                "output": {
                    "reasoning": "The question is open-ended and not a structured atomic query.",
                    "question_type": "unknown"
                }
            },
            {
                "input": "What does this mean?",
                "output": {
                    "reasoning": "The question lacks a clear semantic target.",
                    "question_type": "unknown"
                }
            }
        ]
}

        question_types_list = list(self.question_types_examples.keys())
        json_schema = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Short explanation of the reasoning behind the question type classification.",
                },
                "question_type": {
                    "type": "string",
                    "enum": question_types_list,
                    "description": "The classified question type.",
                },
                #"confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["reasoning", "question_type"],# "confidence"],
            "additionalProperties": False,
        }

        super().__init__(backend=backend, system_prompt=system_prompt, json_schema=json_schema)

    def run(self, chunk_text: str) -> tuple:
        #question_types_and_descriptions = "\n".join([f"- `{qt}`: {desc}" for qt, desc in self.question_types_dict.items()])
        allowed_question_types_n_examples = json.dumps(self.question_types_examples, indent=2)
        user_msg = f"""
                ### Goal
                Classify the Atomic Input into exactly one of the allowed question types.

                ### Allowed Question Types and Examples
                {allowed_question_types_n_examples}

                ### Atomic Input
                {chunk_text}
                """

        return self.generate_with_schema(user_msg)
