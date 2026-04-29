from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt


class ExtractQuestionTypeAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt(
            "C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\inference_module\\extract-question-type.txt"
        )
        self.question_types_dict = {
            "definition": (
                "Questions asking for the identity, nature, or general description of an entity. "
                "These questions seek to establish *what the entity is*, not a specific attribute. "
                "Typical linguistic cues include 'What is X?' or 'Who is X?'. "
                "Also includes questions about the general purpose or role of an entity when framed "
                "as part of its identity (e.g., 'What is X used for?' when the intent is definitional)."
            ),

            "taxonomic": (
                "Questions about class membership, category assignment, or hierarchical relations. "
                "These questions ask whether an entity belongs to a class, type, or category, or what "
                "category it belongs to. They involve relations like 'type of', 'kind of', 'category of', "
                "'subclass of', or 'is every A a B'."
            ),

            "capability": (
                "Questions about what an entity does, can do, or what processes it participates in. "
                "These questions focus on actions, functions, behaviors, or mechanisms. "
                "They often involve verbs like 'measure', 'detect', 'produce', 'generate', 'perform', "
                "or phrases like 'How does X do Y?' or 'What does X do?'."
            ),

            "property": (
                "Questions about attributes, roles, states, labels, or literal values associated with an entity. "
                "These questions request a specific value rather than a general identity. "
                "Includes measurable attributes (height, size), roles (job, position, function), "
                "statuses (state, condition), and literal values (names, IDs, codes). "
                "Typical forms include 'How tall is X?', 'What role does X have?', "
                "'What is the value of Y?', or 'What is X's status?'."
            ),

            "membership": (
                "Questions about the members, parts, or elements contained within a whole or collection. "
                "These questions ask what items, components, or individuals are included in or belong to something. "
                "Typical forms include 'What items are included in X?', 'What parts does X contain?', "
                "'Which elements are in X?', or 'Who are the members of X?'."
            ),

            "comparative": (
                "Questions comparing two or more entities along some dimension or property. "
                "These questions ask which entity has more or less of a property, or which is superior "
                "in some respect. They involve comparative adjectives or explicit comparison structures "
                "such as 'bigger than', 'faster than', 'more reliable than', or 'Which is better, X or Y'."
            ),

            "quantification": (
                "Questions about counts, quantities, or amounts. "
                "These questions ask how many items exist, how much of something there is, or whether "
                "any instances are present. Typical cues include 'How many', 'How much', "
                "'Are there any', or 'What is the number of'."
            ),

            "existential": (
                "Questions about the existence or presence of entities. "
                "These questions ask whether something exists at all, or whether any instances of a class exist. "
                "Typical cues include 'Does X exist', 'Is there a Y', or 'Are there any X'."
            ),

            "unknown": (
                "Fallback category for ambiguous, malformed, or open-ended explanatory requests that do not "
                "fit any other question type. Includes requests for explanation, interpretation, or elaboration "
                "without a clear semantic target, such as 'Explain X' or 'What does this mean'."
            )
        }

        question_types_list = list(self.question_types_dict.keys())
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
        question_types_and_descriptions = "\n".join([f"- `{qt}`: {desc}" for qt, desc in self.question_types_dict.items()])

        user_msg = f"""
                ### Goal
                Classify the Atomic Input into exactly one of the allowed question types.

                ### Allowed Question Types
                {question_types_and_descriptions}

                ### Atomic Input
                {chunk_text}
                """

        return self.generate_with_schema(user_msg)
