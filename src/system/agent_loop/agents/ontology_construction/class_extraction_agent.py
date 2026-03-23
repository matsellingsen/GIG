from .base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class ClassExtractionAgent(BaseConstructionAgent):
    def __init__(self, backend):
        # 1. Define Static Schema
        # We want a list of {"class": "Name", "desc": "Description"}
        class_extraction_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "class": {"type": "string"},
                    "desc": {"type": "string"}
                },
                "required": ["class", "desc"],
                "additionalProperties": False 
            }
        }
        
        # 2. Load Prompt
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system\\prompts\\system\\agents_v3\\class-extraction.txt")
        
        # 3. Initialize Base
        super().__init__(backend=backend,
                         system_prompt=system_prompt, 
                         json_schema=class_extraction_schema)
    
    def run(self, chunk_text: str) -> list:
        """
        Runs the class extraction on a single chunk of text.
        """
        # Add "seed classes" 
        # Format existing classes as "Name: Description" for the prompt
        existing_classes_str = "\n".join([f"- {k}: {v}" for k, v in self.seed_classes.items()])
        user_msg = f"""
                    ### Context
                    {existing_classes_str}

                    ### Text
                    {chunk_text}

                    ### Instruction
                    Extract new classes from the text."""
        
        # Uses the schema defined in __init__
        return self.generate_with_schema(user_msg)