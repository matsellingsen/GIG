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
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\class-extraction.txt")
        
        # 3. Initialize Base
        super().__init__(backend=backend,
                         system_prompt=system_prompt, 
                         json_schema=class_extraction_schema)
    
    def run(self, chunk_text: str) -> list:
        """
        Runs the class extraction on a single chunk of text.
        """
        user_msg = f"""
                    ### Existing Classes
                    {self.seed_classes}

                    ### Source Text
                    {chunk_text}

                    ### Goal
                    Analyze the provided Source Text to identify **new** classes that are not present in the list of Existing Classes."""
        
        # Uses the schema defined in __init__
        return self.generate_with_schema(user_msg)