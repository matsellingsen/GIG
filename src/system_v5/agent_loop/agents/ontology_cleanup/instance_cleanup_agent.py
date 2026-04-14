from ..ontology_construction.base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class InstanceCleanupAgent(BaseConstructionAgent):
    def __init__(self, backend):
        # Load the corresponding system prompt
        system_prompt = load_prompt(r"c:\Users\matse\gig\src\system_v5\prompts\system\agents\instance-cleanup.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, inst: dict) -> tuple:
        """
        Runs the LLM on a list of instances to determine which should be kept or removed.
        Args:
            instances (list): List of dicts, each with keys: 'id', 'name', 'class', 'class_description', 'properties'.
        Returns:
            list: List of resolution dicts for each instance.
            str: The prompt used.
        """
        if not inst:
            return [], "Skipped: No instance provided."

        # Format input for the model
        instance_block = f"- Name: {inst['name']}\n  Class: {inst['class']}\n  Class Description: {inst.get('class_description', 'No description.')}\n  Properties: {', '.join(inst.get('properties', [])) if inst.get('properties') else 'None'}"
        print("--------------------------------")  
        print(f"\n[DEBUG] Formatted instance block for LLM:\n{instance_block}\n") # Debug print to verify formatting
        user_msg = (
            f"### Candidate Instances\n{instance_block}\n\n"
            "### Goal\nFor each instance, decide if it should be kept in the ontology or removed as irrelevant, artifactual, or misclassified. "
            "Base your decision on the instance's name, its class, class description, and properties. "
            "Remove if there is a strong, well-justified reason to believe it is an artifact or misclassification—even if not 100% certain. "
            "If the case is ambiguous or borderline, prefer 'keep', but you may choose 'remove' if you can provide a clear, specific explanation. "
            "Return a decision for EVERY instance: 'keep' (valid instance) or 'remove' (should be deleted), and always provide your reasoning."
        )

        

        # JSON schema for OpenVINO GenAI structured generation
        resolution_schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "action": {"type": "string", "enum": ["keep", "remove"]}
        },
        "required": ["action", "reasoning"],
        "additionalProperties": False
    }

        # Query the backend
        raw_extractions, prompt_used = self.generate_with_schema(user_msg, resolution_schema)

        # Parse outputs
        resolution = {
            "action": raw_extractions.get("action", "keep"),  # Default to 'keep' if not specified
            "reasoning": raw_extractions.get("reasoning", "No reasoning provided.")
        }
        
        return resolution, prompt_used
