from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt

class InstancePolysemyAgent(BaseOntologyAgent):
    def __init__(self, backend):
        # Load the corresponding system prompt
        system_prompt = load_prompt(r"c:\Users\matse\gig\src\system_v5\prompts\system\agents\ontology_cleanup\instance-polysemy.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, group_key: str, class_groups: dict, tbox_classes: dict = None) -> tuple:
        """
        Runs the LLM on a Polysemy Collision group (instances sharing a name but having different classes).
        Args:
            group_key (str): The normalized identical string name (e.g. "SENS motion")
            class_groups (dict): Mapping of canonical class -> list of original extract IDs
                 e.g. {"Sensor": ["SENS motion"], "FAQ": ["SENS motion"]}
            tbox_classes (dict): Mapping of class names to their ontology descriptions.
        Returns:
            list: List of resolution dictionaries for each class.
            str: The prompt used.
        """
        if not class_groups:
            return [], "Skipped: Empty class groups."

        unique_classes = list(class_groups.keys())

        if len(unique_classes) == 1:
            auto_resolve = [{
                "class": unique_classes[0],
                "action": "merge"
            }]
            return auto_resolve, "Skipped LLM: Only 1 class presented to Polysemy Agent."

        # Format input for the model
        classes_block_lines = []
        for cls, ids in class_groups.items():
            desc = tbox_classes.get(cls, "No description available.") if tbox_classes else "No description available."
            classes_block_lines.append(f"- Class '{cls}' ({len(ids)} extractions): {desc}")
        classes_block = "\n".join(classes_block_lines)
        user_msg = (
            f"### Target Entity Name\n'{group_key}'\n\n"
            f"### Assigned Classes and Descriptions\n{classes_block}\n\n"
            "### Goal\nAnalyze the classes and their descriptions assigned to this entity name. "
            "Determine if a single real-world thing can logically belong to this class (based on its description) alongside the others ('merge'), "
            "or if this class represents a completely different thing sharing the same name or a document structure artifact ('reject')."
        )

        # JSON schema for OpenVINO GenAI structured generation
        # Force the SLM to generate an action for every single class
        resolution_schema = {
            "type": "object",
            "properties": {
                "resolutions": {
                    "type": "object",
                    "properties": {
                        cls: {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["merge", "reject"]}
                            },
                            "required": ["action"],
                            "additionalProperties": False
                        } for cls in unique_classes
                    },
                    "required": unique_classes,
                    "additionalProperties": False
                }
            },
            "required": ["resolutions"],
            "additionalProperties": False
        }

        # Query the NPU Backend
        raw_extractions, prompt_used = self.generate_with_schema(user_msg, resolution_schema)
        
        # Parse the outputs safely
        resolutions = []
        if raw_extractions and "resolutions" in raw_extractions:
            parsed_classes = set()
            for cls, res in raw_extractions["resolutions"].items():
                resolutions.append({
                    "class": cls,
                    "action": res.get("action", "reject") # fallback conservative choice
                })
                parsed_classes.add(cls)
            
            # Universal Python Safety Net: If the LLM somehow omitted a class, auto-reject it
            for cls in unique_classes:
                if cls not in parsed_classes:
                    resolutions.append({
                        "class": cls,
                        "action": "reject"
                    })
            
        return resolutions, prompt_used
