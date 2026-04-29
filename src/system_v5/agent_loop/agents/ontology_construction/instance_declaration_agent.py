from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt

class InstanceDeclarationAgent(BaseOntologyAgent):
    def __init__(self, backend):
        system_prompt = load_prompt(r"c:\Users\matse\gig\src\system_v5\prompts\system\agents\instance-declaration.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, chunk_text: str, local_classes: list) -> tuple:
        if not local_classes:
            return [], [], "Skipped: No local classes available."

        local_class_names = sorted({str(c.get("class")) for c in local_classes})

        declaration_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "class": {"enum": local_class_names if local_class_names else [""]}
                },
                "required": ["id", "class"],
                "additionalProperties": False
            }
        }

        classes_block = "\n".join([f"- {c['class']}: {c['desc']}" for c in local_classes])
        
        user_msg = f"""
                    ### Local Classes
                    {classes_block}

                    ### Source Text
                    {chunk_text}

                    ### Goal
                    Identify real-world named entities, specific objects, or instances in the Source Text and assign them to the appropriate Local Class."""

        raw_extractions, prompt_used = self.generate_with_schema(user_msg, declaration_schema)
        
        # Transform the simplified LLM output into strict ABox assertion formats
        abox_assertions = []
        for ext in (raw_extractions or []):
            if "id" in ext and "class" in ext:
                abox_assertions.append({
                    "type": "DeclareIndividual",
                    "id": ext["id"]
                })
                abox_assertions.append({
                    "type": "ClassAssertion",
                    "individual": ext["id"],
                    "class": ext["class"]
                })
                
        return abox_assertions, raw_extractions, prompt_used
