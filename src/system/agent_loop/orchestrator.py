
from tools.structure_prompt import structure_prompt
from tools.load_prompt import load_prompt
import json
import re

class Orchestrator:
    """
    Orchestrator for the 3-step ontology extraction pipeline:
    1. Class Extractor (TBox)
    2. Axiom Extractor (Schema Relationships)
    3. Instance Populator (ABox)
    """

    def __init__(self, agent, model_type: str):
        self.agent = agent
        self.model_type = model_type
        
        # Pre-seed the Global Knowledge Base with some common-sense classes
        # Store as Dict[ClassName, Description] to prevent semantic drift.
        seed_classes = {
            "Person": "An individual human being (e.g., patient, researcher, employee).",
            "Organization": "A social group acting as a single entity (e.g., company, hospital, university).",
            "PhysicalObject": "A tangible object (e.g., sensor, device, equipment).", 
            "Software": "A computer program or application (e.g., app, cloud platform).",
            "Event": "An occurrence or activity (e.g., trial, measurement, study).",
            "Location": "A place or position (e.g., clinic, home, body part)."
        }
        self.prompts = [] #list of all prompts used in the pipeline, for debugging and analysis. 
        # Global Knowledge Base ("The Accumulator")
        self.kb = {
            "classes": seed_classes,    # Dict[str, str]: "ClassName" -> "Description"
            "axioms": [],               # List of schema axioms
            "instances": []             # List of instance assertions
        }

    def run_pipeline(self, chunks):
        """
        Runs the full 3-agent pipeline over a list of TEXT chunks.
        """
        print(f"Starting pipeline with {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            print(f"\n--- Processing Chunk {i+1}/{len(chunks)} ---")

            chunk_text = chunk["chunk_text_clean"]  # Use the cleaned text for processing
            # 1. Class Extraction
            self._run_class_extractor(chunk_text)
            
            # 2. Axiom Extraction (Schema Building)
            self._run_axiom_extractor(chunk_text)
            
            # 3. Instance Population
            self._run_instance_populator(chunk_text)

            if i == 0:
                break  # For testing, we run only the first chunk through the pipeline. Remove this line for full run.

        # Simplify dict back to list format for final JSON serialization
        final_result = {
            "ontology": {
                # Return list of objects {"id": name, "description": desc}
                "classes": [{"id": k, "description": v} for k, v in self.kb["classes"].items()],
                "axioms": self.kb["axioms"]
            },
            "instances": self.kb["instances"]
        }
        return final_result, self.prompts #return the final result and the list of prompts for analysis.

    def _extract_last_json(self, response: str) -> list:
        """
        Extracts the LAST valid JSON list from the response.
        Useful when the model 'thinks out loud' and refines its answer.
        """
        # Finds all JSON-like list blocks
        matches = re.findall(r'\[.*?\]', response, re.DOTALL)
        
        #print last match for debugging
        if matches:
            print("Last JSON match:")
            print("*****************")
            print(matches[-1])
            print("*****************")
        # Iterate backwards to find the last valid JSON list
        for match in reversed(matches):
            try:
                data = json.loads(match)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                continue
        return None

    def _run_class_extractor(self, text: str):
        """
        Agent 1: Extracts potential classes from the text.
        Updates self.kb['classes'].
        """
        # Format existing classes as "Name: Description" for the prompt
        existing_classes_str = "\n".join([f"- {k}: {v}" for k, v in self.kb["classes"].items()])
        
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system\\prompts\\system\\agents\\class-extractor.txt")
        
        user_msg = (
            f"Source Text:\n{text}\n\n"
            f"Existing Classes:\n{existing_classes_str}\n\n"
            "Goal: Extract NEW classes found in this text. Return a JSON list of objects: [{\"class\": \"Name\", \"desc\": \"Description\"}]."
        )

        full_prompt = structure_prompt(self.model_type, system_prompt, user_msg)
        self.prompts.append(full_prompt) #store the prompt for debugging and analysis.
        print("Running Class Extractor with prompt:")
        print(full_prompt)
        print("-------------")
        response = self.agent.run(full_prompt)

        print("Raw response from Class Extractor:")
        print(response)
        print("-------------")
        
        # Robust parsing using last valid list
        new_class_list = self._extract_last_json(response)
        
        if new_class_list:
            count = 0
            for item in new_class_list:
                # Validate item structure
                if isinstance(item, dict) and "class" in item and "desc" in item:
                    cls_name = item["class"]
                    cls_desc = item["desc"]
                    
                    # Only add if not already present
                    if cls_name not in self.kb["classes"]:
                        self.kb["classes"][cls_name] = cls_desc
                        count += 1
                        
            print(f"[Agent 1] Found {count} new classes.")
        else:
            print("[Agent 1] No valid JSON list found in response.")


    def _run_axiom_extractor(self, text: str):
        """
        Agent 2: Extracts schema axioms (SubClassOf, properties).
        Uses the FULL TEXT and the Known Classes.
        Updates self.kb['axioms'].
        """
        # Pass just the names to keep context smaller, or names+descriptions if needed.
        # Ideally, descriptions help reasoning about relationships.
        known_classes_str = "\n".join([f"- {k}: {v}" for k, v in self.kb["classes"].items()])
        
        if not self.kb["classes"]:
            return  

        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system\\prompts\\system\\agents\\axiom-extractor.txt")

        user_msg = (
            f"Source Text:\n{text}\n\n"
            f"Known Classes:\n{known_classes_str}\n\n"
            "Goal: Extract 1) Subclass hierarchies, 2) Object properties (relationships), and 3) Data properties (attributes) for the Known Classes."
        )

        full_prompt = structure_prompt(self.model_type, system_prompt, user_msg)
        self.prompts.append(full_prompt) #store the prompt for debugging and analysis.
        print("Running Axiom Extractor with prompt:")
        print(full_prompt)
        print("-------------")
        response = self.agent.run(full_prompt)
        
        print("Raw response from Axiom Extractor:")
        print(response)
        print("-------------")

        # Robust parsing using last valid list
        new_data = self._extract_last_json(response)
        
        if new_data:
            # Simple dedup based on string representation
            start_len = len(self.kb["axioms"])
            current_strs = {json.dumps(x, sort_keys=True) for x in self.kb["axioms"]}
            for item in new_data:
                s = json.dumps(item, sort_keys=True)
                if s not in current_strs:
                    self.kb["axioms"].append(item)
                    current_strs.add(s)
            print(f"[Agent 2] Added {len(self.kb['axioms']) - start_len} new axioms.")
        else:
            print("[Agent 2] No valid JSON list found in response.")

    def _run_instance_populator(self, text: str):
        """
        Agent 3: Populates instances (ABox).
        Uses the FULL TEXT, Known Classes, and Known Axioms.
        Updates self.kb['instances'].
        """
        # Pass the rich Dict (Name -> Desc) or just names if context is tight.
        # Descs help distinguish instance types.
        known_classes_str = "\n".join([f"- {k}: {v}" for k, v in self.kb["classes"].items()])
        
        # Context summary for the prompt
        # We construct a human-readable schema block
        schema_context = (
            f"## Classes\n{known_classes_str}\n\n"
            f"## Axioms\n{json.dumps(self.kb['axioms'], indent=2)}"
        )

        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system\\prompts\\system\\agents\\instance-populator.txt")

        user_msg = (
            f"Source Text:\n{text}\n\n"
            f"Ontology Schema:\n{schema_context}\n\n"
            "Goal: Extract named individuals and assign them to the Schema Classes. Create object/data property assertions using ONLY the provided Schema properties."
        )

        full_prompt = structure_prompt(self.model_type, system_prompt, user_msg)
        self.prompts.append(full_prompt) #store the prompt for debugging and analysis.
        print("Running Instance Populator with prompt:")
        print(full_prompt)
        print("-------------")

        response = self.agent.run(full_prompt)

        print("Raw response from Instance Populator:")
        print(response)
        print("-------------")
        # Robust parsing using last valid list
        new_instances = self._extract_last_json(response)
        
        if new_instances:
            self.kb["instances"].extend(new_instances)
            print(f"[Agent 3] Extracted {len(new_instances)} instances.")
        else:
            print("[Agent 3] No valid JSON list found in response.")

    def run_once(self, system_prompt: str, source_text: str) -> tuple:
        """Legacy Single-step execution."""
        # Compose final prompt
        prompt = structure_prompt(self.model_type, system_prompt, source_text)
        print("Composed prompt for agent:")
        print("-------------")
        print(prompt)
        print("-------------")
        return self.agent.run(prompt), prompt
