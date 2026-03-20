
from tools.structure_prompt import structure_prompt
from tools.load_prompt import load_prompt
import json
import re

class Orchestrator_v2:
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

    def run_pipeline(self, chunks, run_chunks=3) -> tuple:
        """
        Runs the full 3-agent pipeline over a list of TEXT chunks.
        IMPLEMENTATION: STATELESS (Isolated Chunks)
        Each chunk is processed independently. Classes/Axioms found in previous chunks 
        are NOT fed into the next chunk to prevent Model Collapse/Hallucination.
        Results are aggregated at the end.
        """
        print(f"Starting pipeline with {len(chunks)} chunks (Mode: Stateless/Isolated)...")

        for i, chunk in enumerate(chunks):
            print(f"\n--- Processing Chunk {i+1}/{len(chunks)} ---")

            chunk_text = chunk["chunk_text_clean"]  # Use the cleaned text for processing
            
            # 1. Class Extraction (Local)
            # Returns a list of dicts: [{"class": "Name", "desc": "..."}]
            local_classes = self._run_class_extractor(chunk_text)
            
            # Update Global KB with new findings (for final output), but DON'T feed back to prompt yet
            if local_classes:
               for item in local_classes:
                   if list(item.keys()) == ["class", "desc"]: # Basic validation
                       cls_name = item["class"]
                       self.kb["classes"][cls_name] = item["desc"]
                
            # 2. Axiom Extraction (Local)
            # Uses ONLY the classes found in THIS chunk + Seed Classes (to ground it)
            local_axioms = self._run_axiom_extractor(chunk_text, local_classes)
            
            if local_axioms:
                self.kb["axioms"].extend(local_axioms)

            # 3. Instance Population (Local)
            # Uses ONLY the classes and axioms found in THIS chunk
            self._run_instance_populator(chunk_text, local_classes, local_axioms)

            if i == run_chunks - 1:
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

    def _run_class_extractor(self, text: str) -> list:
        """
        Agent 1: Extracts potential classes from the text.
        RETURNS local_classes list found in this chunk.
        Does NOT look at global history to prevent drift.
        """
        # OPTION: We can pass Seed Classes if we want consistency, or nothing.
        # Let's pass Seed Classes so it doesn't reinvent "Person" every time, but NOT the full history.
        seed_classes_str = "Seed Classes (for reference only):\n- Person\n- Organization\n- Location\n- Event\n- PhysicalObject"
        
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system\\prompts\\system\\agents\\class-extractor.txt")
        
        user_msg = (
            f"Source Text:\n{text}\n\n"
            f"{seed_classes_str}\n\n"
            "Goal: Extract classes found in this text. If a concept matches a Seed Class, use that name. Return a JSON list: [{\"class\": \"Name\", \"desc\": \"Description\"}]."
        )

        full_prompt = structure_prompt(self.model_type, system_prompt, user_msg)
        self.prompts.append(full_prompt) #store the prompt for debugging and analysis.
        print("Running Class Extractor (Stateless)...")
        # print(full_prompt) 
        response = self.agent.run(full_prompt)

        # print("Raw response from Class Extractor:")
        # print(response)
        
        # Robust parsing using last valid list
        new_class_list = self._extract_last_json(response)
        
        if new_class_list:
            print(f"[Agent 1] Found {len(new_class_list)} classes in this chunk.")
            return new_class_list
        else:
            print("[Agent 1] No valid JSON list found in response.")
            return []


    def _run_axiom_extractor(self, text: str, local_classes: list) -> list:
        """
        Agent 2: Extracts schema axioms.
        Uses ONLY the local_classes found in this chunk.
        """
        if not local_classes:
            print("[Agent 2] No local classes to process.")
            return []

        # Convert list of dicts to string for prompt
        # We assume local_classes is [{"class": "Name", "desc": "Desc"}, ...]
        classes_str = "\n".join([f"- {item.get('class', 'Unknown')}: {item.get('desc', '')}" for item in local_classes])
        
        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system\\prompts\\system\\agents\\axiom-extractor.txt")

        user_msg = (
            f"Source Text:\n{text}\n\n"
            f"Relevant Classes:\n{classes_str}\n\n"
            "Goal: Extract Subclass hierarchies and Object/Data properties for the Relevant Classes."
        )

        full_prompt = structure_prompt(self.model_type, system_prompt, user_msg)
        self.prompts.append(full_prompt)
        print("Running Axiom Extractor (Stateless)...")
        response = self.agent.run(full_prompt)
        
        # Robust parsing using last valid list
        new_data = self._extract_last_json(response)
        
        if new_data:
            print(f"[Agent 2] Found {len(new_data)} axioms in this chunk.")
            return new_data
        else:
            print("[Agent 2] No valid JSON list found in response.")
            return []

    def _run_instance_populator(self, text: str, local_classes: list, local_axioms: list):
        """
        Agent 3: Populates instances.
        Uses ONLY the local schema (classes + axioms).
        Updates self.kb['instances'] (Aggregation is safe).
        """
        if not local_classes:
            return 

        # Context summary
        classes_str = "\n".join([f"- {item.get('class')}" for item in local_classes])
        axioms_str = json.dumps(local_axioms, indent=2) if local_axioms else "[]"
        
        schema_context = (
            f"## Classes\n{classes_str}\n\n"
            f"## Axioms\n{axioms_str}"
        )

        system_prompt = load_prompt("C:\\Users\\matse\\gig\\src\\system\\prompts\\system\\agents\\instance-populator.txt")

        user_msg = (
            f"Source Text:\n{text}\n\n"
            f"Local Ontology Schema:\n{schema_context}\n\n"
            "Goal: Extract named individuals and assign them to the Schema Classes."
        )

        full_prompt = structure_prompt(self.model_type, system_prompt, user_msg)
        self.prompts.append(full_prompt)
        print("Running Instance Populator (Stateless)...")

        response = self.agent.run(full_prompt)

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