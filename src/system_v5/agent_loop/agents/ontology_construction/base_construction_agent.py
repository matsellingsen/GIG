"""Superclass Base agent for ontology construction."""
import json
from ..agent import Agent
from tools.structure_prompt import structure_prompt
from tools.base_ontology.load_base_ontology import load_classes, load_axioms

class BaseConstructionAgent(Agent):
    """
    Base agent for ontology construction tasks.
    Enforces a specific JSON schema on the backend's output.
    """

    def __init__(self, backend, system_prompt: str = None, json_schema: dict = None):
        self.backend = backend
        self.system_prompt = system_prompt
        # Some agents have static schemas (ClassExtractor), others dynamic (AxiomExtractor)
        self.json_schema = json_schema 
        self.seed_classes = load_classes()
        self.seed_axioms = load_axioms()

    def generate_with_schema(self, user_msg: str, schema: dict = None) -> list:
        """
        Generates a response constrained by the provided schema.
        If no schema is provided here, it falls back to self.json_schema.
        """
        target_schema = schema if schema else self.json_schema
        if not target_schema:
            raise ValueError("No JSON schema provided for constrained generation.")

        # 1. Structure the prompt (Model-specific formatting)
        # We assume 'phi-4' or similar model type for structure_prompt
        full_prompt = structure_prompt("phi-npu-openvino", self.system_prompt, user_msg)

        # 2. Call Backend with Constraint
        # The backend natively supports 'json_schema' arg now.
        response_str = self.backend.generate(full_prompt, json_schema=target_schema)

        # 3. Parse, Repair, and Deduplicate
        data = self._repair_and_clean_json(response_str)
        return data, full_prompt

    def _repair_and_clean_json(self, text: str):
        """
        Attempts to parse JSON, repairing common model errors:
        1. Truncated output (missing closing brackets).
        2. Looping/Repetition (deduplication).
        """
        parsed_data = None
        # Strategy 1: Direct Load
        try:
            parsed_data = json.loads(text)
        except json.JSONDecodeError:
            pass
        
        print()
        # Strategy 2: Find largest external brackets
        if parsed_data is None:
            text = text.strip()
            # Try to find the outermost valid Structure
            # Object
            if text.startswith('{'):
                # Try to find the last '}'
                last_brace = text.rfind('}')
                if last_brace != -1:
                    try:
                        parsed_data = json.loads(text[:last_brace+1])
                    except:
                        pass
            # Array
            elif text.startswith('['):
                last_brace = text.rfind(']')
                if last_brace != -1:
                    try:
                        parsed_data = json.loads(text[:last_brace+1])
                    except:
                        pass

        # Strategy 3: Aggressive Truncation Repair (if Strategy 2 failed)
        # If the model looped and got cut off, the JSON is invalid. 
        # We can try to repair by appending brackets until it works, 
        # OR we can assume it's a list and we only want the valid prefix items.
        
        if parsed_data is None:
            print("[JSON Repair] Standard parsing failed. Attempting aggressive repair...")
            # If it looks like a list that got cut off: `[{"a":1}, {"b":2}, {"c":`
            # We can find the last valid `},` or `}` and close it.
            last_valid_object_end = text.rfind('}')
            if last_valid_object_end != -1:
                # Try closing as array
                candidate = text[:last_valid_object_end+1] + "]"
                try:
                    parsed_data = json.loads(candidate)
                    print("[JSON Repair] Successfully repaired truncated list.")
                except:
                    # Try closing as object (if it was an object)
                    candidate = text[:last_valid_object_end+1] + "}"
                    try:
                        parsed_data = json.loads(candidate)
                        print("[JSON Repair] Successfully repaired truncated object.")
                    except:
                        pass

        if parsed_data is None:
            print(f"[JSON Repair] Failed completely. Raw text: {text}")
            return [] if text.startswith('[') else {}

        # 4. Deduplication
        return self._deduplicate(parsed_data)

    def _deduplicate(self, data):
        """Recursively removes duplicates from lists."""
        if isinstance(data, list):
            unique_items = []
            seen_hashes = set()
            for item in data:
                # Recursively clean children first
                cleaned_item = self._deduplicate(item)
                
                # Create a hashable representation for deduplication
                # (Serialize to JSON string with sorted keys)
                item_hash = json.dumps(cleaned_item, sort_keys=True)
                
                if item_hash not in seen_hashes:
                    seen_hashes.add(item_hash)
                    unique_items.append(cleaned_item)
            return unique_items
            
        elif isinstance(data, dict):
            # For the hierarchical extraction schema (entities list inside object)
            clean_dict = {}
            for k, v in data.items():
                clean_dict[k] = self._deduplicate(v)
            return clean_dict
            
        return data

    def run(self, prompt: str) -> str:
        """Default run method."""
        data, _ = self.generate_with_schema(prompt)
        return data