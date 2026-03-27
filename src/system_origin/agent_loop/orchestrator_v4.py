from .agents.ontology_construction_v4.class_extraction_agent import ClassExtractionAgent
from .agents.ontology_construction_v4.axiom_extraction_agent import AxiomExtractionAgent
from .agents.ontology_construction_v4.instance_extraction_agent import InstanceExtractionAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Orchestrator_v4:
    """
    Orchestrator for the 3-step ontology extraction pipeline (V4).
    Delegates task execution to specialized agents:
    1. local_classes = ClassExtractionAgent.run(chunk)
    2. local_axioms  = AxiomExtractionAgent.run(chunk, local_classes)
    3. local_inst    = InstanceExtractionAgent.run(chunk, local_classes, local_axioms)
    """

    def __init__(self, backend):
        self.backend = backend

        # Initialize agents
        self.class_agent = ClassExtractionAgent(backend=backend)
        self.axiom_agent = AxiomExtractionAgent(backend=backend)
        self.instance_agent = InstanceExtractionAgent(backend=backend)

        # Global Knowledge Base ("The Accumulator")
        # We fetch the seed classes from the agent to initialize our KB
        self.kb = {
            "classes": self.class_agent.seed_classes.copy(),    # Dict[str, str]: "ClassName" -> "Description"
            "axioms": [],               # List of schema axioms
            "instances": []             # List of instance assertions
        }
        self.extraction_status = {
            "chunk_index": [],
            "class_failed": [],
            "axiom_failed": [],
            "instance_failed": []
        }

        self.prompts = {
                "class_extraction": [],
                "axiom_extraction": [],
                "instance_population": []
            }
        
    def run_pipeline(self, chunks: list, run_chunks: int = None) -> tuple:
        
        limit = run_chunks if run_chunks else len(chunks)
        print(f"Starting V4 Pipeline with {limit} chunks...")

        for i, chunk in enumerate(chunks):
            if i >= limit:
                break

            chunk_text = chunk["chunk_text_clean"]
            print(f"\n--- Processing Chunk {i+1}/{limit} ---")
            self.extraction_status["chunk_index"].append(i)
            # 1. Class Extraction (TBox)
            # Returns list of dicts: [{"class": "Name", "desc": "..."}]
            local_classes, class_prompt = self.class_agent.run(chunk_text)
            self.prompts["class_extraction"].append(class_prompt)

            # Update Global KB immediately
            if local_classes:
                print(f"[ClassAgent] Found {len(local_classes)} new classes.")
                for item in local_classes:
                    # Validate item structure before adding
                    if isinstance(item, dict) and "class" in item and "desc" in item:
                        cls_name = item["class"]
                        # Only add if we don't know it (or overwrite if we prefer local definitions)
                        if cls_name not in self.kb["classes"]:
                            self.kb["classes"][cls_name] = item["desc"]
                self.extraction_status["class_failed"].append(False)
                
            else:
                print("[ClassAgent] No new classes found.")
                self.extraction_status["class_failed"].append(True)

            # 2. Axiom Extraction (Schema Relationships)
            # Pass local classes so the agent knows what to link
            local_axioms, axiom_prompt = self.axiom_agent.run(chunk_text, local_existing_classes=local_classes)
            self.prompts["axiom_extraction"].append(axiom_prompt)

            if local_axioms:
                print(f"[AxiomAgent] Found {len(local_axioms)} axioms.")
                self.kb["axioms"].extend(local_axioms)
                self.extraction_status["axiom_failed"].append(False)
            else:
                print("[AxiomAgent] No axioms found.")
                self.extraction_status["axiom_failed"].append(True)

            # 3. Instance Population (ABox)
            # Pass local schema (classes + axioms) so assertions are valid
            local_instances, instance_prompt = self.instance_agent.run(chunk_text, local_classes=local_classes, local_axioms=local_axioms)
            self.prompts["instance_population"].append(instance_prompt)

            if local_instances:
                print(f"[InstanceAgent] Extracted {len(local_instances)} instances.")
                self.kb["instances"].extend(local_instances)
                self.extraction_status["instance_failed"].append(False)
            else:
                print("[InstanceAgent] No instances found.")
                self.extraction_status["instance_failed"].append(True)


        # Final Formatting for Output
        final_result = {
            "ontology": {
                # Convert Dict back to List for JSON output
                "classes": [{"id": k, "description": v} for k, v in self.kb["classes"].items()],
                "axioms": self.kb["axioms"]
            },
            "instances": self.kb["instances"]
        }
        
        return final_result, self.extraction_status, self.prompts