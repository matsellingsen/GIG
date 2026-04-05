from .agents.ontology_construction.class_extraction_agent import ClassExtractionAgent
from .agents.ontology_construction.hierarchical_axiom_extraction_agent import HierarchicalAxiomExtractionAgent
from .agents.ontology_construction.linear_axiom_extraction_agent import LinearAxiomExtractionAgent
from .agents.ontology_construction.instance_extraction_agent import InstanceExtractionAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrator for the 4-step ontology extraction pipeline (V5).
    Delegates task execution to specialized agents:
    1.  local_classes      = ClassExtractionAgent.run(...)
    2a. local_hierarchies  = HierarchicalAxiomExtractionAgent.run(...)
    2b. local_relations    = LinearAxiomExtractionAgent.run(...)
    3.  local_inst         = InstanceExtractionAgent.run(...)
    """

    def __init__(self, backend):
        self.backend = backend

        # Initialize agents
        self.class_agent = ClassExtractionAgent(backend=backend)
        self.hierarchy_agent = HierarchicalAxiomExtractionAgent(backend=backend)
        self.relation_agent = LinearAxiomExtractionAgent(backend=backend)
        self.instance_agent = InstanceExtractionAgent(backend=backend)

        # Global Knowledge Base ("The Accumulator")
        self.kb = {
            "classes": self.class_agent.seed_classes.copy(),    # Dict[str, str]: "ClassName" -> "Description"
            "axioms": self.class_agent.seed_axioms.copy(),      # List of schema axioms
            "instances": []                                     # List of instance assertions
        }
        self.extraction_status = {
            "chunk_index": [],
            "class_failed": [],
            "hierarchy_failed": [],
            "relation_failed": [],
            "instance_failed": []
        }

        self.prompts = {
                "class_extraction": [],
                "hierarchy_extraction": [],
                "relation_extraction": [],
                "instance_population": []
            }
        
    def run_pipeline(self, chunks: list, run_chunks: int = None) -> tuple:
        
        limit = run_chunks if run_chunks else len(chunks)
        print(f"Starting V5 Pipeline with {limit} chunks...")

        for i, chunk in enumerate(chunks):
            if i >= limit:
                break

            chunk_text = chunk["chunk_text_clean"]
            print(f"\n--- Processing Chunk {i+1}/{limit} ---")
            self.extraction_status["chunk_index"].append(i)
            
            # --- 1. Class Extraction (TBox Concepts) ---
            local_classes, class_prompt = self.class_agent.run(chunk_text)
            self.prompts["class_extraction"].append(class_prompt)

            if local_classes:
                print(f"[ClassAgent] Found {len(local_classes)} new classes.")
                for item in local_classes:
                    if isinstance(item, dict) and "class" in item and "desc" in item:
                        cls_name = item["class"]
                        if cls_name not in self.kb["classes"]:
                            self.kb["classes"][cls_name] = item["desc"]
                self.extraction_status["class_failed"].append(False)
            else:
                print("[ClassAgent] No new classes found.")
                self.extraction_status["class_failed"].append(True)

            # --- 2a. Hierarchical Axiom Extraction (TBox Taxonomy) ---
            local_hierarchical_axioms, hierarchy_prompt = self.hierarchy_agent.run(chunk_text, local_existing_classes=local_classes)
            self.prompts["hierarchy_extraction"].append(hierarchy_prompt)

            if local_hierarchical_axioms:
                print(f"[HierarchyAgent] Found {len(local_hierarchical_axioms)} hierarchical axioms.")
                self.kb["axioms"].extend(local_hierarchical_axioms)
                self.extraction_status["hierarchy_failed"].append(False)
            else:
                print("[HierarchyAgent] No hierarchical axioms found.")
                self.extraction_status["hierarchy_failed"].append(True)

            # --- 2b. Linear Axiom Extraction (TBox Properties & Links) ---
            local_linear_axioms, relation_prompt = self.relation_agent.run(chunk_text, local_existing_classes=local_classes)
            self.prompts["relation_extraction"].append(relation_prompt)

            if local_linear_axioms:
                print(f"[RelationAgent] Found {len(local_linear_axioms)} linear relations.")
                self.kb["axioms"].extend(local_linear_axioms)
                self.extraction_status["relation_failed"].append(False)
            else:
                print("[RelationAgent] No linear relations found.")
                self.extraction_status["relation_failed"].append(True)

            # --- 3. Instance Population (ABox) ---
            # Pass ONLY the linear axioms so it looks for property/attribute links correctly
            local_instances, instance_prompt = self.instance_agent.run(chunk_text, local_classes=local_classes, local_axioms=local_linear_axioms)
            self.prompts["instance_population"].append(instance_prompt)

            if local_instances:
                print(f"[InstanceAgent] Extracted {len(local_instances)} assertions/instances.")
                self.kb["instances"].extend(local_instances)
                self.extraction_status["instance_failed"].append(False)
            else:
                print("[InstanceAgent] No instances found.")
                self.extraction_status["instance_failed"].append(True)

        # Final Formatting for Output
        final_result = {
            "ontology": {
                "classes": [{"id": k, "description": v} for k, v in self.kb["classes"].items()],
                "axioms": self.kb["axioms"]
            },
            "instances": self.kb["instances"]
        }
        
        return final_result, self.extraction_status, self.prompts