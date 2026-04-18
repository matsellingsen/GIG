from ...agent_loop.agents.ontology_construction.class_extraction_agent import ClassExtractionAgent
from ...agent_loop.agents.ontology_construction.hierarchical_base_grounding_agent import HierarchicalBaseGroundingAgent
from ...agent_loop.agents.ontology_construction.hierarchical_local_subclassing_agent import HierarchicalLocalSubclassingAgent
from ...agent_loop.agents.ontology_construction.linear_object_property_agent import LinearObjectPropertyAgent
from ...agent_loop.agents.ontology_construction.linear_data_property_agent import LinearDataPropertyAgent
from ...agent_loop.agents.ontology_construction.instance_declaration_agent import InstanceDeclarationAgent
from ...agent_loop.agents.ontology_construction.instance_object_property_agent import InstanceObjectPropertyAgent
from ...agent_loop.agents.ontology_construction.instance_data_property_agent import InstanceDataPropertyAgent
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
        self.base_grounding_agent = HierarchicalBaseGroundingAgent(backend=backend)
        self.local_subclassing_agent = HierarchicalLocalSubclassingAgent(backend=backend)
        self.object_property_agent = LinearObjectPropertyAgent(backend=backend)
        self.data_property_agent = LinearDataPropertyAgent(backend=backend)
        self.instance_declaration_agent = InstanceDeclarationAgent(backend=backend)
        self.instance_object_property_agent = InstanceObjectPropertyAgent(backend=backend)
        self.instance_data_property_agent = InstanceDataPropertyAgent(backend=backend)

        # Global Knowledge Base ("The Accumulator")
        self.kb = {
            "classes": self.class_agent.seed_classes.copy(),    # Dict[str, str]: "ClassName" -> "Description"
            "class_chunk_ids": {},                              # Dict[str, List[int]]: "ClassName" -> [chunk_ids]
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
                        
                        # Tag chunk ID for merging
                        if cls_name not in self.kb["class_chunk_ids"]:
                            self.kb["class_chunk_ids"][cls_name] = []
                        if i not in self.kb["class_chunk_ids"][cls_name]:
                            self.kb["class_chunk_ids"][cls_name].append(i)
                self.extraction_status["class_failed"].append(False)
            else:
                print("[ClassAgent] No new classes found.")
                self.extraction_status["class_failed"].append(True)

            # --- 2a. Hierarchical Axiom Extraction (Base Grounding) ---
            base_hierarchical_axioms, base_hierarchy_prompt = self.base_grounding_agent.run(chunk_text, local_existing_classes=local_classes)
            self.prompts["hierarchy_extraction"].append(base_hierarchy_prompt)

            if base_hierarchical_axioms:
                print(f"[BaseGroundingAgent] Found {len(base_hierarchical_axioms)} grounding axioms.")
                for ax in base_hierarchical_axioms: ax["chunk_id"] = i
                self.kb["axioms"].extend(base_hierarchical_axioms)

            # --- 2b. Hierarchical Axiom Extraction (Local Subclassing) ---
            local_hierarchical_axioms, local_hierarchy_prompt = self.local_subclassing_agent.run(chunk_text, local_existing_classes=local_classes)
            self.prompts["hierarchy_extraction"].append(local_hierarchy_prompt)

            if local_hierarchical_axioms:
                print(f"[LocalSubclassingAgent] Found {len(local_hierarchical_axioms)} subclassing axioms.")
                for ax in local_hierarchical_axioms: ax["chunk_id"] = i
                self.kb["axioms"].extend(local_hierarchical_axioms)

            if base_hierarchical_axioms or local_hierarchical_axioms:
                self.extraction_status["hierarchy_failed"].append(False)
            else:
                print("[HierarchyAgents] No hierarchical axioms found.")
                self.extraction_status["hierarchy_failed"].append(True)

            # --- 2c. Linear Axiom Extraction (Object Properties) ---
            local_object_axioms, object_relation_prompt = self.object_property_agent.run(chunk_text, local_existing_classes=local_classes)
            self.prompts["relation_extraction"].append(object_relation_prompt)

            if local_object_axioms:
                print(f"[ObjectPropertyAgent] Found {len(local_object_axioms)} object relations.")
                for ax in local_object_axioms: ax["chunk_id"] = i
                self.kb["axioms"].extend(local_object_axioms)

            # --- 2d. Linear Axiom Extraction (Data Properties) ---
            local_data_axioms, data_relation_prompt = self.data_property_agent.run(chunk_text, local_existing_classes=local_classes)
            self.prompts["relation_extraction"].append(data_relation_prompt)

            if local_data_axioms:
                print(f"[DataPropertyAgent] Found {len(local_data_axioms)} data relations.")
                for ax in local_data_axioms: ax["chunk_id"] = i
                self.kb["axioms"].extend(local_data_axioms)

            if local_object_axioms or local_data_axioms:
                self.extraction_status["relation_failed"].append(False)
            else:
                print("[RelationAgents] No linear relations found.")
                self.extraction_status["relation_failed"].append(True)

            local_linear_axioms = (local_object_axioms or []) + (local_data_axioms or [])

            # --- 3. Instance Population (ABox) ---
            abox_declarations, raw_individuals, decl_prompt = self.instance_declaration_agent.run(chunk_text, local_classes=local_classes)
            self.prompts["instance_population"].append(decl_prompt)

            if abox_declarations:
                print(f"[InstanceDeclarationAgent] Extracted {len(raw_individuals)} individuals ({len(abox_declarations)} assertions).")
                for inst in abox_declarations: inst["chunk_id"] = i
                self.kb["instances"].extend(abox_declarations)
                self.extraction_status["instance_failed"].append(False)

                # --- 3b. Instance Population (Object Properties) ---
                if len(raw_individuals) >= 2:
                    abox_obj_props, obj_prompt = self.instance_object_property_agent.run(chunk_text, declared_individuals=raw_individuals, object_axioms=local_object_axioms)
                    self.prompts["instance_population"].append(obj_prompt)

                    if abox_obj_props:
                        print(f"[InstanceObjectPropertyAgent] Found {len(abox_obj_props)} object assertions.")
                        for inst in abox_obj_props: inst["chunk_id"] = i
                        self.kb["instances"].extend(abox_obj_props)
                    else:
                        print("[InstanceObjectPropertyAgent] No object assertions found.")
                else:
                    print(f"[InstanceObjectPropertyAgent] Skipping object properties (only {len(raw_individuals)} individuals found, need >= 2).")

                # --- 3c. Instance Population (Data Properties) ---
                abox_data_props, data_prompt = self.instance_data_property_agent.run(chunk_text, declared_individuals=raw_individuals, data_axioms=local_data_axioms)
                self.prompts["instance_population"].append(data_prompt)

                if abox_data_props:
                    print(f"[InstanceDataPropertyAgent] Found {len(abox_data_props)} data assertions.")
                    for inst in abox_data_props: inst["chunk_id"] = i
                    self.kb["instances"].extend(abox_data_props)
                else:
                    print("[InstanceDataPropertyAgent] No data assertions found.")

            else:
                print("[InstanceDeclarationAgent] No individuals found.")
                self.extraction_status["instance_failed"].append(True)

        # Final Formatting for Output
        final_result = {
            "ontology": {
                "classes": [
                    {
                        "id": k, 
                        "description": v, 
                        "chunk_ids": self.kb["class_chunk_ids"].get(k, []) # Attaches the tracked chunk IDs
                    } for k, v in self.kb["classes"].items()
                ],
                "axioms": self.kb["axioms"]
            },
            "instances": self.kb["instances"]
        }
        
        return final_result, self.extraction_status, self.prompts