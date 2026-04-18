import sys
sys.path.append("C:\\Users\\matse\\gig\\src\\system_v5") # Adjust this path to point to the root of your project if needed

import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from agent_loop.agents.ontology_cleanup.equivalency_agent import EquivalencyAgent
from backends import load_backend
import argparse

def load_classes_from_ontology(ontology: dict, num_chunks: int = None) -> list:
    """
    Extracts classes from a given ontology structure.
    
    Args:
        ontology (dict): The loaded ontology dictionary.
        num_chunks (int): The number of chunks to consider. If None, considers all chunks. This is useful for testing with a subset of the ontology.
        
    Returns:
        list: A list of class dictionaries, e.g., [{"id": "ClassA", "description": "..."}]
    """
    # check if ontology was loaded successfully
    if "ontology_tbox" not in ontology or "classes" not in ontology["ontology_tbox"]:
        print("Error: Ontology structure is invalid. 'ontology_tbox' or 'classes' key is missing.")
        return []
    print("Ontology loaded successfully. Extracting classes...")

    classes_to_process = ontology["ontology_tbox"]["classes"]
    if num_chunks is not None:
        # Filter classes to only include those from the specified number of chunks
        classes_to_process = [c for c in classes_to_process if c.get("chunk_ids") and int(c["chunk_ids"][0]) < num_chunks]

    classes = []
    for cls in classes_to_process:
        if cls.get("chunk_ids"): # avoiding the base ontology classes which have empty chunk_id
            cls_id = cls.get("id", "").strip()
            cls_desc = cls.get("description", "").strip()
            if cls_id:
                classes.append({"id": cls_id, "description": cls_desc})
    
    print(f"Extracted {len(classes)} classes from the ontology.")
    return classes

def cluster_classes(classes: list, threshold: float = 0.30) -> list:
    """
    Clusters ontology classes based on the semantic similarity of their names and descriptions.
    
    Args:
        classes (list): A list of dictionaries, e.g., [{"id": "ClassA", "description": "..."}]
        threshold (float): Distance threshold for clustering (1.0 - cosine_similarity). 
                           Lower = stricter clusters (e.g., 0.30 = 70% similarity).
        
    Returns:
        list: A list of clusters, where each cluster is a list of class dictionaries.
    """
    if not classes:
        return []
        
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    # all-MiniLM-L6-v2 is an extremely fast, tiny 80MB model specifically designed for semantic clustering
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Embedding {len(classes)} classes...")
    # Combine the ID and description to get a rich semantic representation
    texts_to_embed = [f"{c.get('id', '')}: {c.get('description', '')}" for c in classes]
    embeddings = model.encode(texts_to_embed)
    
    print("Clustering classes...")
    # We use Agglomerative Clustering because we don't know the final number of clusters upfront
    clustering_model = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=threshold, 
        metric='cosine', 
        linkage='average'
    )
    clustering_model.fit(embeddings)
    
    # Group classes by their assigned cluster labels
    clusters = {}
    for idx, label in enumerate(clustering_model.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(classes[idx])
        
    print(f"Formed {len(clusters)} clusters out of {len(classes)} classes.")
    return list(clusters.values())

def apply_resolutions(ontology: dict, resolutions: list):
    """
    Applies the agent's resolutions to the ontology.
    Classes marked 'equivalent' are linked to their canonical target via TBox equivalentClass axioms.
    """
    equivalent_mappings = {r["original_id"]: r["target_id"] for r in resolutions if r.get("action") == "equivalent" and r.get("target_id")}

    print(f"\nApplying resolutions: {len(equivalent_mappings)} equivalence mappings.")
    
    print("--- Resolution Summary ---")
    if equivalent_mappings:
        print("Equivalent Mappings:")
        for orig, target in sorted(equivalent_mappings.items()):
            print(f"  - {orig} -> {target}")
    print("--------------------------\n")

    # 1. Add equivalentClass axioms to TBox
    if "ontology_tbox" in ontology and "classes" in ontology["ontology_tbox"]:
        axioms = ontology["ontology_tbox"].setdefault("axioms", [])
        has_equiv = any(a.get("type") == "equivalentClass" for a in axioms)
        
        if not has_equiv and equivalent_mappings:
            axioms.append({
                "type": "equivalentClass",
                "domains": ["Entity"],
                "ranges": ["Entity"],
                "description": "States that two classes are semantically identical and share the exact same instances."
            })
            
        for orig, target in equivalent_mappings.items():
            if target:
                axioms.append({
                    "type": "equivalentClass",
                    "domains": [orig],
                    "ranges": [target],
                    "description": f"Generated by Entity Resolution: '{orig}' is semantically equivalent to '{target}'."
                })

class ClassResolver:
    def __init__(self, backend, num_chunks: int = None):
        self.backend = backend
        self.num_chunks = num_chunks

    def resolve(self, ontology: dict) -> dict:
        # Load classes from the ontology and cluster them
        classes = load_classes_from_ontology(ontology, num_chunks=self.num_chunks)
        print(f"Loaded {len(classes)} classes from ontology for all chunks.")
        
        clustered_classes = cluster_classes(classes)
        actual_clusters = [c for c in clustered_classes if len(c) > 1]
        print(f"Actual clusters (with more than 1 class): {len(actual_clusters)}")
        
        # Resolve clusters through the Equivalency agent
        equiv_agent = EquivalencyAgent(backend=self.backend)
        
        print("Resolving clusters with the Equivalency Agent...")
        all_resolutions = []
        
        for cluster in actual_clusters:
            print(f"--- Cluster {actual_clusters.index(cluster)+1}/{len(actual_clusters)} ---")
            print(f"Cluster contains {len(cluster)} classes:")
            for c in cluster:
                print(f"  - {c['id']}: {c['description']}")
            print("-----------------------------")
            print("-----------------------------")
            # 1. Equivalency Pass
            equiv_res, _ = equiv_agent.run(cluster)
            all_resolutions.extend([r for r in equiv_res if r.get("action") == "equivalent"])
            
            # print ratio of classes in cluster found equivalent
            equiv_count = sum(1 for r in equiv_res if r.get("action") == "equivalent")
            print(f"Equivalency Agent: Marked {equiv_count}/{len(cluster)} classes as equivalent.")
            if equiv_count > 0:
                print("Classes marked as equivalent:")
                for r in equiv_res:
                    if r.get("action") == "equivalent":
                        cls = next((c for c in cluster if c["id"] == r["original_id"]), None)
                        target = r.get("target_id", "None")
                        desc = cls["description"] if cls else "No description found"
                        print(f"  - {r['original_id']} (equivalent to {target}): {desc}")
                print("------------------------------")
                    
        print(f"Pipeline finished processing {len(actual_clusters)} test clusters. Found {len(all_resolutions)} total action mappings.")
        
        # Pass all resolutions at once to the application handler
        apply_resolutions(ontology, all_resolutions)
        
        return ontology

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="phi-npu-openvino") # decides which backend to use here.
    parser.add_argument("--ontology_path", default="C:\\Users\\matse\\gig\\src\\system_v5\\intermediate_results\\phi-npu-openvino_ontology_20260406_232546.json", help="Path to the ontology JSON file to resolve.")
    parser.add_argument("--dev", action="store_true", help="Run in development mode (does not save the output file).")
    parser.add_argument("--num_chunks", type=int, default=None, help="Number of chunks to process from the ontology for testing. If None, processes all chunks.")
    args = parser.parse_args()

    backend = load_backend(name=args.backend)
    ontology_path = args.ontology_path

    with open(ontology_path, 'r', encoding='utf-8') as f:
        ontology = json.load(f)

    resolver = ClassResolver(backend=backend, num_chunks=args.num_chunks)
    ontology = resolver.resolve(ontology)

    # Save the updated ontology to a new file (skip saving in dev mode)
    if not args.dev:    
        resolved_path = ontology_path.replace(".json", "_classResolved.json")
        with open(resolved_path, 'w', encoding='utf-8') as f:
            json.dump(ontology, f, indent=4)   
        print(f"Successfully saved resolved ontology to: {resolved_path}")
    else:
        print("Dev mode enabled: Skipping file save.")

if __name__ == "__main__":
    main()