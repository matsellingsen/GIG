import os
import json
import argparse
import re
import sys
from collections import defaultdict

sys.path.append(r"C:\Users\matse\gig\src\system_v5")
from backends import load_backend
from agent_loop.agents.ontology_cleanup.instance_polysemy_agent import InstancePolysemyAgent
from agent_loop.agents.ontology_cleanup.semantic_cluster_agent import SemanticClusterAgent

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import hstack

class InstanceResolver:
    def __init__(self, backend):
        self.backend = backend

    def __resolve_class(self, raw_class: str, equivalence_map: dict) -> str:
        """Helper to resolve a class to its canonical TBox equivalent."""
        c = raw_class
        visited = set()
        while c in equivalence_map and c not in visited:
            visited.add(c)
            c = equivalence_map[c]
        return c

    def _prune_abox(self, abox: list, class_actions: dict, equivalence_map: dict) -> tuple[list, int]:
        """
        Safely removes rejected classes and all their associated properties based on Phase 2 actions.
        Uses (instance_id, chunk_id) to ensure exact local sub-graph deletion without affecting 
        homonyms in other document chunks.
        """
        rejected_nodes = set()
        # Pass 1: Identify all specific (orig_id, chunk_id) nodes to be rejected
        for assertion in abox:
            if assertion.get("type") == "ClassAssertion":
                orig_id = assertion.get("individual", "")
                raw_class = assertion.get("class", "")
                resolved_class = self.__resolve_class(raw_class, equivalence_map)
                
                if class_actions.get((orig_id, resolved_class), "merge") == "reject":
                    rejected_nodes.add((orig_id, assertion.get("chunk_id")))
                    
        # Pass 2: Filter the ABox
        new_abox = []
        deleted_count = 0
        
        for assertion in abox:
            chunk_id = assertion.get("chunk_id")
            a_type = assertion.get("type")
            is_dead = False
            
            if a_type == "ClassAssertion":
                is_dead = (assertion.get("individual", ""), chunk_id) in rejected_nodes
            elif a_type == "DeclareIndividual":
                is_dead = (assertion.get("id", ""), chunk_id) in rejected_nodes
            elif a_type == "DataPropertyAssertion":
                is_dead = (assertion.get("subject", ""), chunk_id) in rejected_nodes
            elif a_type == "ObjectPropertyAssertion":
                is_dead = ((assertion.get("subject", ""), chunk_id) in rejected_nodes or
                           (assertion.get("object", ""), chunk_id) in rejected_nodes)
                           
            if is_dead:
                deleted_count += 1
            else:
                new_abox.append(assertion)
                
        return new_abox, deleted_count

    def _phase1_deterministic_normalization(self, ontology: dict, equivalence_map: dict) -> tuple[dict, list]:
        print("\n" + "="*50)
        print("PHASE 1: DETERMINISTIC NORMALIZATION (CLASS-AWARE)")
        print("="*50)
        
        # Group instances by normalized string match
        exact_clusters = defaultdict(list)
        unique_original_instances = set()
        
        if "ontology_abox" in ontology:
            for inst in ontology["ontology_abox"]:
                if inst.get("type") == "ClassAssertion":
                    inst_id = inst.get("individual", "").strip()
                    if not inst_id:
                        continue
                    
                    unique_original_instances.add(inst_id)
                    group_key = re.sub(r'[\W_]+', '', inst_id).lower()
                    inst_class = inst.get("class")

                    exact_clusters[group_key].append({
                        "original_id": inst_id,
                        "class": inst_class
                    })

        print(f"Found {len(unique_original_instances)} unique raw instances.")
        
        polysemy_candidates = {} # To pass to Phase 2
        same_individual_assertions = []
        
        for group_key, items in exact_clusters.items():
            # Group by class within the exact string match
            class_groups = defaultdict(list)
            
            for item in items:
                orig_id = item["original_id"]
                resolved_class = self.__resolve_class(item["class"], equivalence_map)
                class_groups[resolved_class].append(orig_id)
            
            # For each class group, if there are multiple unique string variations/instances, assert SameIndividual
            for cls, ids in class_groups.items():
                unique_ids = sorted(list(set(ids)))
                if len(unique_ids) > 1:
                    same_individual_assertions.append({
                        "type": "SameIndividual",
                        "individuals": unique_ids,
                        "_class_ref": cls # Temporary ref to filter after Phase 2
                    })
                
            # If there are multiple classes for this normalized string, this is Polysemy / Extraction Bleed!
            if len(class_groups) > 1:
                polysemy_candidates[group_key] = class_groups

        print(f"Phase 1 generated {len(same_individual_assertions)} 'SameIndividual' assertions.")
        print(f"Identified {len(polysemy_candidates)} Polysemy / Contextual Bleed candidates.")
        
        return polysemy_candidates, same_individual_assertions

    def _phase2_polysemy_resolution(self, ontology: dict, polysemy_candidates: dict, tbox_classes: dict, equivalence_map: dict, same_individual_assertions: list) -> list:
        print("\n" + "="*50)
        print("PHASE 2: POLYSEMY & ARTIFACT RESOLUTION")
        print("="*50)
        
        class_actions = {}
        
        if not polysemy_candidates:
            print("No identical-name collisions found. Skipping Phase 2.")
        else:
            print(f"Preparing {len(polysemy_candidates)} identical-string collisions for LLM review...")
            agent = InstancePolysemyAgent(backend=self.backend)
            
            for i, (group_key, class_groups) in enumerate(polysemy_candidates.items()):
                print(f"\n  [Collision {i+1}/{len(polysemy_candidates)}] String: '{group_key}'")
                resolutions, _ = agent.run(group_key, class_groups, tbox_classes)
                
                for res in resolutions:
                    cls = res.get("class")
                    action = res.get("action")
                    print(f"    - Sub-Entity Class '{cls}' -> Action: {action.upper()}")
                    
                    if cls in class_groups:
                        for orig_id in class_groups[cls]:
                            class_actions[(orig_id, cls)] = action

            print("\n[!] Applying Phase 2 Actions to ABox...")
            abox = ontology.get("ontology_abox", [])
            new_abox, deleted_count = self._prune_abox(abox, class_actions, equivalence_map)
            
            # Prune Phase 1 SameIndividual assertions that have been rejected in Phase 2
            pruned_same_individuals = []
            for sa in same_individual_assertions:
                cls_ref = sa.pop("_class_ref", None) # Remove it so it doesn't leak into output
                valid_individuals = [
                    ind for ind in sa["individuals"] 
                    if class_actions.get((ind, cls_ref), "merge") != "reject"
                ]
                if len(valid_individuals) > 1:
                    sa["individuals"] = valid_individuals
                    pruned_same_individuals.append(sa)
                    
            same_individual_assertions = pruned_same_individuals

            ontology["ontology_abox"] = new_abox
            print(f"Phase 2 applied. Automatically dropped {deleted_count} artifact assertions. (counting both ClassAssertions and associated properties)")
            
        return same_individual_assertions

    def _phase3_fuzzy_clustering(self, ontology: dict) -> list:
        print("\n" + "="*50)
        print("PHASE 3: FUZZY SEMANTIC CLUSTERING")
        print("="*50)
        
        # Extract surviving canonical instances from the ABox
        surviving_instances = set()
        for assertion in ontology.get("ontology_abox", []):
            if assertion.get("type") == "ClassAssertion":
                ind_id = assertion.get("individual", "").strip()
                if ind_id:
                    surviving_instances.add(ind_id)
                    
        surviving_instances = sorted(list(surviving_instances))
        
        fuzzy_clusters = []
        if len(surviving_instances) > 1:
            print(f"Vectorizing {len(surviving_instances)} instances using TF-IDF token & character matching...")
            
            # Word-level TF-IDF (captures whole token matches, includes single chars/digits)
            word_vectorizer = TfidfVectorizer(
                analyzer='word', 
                ngram_range=(1, 2), 
                token_pattern=r'(?u)\b\w+\b'
            )
            word_embeddings = word_vectorizer.fit_transform(surviving_instances)
            
            # Char-level TF-IDF (captures typos)
            char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4))
            char_embeddings = char_vectorizer.fit_transform(surviving_instances)
            
            # Optimal hyperparameters from grid search evaluation
            w_w = 20.0
            c_w = 5.0
            thresh = 0.4
            
            # Combine and weight matrices
            sparse_embeddings = hstack([word_embeddings * w_w, char_embeddings * c_w])
            embeddings = sparse_embeddings.toarray()
            
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=thresh,
                metric='cosine', 
                linkage='average'
            )
            
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Extract clusters
            clusters_dict = defaultdict(list)
            for inst, label in zip(surviving_instances, cluster_labels):
                clusters_dict[label].append(inst)
                
            fuzzy_clusters = [members for members in clusters_dict.values() if len(members) > 1]
            
            print(f"-> Generated {len(fuzzy_clusters)} potential synonym clusters for Phase 4 investigation.")
            
        else:
            print("Not enough instances remaining to cluster.")
            
        return fuzzy_clusters

    def _phase4_semantic_resolution(self, ontology: dict, fuzzy_clusters: list, tbox_classes: dict, full_equivalence_map: dict):
        print("\n" + "="*50)
        print("PHASE 4: SEMANTIC CLUSTER RESOLUTION")
        print("="*50)
        
        if not fuzzy_clusters:
            print("No fuzzy clusters to resolve. Skipping Phase 4.")
            return

        print(f"Passing {len(fuzzy_clusters)} Phase 3 fuzzy clusters to the LLM agent for semantic resolution...\n")
        
        # Build a lookup for instance details (class and properties)
        instance_details = defaultdict(lambda: {"class": None, "properties": []})
        for assertion in ontology.get("ontology_abox", []):
            a_type = assertion.get("type")
            if a_type == "ClassAssertion":
                ind = assertion.get("individual", "")
                if ind:
                    instance_details[ind]["class"] = assertion.get("class")
            elif a_type == "DataPropertyAssertion":
                subj = assertion.get("subject", "")
                if subj:
                    instance_details[subj]["properties"].append(f"{assertion.get('property')}: {assertion.get('value')}")
            elif a_type == "ObjectPropertyAssertion":
                subj = assertion.get("subject", "")
                if subj:
                    instance_details[subj]["properties"].append(f"{assertion.get('property')}: {assertion.get('object')}")

        """
        # Build reverse equivalence map for all equivalent classes
        reverse_equivalence = defaultdict(set)
        for k, v in full_equivalence_map.items():
            reverse_equivalence[k].add(k)
            reverse_equivalence[k].add(v)
            reverse_equivalence[v].add(k)
            reverse_equivalence[v].add(v)
        """ 

        # Also, for each class, collect all classes that are equivalent (including itself)
        def get_equivalent_classes(cls_name):
            return full_equivalence_map.get(cls_name, [cls_name])

        agent = SemanticClusterAgent(backend=self.backend)
        new_same_individual_assertions = []

        for i, cluster in enumerate(fuzzy_clusters):
            print(f"\n  [Cluster {i+1}/{len(fuzzy_clusters)}]")
            cluster_with_context = {}
            for item in cluster:
                print(f"    * {item}")
                details = instance_details[item]
                cls_name = details["class"]
                cls_desc = tbox_classes.get(cls_name, "No description available.") if cls_name else "Unknown class"
                props = details["properties"]
                eq_classes = get_equivalent_classes(cls_name)
                cluster_with_context[item] = {
                    "class": cls_name,
                    "class_description": cls_desc,
                    "properties": props,
                    "equivalent_classes": eq_classes
                }
                
            resolutions, _ = agent.run(cluster, cluster_with_context)
            
            # Map canonical targets to lists of equivalent original_ids
            equivalency_groups = defaultdict(list)
            
            for res in resolutions:
                action = res.get("action")
                orig_id = res.get("original_id")
                target_id = res.get("target_id")
                
                if action == "equivalent" and target_id and orig_id != target_id:
                    equivalency_groups[target_id].append(orig_id)
                    print(f"    -> Merging: '{orig_id}' equivalent to '{target_id}'")
                elif action == "keep_distinct" or target_id is None:
                    print(f"    -> Keeping distinct: '{orig_id}'")
                    
            for target_id, equivalent_ids in equivalency_groups.items():
                if equivalent_ids:
                    # The SameIndividual assertion should include the target_id and all its equivalent items
                    group = [target_id] + equivalent_ids
                    # Deduplicate and sort
                    group = sorted(list(set(group)))
                    if len(group) > 1:
                        new_same_individual_assertions.append({
                            "type": "SameIndividual",
                            "individuals": group
                        })
                        
        if new_same_individual_assertions:
            print(f"\n[!] Phase 4 produced {len(new_same_individual_assertions)} new SameIndividual assertions.")
            if "ontology_abox" not in ontology:
                ontology["ontology_abox"] = []
            ontology["ontology_abox"].extend(new_same_individual_assertions)
        else:
            print("\n[!] Phase 4 did not produce any new SameIndividual assertions.")

    def _update_abox_with_assertions(self, ontology: dict, same_individual_assertions: list):
        print("\n[!] Updating ABox with Phase 1 SameIndividual assertions...")
        
        # In case Phase 2 didn't happen, we must pop the _class_ref to prevent invalid JSON properties.
        for sa in same_individual_assertions:
            sa.pop("_class_ref", None)
        
        if "ontology_abox" not in ontology:
            ontology["ontology_abox"] = []

        ontology["ontology_abox"].extend(same_individual_assertions)
        
        print(f"\nABox successfully enriched with {len(same_individual_assertions)} SameIndividual axioms.")

    def resolve(self, ontology: dict, save_phase2: str = None, resume_from_phase3: bool = False) -> dict:
        
        # Build class lookup dictionary for descriptions
        tbox_classes = {}
        if "ontology_tbox" in ontology and "classes" in ontology["ontology_tbox"]:
            for cls in ontology["ontology_tbox"]["classes"]:
                cls_id = cls.get("id")
                if cls_id:
                    tbox_classes[cls_id] = cls.get("description", "No description available.")

        """
        # Build equivalence map from TBox axioms to act as an in-memory reasoner
        equivalence_map = {}
        if "ontology_tbox" in ontology and "axioms" in ontology["ontology_tbox"]:
            for axiom in ontology["ontology_tbox"]["axioms"]:
                if axiom.get("type") == "equivalentClass":
                    domains = axiom.get("domains", [])
                    ranges = axiom.get("ranges", [])
                    for d in domains:
                        for r in ranges:
                            if d != "Entity" and r != "Entity":
                                union(d, r)

        # Build equivalence sets
        equivalence_sets = defaultdict(set)
        for cls in parent:
            root = find(cls)
            equivalence_sets[root].add(cls)
        # For each class, map to its full equivalence set (including itself)
        full_equivalence_map = {}
        for eq_group in equivalence_sets.values():
            for cls in eq_group:
                full_equivalence_map[cls] = sorted(eq_group) """
        # --- Build a complete, bidirectional, transitively closed equivalence mapping ---
        # Union-Find (Disjoint Set) for equivalence classes
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent.setdefault(x, x)
            parent.setdefault(y, y)
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px
             
        # Add all equivalenceClass axioms
        if "ontology_tbox" in ontology and "axioms" in ontology["ontology_tbox"]:
            for axiom in ontology["ontology_tbox"]["axioms"]:
                if axiom.get("type") == "equivalentClass":
                    domains = axiom.get("domains", [])
                    ranges = axiom.get("ranges", [])
                    for d in domains:
                        for r in ranges:
                            if d != "Entity" and r != "Entity":
                                union(d, r)
        # Build equivalence sets
        equivalence_sets = defaultdict(set)
        for cls in parent:
            root = find(cls)
            equivalence_sets[root].add(cls)
        # For each class, map to its full equivalence set (including itself)
        full_equivalence_map = {}
        for eq_group in equivalence_sets.values():
            for cls in eq_group:
                full_equivalence_map[cls] = sorted(eq_group)
        print(f"Loaded {len(full_equivalence_map)} equivalence class groups from TBox axioms.")

        if not resume_from_phase3:
            # Phase 1
            polysemy_candidates, same_individual_assertions = self._phase1_deterministic_normalization(ontology, equivalence_map)

            # Phase 2
            same_individual_assertions = self._phase2_polysemy_resolution(ontology, polysemy_candidates, tbox_classes, equivalence_map, same_individual_assertions)

            # Update ABox with Phase 1 & 2 SameIndividual assertions so they are saved
            self._update_abox_with_assertions(ontology, same_individual_assertions)

            if save_phase2:
                print(f"\n[!] Saving intermediate Phase 1 & 2 results to {save_phase2}...")
                with open(save_phase2, 'w', encoding='utf-8') as f:
                    json.dump(ontology, f, indent=4)
        else:
            print("\n[!] Resuming from Phase 3: Skipping Phase 1 and Phase 2.")

        # Phase 3
        fuzzy_clusters = self._phase3_fuzzy_clustering(ontology)

        # Phase 4
        self._phase4_semantic_resolution(ontology, fuzzy_clusters, tbox_classes, full_equivalence_map)

        return ontology


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="phi-npu-openvino")
    parser.add_argument("--ontology_path", default=r"plsInsertPathHere", help="Path to the ontology JSON file.")
    parser.add_argument("--dev", action="store_true", help="Run in development mode (does not save the output file).")
    parser.add_argument("--save_phase2", action="store_true", help="Save the ontology after Phase 2 completes in a local cache folder.")
    parser.add_argument("--resume_from_phase3", action="store_true", help="Skip Phase 1 and 2; assume the input ontology was already saved after Phase 2.")
    args = parser.parse_args()

    backend = load_backend(name=args.backend)

    try:
        with open(args.ontology_path, 'r', encoding='utf-8') as f:
            ontology = json.load(f)
    except Exception as e:
        print(f"Error loading ontology: {e}")
        return

    save_path = None
    if args.save_phase2:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(script_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        base_name = os.path.basename(args.ontology_path)
        name, ext = os.path.splitext(base_name)
        save_path = os.path.join(cache_dir, f"{name}_phase2_cached{ext}")

    resolver = InstanceResolver(backend=backend)
    ontology = resolver.resolve(ontology, save_phase2=save_path, resume_from_phase3=args.resume_from_phase3)

    # Save the updated ontology to a new file (skip saving in dev mode)
    if not args.dev:    
        resolved_path = args.ontology_path.replace("_classResolved.json", "_fullyResolved.json")
        if "_fullyResolved.json" not in resolved_path:
            resolved_path = args.ontology_path.replace(".json", "_fullyResolved.json")
            
        with open(resolved_path, 'w', encoding='utf-8') as f:
            json.dump(ontology, f, indent=4)   
        print(f"Successfully saved fully resolved ontology to: {resolved_path}")
    else:
        print("Dev mode enabled: Skipping file save.")

if __name__ == "__main__":
    main()