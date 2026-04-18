import sys
sys.path.append("C:\\Users\\matse\\gig\\src\\system_v5") # Adjust this path to point to the root of your project if needed

import argparse
import os
import json
from backends import load_backend
from system_v5.tools.ontology_cleanup.log_cleanup_metedata import add_cleanup_metadata
from system_v5.tools.ontology_cleanup.resolve_classes import ClassResolver
from system_v5.tools.ontology_cleanup.resolve_instances import InstanceResolver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="phi-npu-openvino", help="LLM backend to use")
    parser.add_argument("--ontology_path", default=r"C:\Users\matse\gig\src\system_v5\intermediate_results\phi-npu-openvino_ontology_20260406_232546.json", help="Path to the ontology JSON file.")
    parser.add_argument("--dev", action="store_true", help="Run in development mode (does not save the output file).")
    parser.add_argument("--num_chunks", type=int, default=None, help="Number of chunks to process for testing.")
    parser.add_argument("--start_from_phase", type=int, default=1, help="Phase number to start from (1 for class resolution, 2 for instance resolution, 3 for cleanup metadata logging).")
    args = parser.parse_args()

    print(f"Loading ontology from: {args.ontology_path}")
    with open(args.ontology_path, 'r', encoding='utf-8') as f:
        ontology = json.load(f)

    if args.start_from_phase == 1:
        # 1. Run Class Resolution
        print("\n" + "="*50)
        print("STEP 1: RESOLVING CLASSES (TBOX)")
        print("="*50)
        
        backend = load_backend(name=args.backend)
        class_resolver = ClassResolver(backend=backend, num_chunks=args.num_chunks)
        ontology = class_resolver.resolve(ontology)

    if args.start_from_phase <= 2:
        # 2. Run Instance Resolution
        print("\n" + "="*50)
        print("STEP 2: RESOLVING INSTANCES (ABOX)")
        print("="*50)
        
        instance_resolver = InstanceResolver()
        ontology = instance_resolver.resolve(ontology)

    if args.start_from_phase <= 3:
        # 3. Add Cleanup Metadata
        print("\n" + "="*50)
        print("STEP 3: ADDING CLEANUP METADATA")
        print("="*50)

        ontology = add_cleanup_metadata(ontology)

    # Save Final Extracted
    print("\n" + "="*50)
    print("ONTOLOGY RESOLUTION PIPELINE COMPLETE")
    print("="*50 + "\n")

    if not args.dev:
        resolved_path = args.ontology_path.replace(".json", "_fullyResolved_and_logged.json")
        with open(resolved_path, 'w', encoding='utf-8') as f:
            json.dump(ontology, f, indent=4)
        print(f"Successfully saved fully resolved ontology to: {resolved_path}")
    else:
        print("Dev mode enabled: Skipping file save.")

if __name__ == "__main__":
    main()
