def save_result(result, model_name: str, pipeline_info: dict = None) -> None:
    """
    Save the generated ontology result along with metadata for future reference.
    
    Args:
        result: The structured output from the Orchestrator (Dict with 'ontology' and 'instances').
        model_name: Name of the backend model used.
        pipeline_info: Optional dictionary containing metadata like chunk processing count, 
                       input source path, execution time, parameters, etc.
    """
    import os
    import json
    from datetime import datetime

    output_dir = "C:\\Users\\matse\\gig\\src\\system\\intermediate_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_ontology_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)

    # Clean up the result if needed (e.g. if it contains raw strings instead of objects)
    # The Orchestrator guarantees a structure, but we ensure it's JSON serializable here.
    # get number of classes, axioms and instances for metadata
    num_classes = len(result.get("ontology", {}).get("classes", []))
    num_axioms = len(result.get("ontology", {}).get("axioms", []))
    num_instances = len(result.get("instances", []))

    data_to_save = {
        "metadata": {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "pipeline_info": pipeline_info or {},
            "num_classes": num_classes,
            "num_axioms": num_axioms,
            "num_instances": num_instances

        },
        "ontology_tbox": result.get("ontology", {}),
        "ontology_abox": result.get("instances", [])
    }

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)

    print(f"Saved structured ontology result to {output_path}")
