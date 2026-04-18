import datetime

def add_cleanup_metadata(ontology, cleanup_log=None, pipeline_version=None, extra_notes=None):
    """
    Adds a 'cleanup_metadata' field to the ontology dict with post-cleanup stats and summary.
    Optionally include a cleanup_log (e.g., pruned/merged info), pipeline_version, and extra_notes.
    """
    # Compute counts
    num_classes = len(ontology.get("ontology_tbox", {}).get("classes", []))
    num_axioms = len(ontology.get("ontology_tbox", {}).get("axioms", []))
    num_instances = len(ontology.get("ontology_abox", []))
    pruned = ontology.get("pruned", {})
    pruned_total = sum(len(v) for v in pruned.values()) if pruned else 0

    # Compose metadata
    cleanup_metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "num_classes": num_classes,
        "num_axioms": num_axioms,
        "num_instances": num_instances,
        "num_pruned_assertions": pruned_total,
        "phases": [
            {"phase": k, "num_pruned": len(v)} for k, v in pruned.items()
        ] if pruned else [],
    }
    if cleanup_log is not None:
        cleanup_metadata["cleanup_log"] = cleanup_log
    if pipeline_version is not None:
        cleanup_metadata["pipeline_version"] = pipeline_version
    if extra_notes is not None:
        cleanup_metadata["notes"] = extra_notes

    ontology["cleanup_metadata"] = cleanup_metadata
    return ontology