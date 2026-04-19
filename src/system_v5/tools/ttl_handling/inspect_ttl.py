from rdflib import Graph
from rdflib.namespace import RDF, OWL, RDFS
import argparse
from .load_ttl import load_ttl
from .resolve_ttl_path import resolve_ttl_path

def inspect_ttl_file(file_path):

    resolved_ttl_path = resolve_ttl_path(file_path)
    ttl = load_ttl(file_path=resolved_ttl_path)
    print(f"Total triples: {len(ttl['graph'])}")

    g = ttl["graph"]
    classes = ttl["classes"]
    individuals = ttl["individuals"]
    object_properties = ttl["object_properties"]
    datatype_properties = ttl["datatype_properties"]

    print(f"Classes: {len(classes)}")
    print(f"Individuals: {len(individuals)}")
    print(f"Object Properties: {len(object_properties)}")
    print(f"Datatype Properties: {len(datatype_properties)}") 


    # Classes with no subclasses and no instances
    loose_classes = []
    for cls in classes:
        has_subclass = (cls, RDFS.subClassOf, None) in g or (None, RDFS.subClassOf, cls) in g
        has_instance = (None, RDF.type, cls) in g
        if not has_subclass and not has_instance:
            loose_classes.append(cls)

    print(f"Loose branches (classes with no subclasses or instances): {len(loose_classes)}")
    for cls in loose_classes:
        print(cls)

def main():
    parser = argparse.ArgumentParser(description="Inspect a Turtle (.ttl) file and provide statistics about its contents.")
    parser.add_argument("ttl_file_path", help="Path to the Turtle file to inspect.")
    args = parser.parse_args()

    inspect_ttl_file(args.ttl_file_path)

if __name__ == "__main__":
    main()

