from rdflib import Graph
from rdflib.namespace import RDF, OWL, RDFS
import argparse

def inspect_ttl_file(file_path):

    g = Graph()
    g.parse(file_path, format="turtle")
    print(f"Total triples: {len(g)}")


    classes = set(g.subjects(RDF.type, OWL.Class))
    individuals = set(g.subjects(RDF.type, OWL.NamedIndividual))
    object_properties = set(g.subjects(RDF.type, OWL.ObjectProperty))
    datatype_properties = set(g.subjects(RDF.type, OWL.DatatypeProperty))

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

