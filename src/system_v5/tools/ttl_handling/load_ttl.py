"""function to load the TTL-file. Either the whole file, or only classes or and/or certain properties etc.."""
from rdflib import Graph
from rdflib.namespace import RDF, OWL, RDFS
#from asyncio import graph

def load_ttl(file_path: str) -> dict:
    
    g = Graph()
    g.parse(file_path, format="turtle")

    classes = set(g.subjects(RDF.type, OWL.Class))
    individuals = set(g.subjects(RDF.type, OWL.NamedIndividual))
    object_properties = set(g.subjects(RDF.type, OWL.ObjectProperty))
    datatype_properties = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    
    return {
        "graph": g,
        "classes": classes,
        "individuals": individuals,
        "object_properties": object_properties,
        "datatype_properties": datatype_properties
    }