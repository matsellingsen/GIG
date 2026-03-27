import json
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, SKOS, OWL
import argparse

def extract_bfo_classes(ttl_file_path):
    """
    Reads the BFO OWL/TTL file and extracts all OWL classes
    along with their labels, definitions, and parent classes.
    """
    g = Graph()
    g.parse(ttl_file_path, format="turtle")
    
    classes_list = []
    for s in g.subjects(RDF.type, OWL.Class):
        if not isinstance(s, URIRef): continue
        
        label = g.value(s, RDFS.label)
        definition = g.value(s, SKOS.definition)
        
        # Get parent classes (superclasses)
        parents = []
        for p in g.objects(s, RDFS.subClassOf):
            if isinstance(p, URIRef):
                p_label = g.value(p, RDFS.label)
                if p_label: parents.append(str(p_label))
        
        if label:
            classes_list.append({
                "class": str(label),
                "parents": parents,
                "description": str(definition) if definition else "No definition provided."
            })
            
    return classes_list

def extract_bfo_axioms(ttl_file_path):
    """
    Reads the BFO OWL/TTL file and extracts property relationships (linear)
    along with their defined domains and ranges.
    """
    g = Graph()
    g.parse(ttl_file_path, format="turtle")
    
    axioms_list = []
    
    # 1. Add the core hierarchical connector (Taxonomy)
    axioms_list.append({
        "type": "subClassOf",
        "domains": ["Any Class"],
        "ranges": ["Any Class"],
        "description": "A hierarchical connection indicating that one class is a sub-type or child of another class."
    })
    
    # 2. Extract specific Object Properties as linear connectors (Relational)
    for s in g.subjects(RDF.type, OWL.ObjectProperty):
        if not isinstance(s, URIRef): continue
        
        label = g.value(s, RDFS.label)
        definition = g.value(s, SKOS.definition)
        
        # Get Domains (what types of things can be the 'subject')
        domains = []
        for d in g.objects(s, RDFS.domain):
            if isinstance(d, URIRef):
                d_label = g.value(d, RDFS.label)
                if d_label: domains.append(str(d_label))
                
        # Get Ranges (what types of things can be the 'object')
        ranges = []
        for r in g.objects(s, RDFS.range):
            if isinstance(r, URIRef):
                r_label = g.value(r, RDFS.label)
                if r_label: ranges.append(str(r_label))
        
        if label:
            axioms_list.append({
                "type": str(label),
                "domains": domains,
                "ranges": ranges,
                "description": str(definition) if definition else "No definition provided."
            })
            
    return axioms_list

def main():
    parser = argparse.ArgumentParser(description="Extract BFO classes and axioms from TTL file.")
    parser.add_argument("--ttl_path", type=str, default="C:\\Users\\matse\\bfo-2020\\21838-2\\owl\\bfo-core.ttl", help="Path to the BFO TTL file.")
    parser.add_argument("--output-folder", type=str, default="C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\base_ontology\\json", help="Folder to save the extracted JSON files.")
    args = parser.parse_args()

    # File path to the BFO 2020 TTL file
    ttl_path = args.ttl_path
    output_folder = args.output_folder

    # 1. Extract Classes 
    bfo_classes = extract_bfo_classes(ttl_path)

    # 2. Extract Axioms 
    bfo_axioms = extract_bfo_axioms(ttl_path)

    # 3. Save to JSON
    with open(f"{output_folder}/bfo_classes.json", "w", encoding="utf-8") as f:
        json.dump(bfo_classes, f, indent=4)
    
    with open(f"{output_folder}/bfo_axioms.json", "w", encoding="utf-8") as f:
        json.dump(bfo_axioms, f, indent=4)

if __name__ == "__main__":
    main()
    