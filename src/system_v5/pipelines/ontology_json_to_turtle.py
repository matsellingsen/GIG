import json
import os
from rdflib import RDFS, Graph, Namespace, URIRef, Literal, RDF, OWL, XSD
import argparse
import urllib.parse

def safe_uri_fragment(s):
    # Percent-encode everything except unreserved URI characters
    return urllib.parse.quote(str(s), safe="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def convert_ontology_json_to_turtle(ontology_json: dict) -> str:
    g = Graph()

    # Define namespaces
    EX = Namespace("http://example.org/sensInnovationAps_ontology#")
    g.bind("ex", EX)

    # PROCESS TBOX 
    classes = ontology_json.get("ontology_tbox", {}).get("classes", [])
    axioms = ontology_json.get("ontology_tbox", {}).get("axioms", [])
    ## classes
    for cls in classes:
        cls_uri = EX[safe_uri_fragment(cls["id"])]
        g.add((cls_uri, RDF.type, OWL.Class))
        g.add((cls_uri, RDFS.label, Literal(cls.get("id"))))
        if "description" in cls:
            g.add((cls_uri, RDFS.comment, Literal(cls.get("description"))))

    ## axioms
    for axiom in axioms:
        # check if axioms contains "description"
        if axiom.get("description"): # this is a base ontology axiom, so we can add the description as a comment to the graph for better context.             if axiom.get("type") != "equivalentClass": # we handle EquivalentClass axioms differently 
            if axiom.get("type") != "equivalentClass": # is handled further down.    
                prop_uri = EX[safe_uri_fragment(axiom["type"])]
                g.add((prop_uri, RDF.type, OWL.ObjectProperty))
                g.add((prop_uri, RDFS.label, Literal(axiom["type"])))
                if "description" in axiom:
                    g.add((prop_uri, RDFS.comment, Literal(axiom["description"])))
                for domain in axiom.get("domains", []):
                    g.add((prop_uri, RDFS.domain, EX[safe_uri_fragment(domain)]))
                for range_ in axiom.get("ranges", []):
                    g.add((prop_uri, RDFS.range, EX[safe_uri_fragment(range_)]))    

                continue # skip to next axiom since this one is already processed as a property with description.
            
        
        # check if axioms has a type field
        if axiom.get("type"):
            print(f"Processing axiom of type: {axiom['type']}")
            # Handle SubClassOf axioms
            if axiom.get("type") == "SubClassOf":
                subclass_uri = EX[safe_uri_fragment(axiom["subclass"])]
                superclass_uri = EX[safe_uri_fragment(axiom["superclass"])]
                g.add((subclass_uri, RDFS.subClassOf, superclass_uri))
            if axiom.get("type") == "equivalentClass":
                class1_uri = EX[safe_uri_fragment(axiom["domains"][0])]
                class2_uri = EX[safe_uri_fragment(axiom["ranges"][0])]
                g.add((class1_uri, OWL.equivalentClass, class2_uri))

        else:
            # Handle object/data properties
            if axiom.get("datatype"): #this is a data property
                prop_uri = EX[safe_uri_fragment(axiom["property"])]
                g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
                g.add((prop_uri, RDFS.label, Literal(axiom["property"])))
                g.add((prop_uri, RDFS.domain, EX[safe_uri_fragment(axiom["domain"])]))
                # Extract the XSD datatype (e.g., "boolean" from "xsd:boolean")
                datatype_str = axiom["datatype"].split(":")[-1]
                g.add((prop_uri, RDFS.range, getattr(XSD, datatype_str)))
            else: # this is an object property
                prop_uri = EX[safe_uri_fragment(axiom["property"])]
                g.add((prop_uri, RDF.type, OWL.ObjectProperty))
                g.add((prop_uri, RDFS.label, Literal(axiom["property"])))
                g.add((prop_uri, RDFS.domain, EX[safe_uri_fragment(axiom["domain"])]))
                g.add((prop_uri, RDFS.range, EX[safe_uri_fragment(axiom["range"])]))

    # PROCESS ABOX
    for stmt in ontology_json.get("ontology_abox", []):
        if stmt["type"] == "DeclareIndividual":
            ind_uri = EX[safe_uri_fragment(stmt["id"])]
            g.add((ind_uri, RDF.type, OWL.NamedIndividual))
            # Optionally, add provenance as annotation
            g.add((ind_uri, EX["chunk_id"], Literal(stmt["chunk_id"])))

        elif stmt["type"] == "ClassAssertion":
            ind_uri = EX[safe_uri_fragment(stmt["individual"])]
            class_uri = EX[safe_uri_fragment(stmt["class"])]
            g.add((ind_uri, RDF.type, class_uri))
            # Optionally, add provenance
            g.add((ind_uri, EX["chunk_id"], Literal(stmt["chunk_id"])))

        elif stmt["type"] == "ObjectPropertyAssertion":
            subj_uri = EX[safe_uri_fragment(stmt["subject"])]
            prop_uri = EX[safe_uri_fragment(stmt["property"])]
            obj_uri = EX[safe_uri_fragment(stmt["object"])]
            g.add((subj_uri, prop_uri, obj_uri))
            # Optionally, add provenance
            g.add((subj_uri, EX["chunk_id"], Literal(stmt["chunk_id"])))

        elif stmt["type"] == "DataPropertyAssertion":
            subj_uri = EX[safe_uri_fragment(stmt["subject"])]
            prop_uri = EX[safe_uri_fragment(stmt["property"])]

            # Extract datatype (e.g., "boolean" from "xsd:boolean")
            datatype_str = stmt["datatype"].split(":")[-1]
            value = stmt["value"]

            literal_kwargs = {}
            try:
                if datatype_str == "boolean":
                    value = value.lower() == "true"
                    literal_kwargs["datatype"] = XSD.boolean
                elif datatype_str == "integer":
                    value = int(value)
                    literal_kwargs["datatype"] = XSD.integer
                elif datatype_str == "decimal":
                    value = float(value)
                    literal_kwargs["datatype"] = XSD.decimal
                else:
                    literal_kwargs["datatype"] = getattr(XSD, datatype_str)
                g.add((subj_uri, prop_uri, Literal(value, **literal_kwargs)))
            except (ValueError, TypeError, AttributeError):
                # Fallback: store as plain string (no datatype)
                g.add((subj_uri, prop_uri, Literal(stmt["value"])))
            # Optionally, add provenance
            g.add((subj_uri, EX["chunk_id"], Literal(stmt["chunk_id"])))

    return g.serialize(format="turtle")#.decode("utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology_json_path", required=True, help="Path to the ontology JSON file.")
    parser.add_argument("--output_ttl_path", required=False, default="C:\\Users\\matse\\gig\\src\\system_v5\\KB\\new_additions\\", help="Path to save the output Turtle file.")
    args = parser.parse_args()

    print(f"Loading ontology from: {args.ontology_json_path}")
    with open(args.ontology_json_path, 'r', encoding='utf-8') as f:
        ontology = json.load(f)
    
    print("Converting ontology JSON to Turtle format...")
    ttl_output = convert_ontology_json_to_turtle(ontology)

    output_path = os.path.join(args.output_ttl_path, args.ontology_json_path.split("\\")[-1].replace(".json", ".ttl"))
    print(f"Saving Turtle output to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ttl_output)

    print("Conversion complete.")

if __name__ == "__main__":
    main()