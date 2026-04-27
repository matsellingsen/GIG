from rdflib import Graph, RDF, RDFS, OWL, URIRef

class SchemaValidationEngine:
    """
    Validates object property assertions against the ontology's declared
    domain and range. Removes invalid triples before inference.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def validate(self) -> list:
        """
        Validate all object property assertions in the graph.
        Returns a list of removed triples for logging/debugging.
        """
        removed = []

        # Iterate over all object property assertions
        for s, p, o in list(self.graph.triples((None, None, None))):
            # Only validate object properties (ignore data properties)
            if not isinstance(o, URIRef):
                continue

            # Check if p is an ObjectProperty
            if (p, RDF.type, OWL.ObjectProperty) not in self.graph:
                continue

            # Collect declared domain and range classes
            domains = set(self.graph.objects(p, RDFS.domain))
            ranges = set(self.graph.objects(p, RDFS.range))

            # Get types of subject and object
            subject_types = set(self.graph.objects(s, RDF.type))
            object_types = set(self.graph.objects(o, RDF.type))

            # Domain validation
            domain_ok = (
                not domains or
                subject_types.intersection(domains)
            )

            # Range validation
            range_ok = (
                not ranges or
                object_types.intersection(ranges)
            )

            # If invalid, remove the triple
            if not domain_ok or not range_ok:
                self.graph.remove((s, p, o))
                removed.append((s, p, o))

        return removed
