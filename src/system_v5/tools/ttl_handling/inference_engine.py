from rdflib import Graph, RDF, RDFS, OWL, URIRef, Literal
from rdflib.namespace import XSD


class OntologyInferenceEngine:
    """
    Lightweight, TTL-aligned inference engine.

    Implements ONLY what your ontology actually supports:
    - Subclass-based type inference
    - Equivalent-class-based type inference
    - Domain-based typing
    - Range-based typing
    - Custom inverse rule for hasPart / isPartOf (no owl:inverseOf)
    - Optional annotation / provenance propagation (no new triples, just context-level)
    """

    def __init__(self, graph: Graph, ex_namespace: str = "http://example.org/sensInnovationAps_ontology#"):
        self.graph = graph
        self.EX = URIRef(ex_namespace)  # not heavily used, but kept for clarity

        # Custom inverse mapping (no owl:inverseOf in TTL, so we define it here)
        # You can extend this dict if you add more custom inverses.
        self.custom_inverse_properties = {
            URIRef(f"{ex_namespace}hasPart"): URIRef(f"{ex_namespace}isPartOf")
        }

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def materialize_inferences(self) -> None:
        """
        Materialize all supported inferences into the graph.
        This mutates self.graph by adding inferred triples.
        """
        self._infer_types_from_subclass()
        self._infer_types_from_equivalent_class()
        #self._infer_types_from_domain()
        #self._infer_types_from_range()
        self._infer_custom_inverse_properties()

    # -------------------------------------------------------------------------
    # Type inference
    # -------------------------------------------------------------------------

    def _infer_types_from_subclass(self) -> None:
        """
        If:
          x rdf:type C
          C rdfs:subClassOf D
        then:
          x rdf:type D
        (transitively)
        """
        new_triples = True
        while new_triples:
            new_triples = False
            for x, _, c in list(self.graph.triples((None, RDF.type, None))):
                for d in self.graph.objects(c, RDFS.subClassOf):
                    triple = (x, RDF.type, d)
                    if triple not in self.graph:
                        self.graph.add(triple)
                        new_triples = True

    def _infer_types_from_equivalent_class(self) -> None:
        """
        If:
          x rdf:type C1
          C1 owl:equivalentClass C2
        then:
          x rdf:type C2
        (and vice versa)
        """
        new_triples = True
        while new_triples:
            new_triples = False
            for c1, _, c2 in list(self.graph.triples((None, OWL.equivalentClass, None))):
                for x in self.graph.subjects(RDF.type, c1):
                    triple = (x, RDF.type, c2)
                    if triple not in self.graph:
                        self.graph.add(triple)
                        new_triples = True
                for x in self.graph.subjects(RDF.type, c2):
                    triple = (x, RDF.type, c1)
                    if triple not in self.graph:
                        self.graph.add(triple)
                        new_triples = True

    def _infer_types_from_domain(self) -> None:
        """
        If:
          p rdfs:domain C
          x p y
        then:
          x rdf:type C
        """
        new_triples = True
        while new_triples:
            new_triples = False
            for p, _, c in list(self.graph.triples((None, RDFS.domain, None))):
                for x, _, _ in self.graph.triples((None, p, None)):
                    triple = (x, RDF.type, c)
                    if triple not in self.graph:
                        self.graph.add(triple)
                        new_triples = True

    def _infer_types_from_range(self) -> None:
        """
        If:
          p rdfs:range C
          x p y
        then:
          y rdf:type C
        """
        new_triples = True
        while new_triples:
            new_triples = False
            for p, _, c in list(self.graph.triples((None, RDFS.range, None))):
                for _, _, y in self.graph.triples((None, p, None)):
                    triple = (y, RDF.type, c)
                    if triple not in self.graph:
                        self.graph.add(triple)
                        new_triples = True

    # -------------------------------------------------------------------------
    # Custom inverse rule
    # -------------------------------------------------------------------------

    def _infer_custom_inverse_properties(self) -> None:
        """
        Custom inverse rule (no owl:inverseOf in TTL):

        For each (s, p, o) where p has a configured inverse q:
          infer (o, q, s)

        Example:
          hasPart ↔ isPartOf
          SENS motion hasPart ArticleID9230081
          → ArticleID9230081 isPartOf SENS motion
        """
        new_triples = []
        for p, q in self.custom_inverse_properties.items():
            for s, _, o in self.graph.triples((None, p, None)):
                triple = (o, q, s)
                if triple not in self.graph:
                    new_triples.append(triple)

        for t in new_triples:
            self.graph.add(t)

    # -------------------------------------------------------------------------
    # Optional: convenience method to get enriched context after inference
    # (You already have a retrieval function; this is just a helper if needed.)
    # -------------------------------------------------------------------------

    def get_types(self, entity_uri: URIRef):
        return list(self.graph.objects(entity_uri, RDF.type))

    def get_superclasses(self, class_uri: URIRef):
        """
        Return transitive superclasses of a class.
        Assumes _infer_types_from_subclass has already materialized subclass closure.
        """
        superclasses = set()

        def collect(c):
            for sup in self.graph.objects(c, RDFS.subClassOf):
                if sup not in superclasses:
                    superclasses.add(sup)
                    collect(sup)

        collect(class_uri)
        return list(superclasses)
