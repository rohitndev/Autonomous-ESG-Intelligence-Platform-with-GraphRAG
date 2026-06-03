"""GraphRAG retriever.

Full platform:  Microsoft GraphRAG builds community summaries over the Neo4j graph
and ChromaDB stores document embeddings for multi-hop retrieval.

Prototype:  retrieval combines two transparent signals that together give the
"multi-hop" behaviour standard vector RAG cannot —
  1. **Document relevance** via TF-IDF cosine similarity (a ChromaDB-embedding proxy).
  2. **Graph expansion**: every company surfaced by step 1 is expanded along its
     supply-chain edges so upstream suppliers and their controversies join the context.

The retrieved sub-graph + documents become the grounding "evidence chain" handed to
the narrative generator.
"""
from __future__ import annotations

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..crawlers import loader


class GraphRAGRetriever:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.documents = loader.load_documents()
        self._corpus = [d["text"] for d in self.documents]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._doc_matrix = self._vectorizer.fit_transform(self._corpus)

    # -- step 1: vector-style document retrieval -------------------------------
    def _rank_documents(self, query: str, top_k: int) -> list[dict]:
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._doc_matrix).ravel()
        ranked = sorted(
            zip(self.documents, sims), key=lambda x: x[1], reverse=True
        )
        hits = []
        for doc, score in ranked[:top_k]:
            if score <= 0:
                continue
            hits.append({**doc, "relevance": round(float(score), 3)})
        return hits

    # -- step 2: multi-hop graph expansion -------------------------------------
    def _expand_evidence(self, company_id: str) -> dict:
        g = self.graph
        node = g.nodes[company_id]
        evidence = {
            "company_id": company_id,
            "name": node["name"],
            "controversies": [],
            "supply_chain": [],
        }
        # direct controversies
        for _, tgt, attrs in g.out_edges(company_id, data=True):
            if attrs.get("relation") == "has_controversy":
                evidence["controversies"].append(g.nodes[tgt]["headline"])
        # suppliers + their controversies (the multi-hop part)
        for supplier in nx.descendants(g, company_id):
            if g.nodes[supplier].get("kind") != "company":
                continue
            sup_controversies = [
                g.nodes[t]["headline"]
                for _, t, a in g.out_edges(supplier, data=True)
                if a.get("relation") == "has_controversy"
            ]
            evidence["supply_chain"].append(
                {
                    "supplier_id": supplier,
                    "supplier_name": g.nodes[supplier]["name"],
                    "tier": nx.shortest_path_length(g, company_id, supplier),
                    "controversies": sup_controversies,
                }
            )
        return evidence

    # -- public API ------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3) -> dict:
        """Return ranked documents + the expanded evidence sub-graph for a query."""
        docs = self._rank_documents(query, top_k)
        evidence = [self._expand_evidence(d["company_id"]) for d in docs]
        return {"query": query, "documents": docs, "evidence": evidence}
