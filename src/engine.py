"""ESG Intelligence engine — orchestrates the full GraphRAG pipeline.

This single object wires together every layer in the architecture diagram:

    crawlers (load) -> graph (build + propagate) -> graphrag (retrieve)
                    -> llm (narrate) -> compliance (SFDR)

The FastAPI app holds one shared :class:`ESGEngine` so the graph is built once.
"""
from __future__ import annotations

from .compliance.sfdr import classify_sfdr
from .crawlers import loader
from .graph.knowledge_graph import build_graph, propagate_supply_chain_risk
from .graphrag.retriever import GraphRAGRetriever
from .llm.narrative import generate_narrative


class ESGEngine:
    def __init__(self) -> None:
        self.graph = build_graph()
        self.risk = propagate_supply_chain_risk(self.graph)
        self.retriever = GraphRAGRetriever(self.graph)
        self.companies = {c["id"]: c for c in loader.load_companies()}

    # -- portfolio views -------------------------------------------------------
    def portfolio_scores(self) -> list[dict]:
        rows = []
        for cid, comp in self.companies.items():
            if not comp.get("in_portfolio"):
                continue
            r = self.risk[cid]
            sfdr = classify_sfdr(comp, r["composite_esg"])
            rows.append(
                {
                    "company_id": cid,
                    "name": comp["name"],
                    "sector": comp["sector"],
                    "base_esg": r["base_esg"],
                    "composite_esg": r["composite_esg"],
                    "supply_chain_risk": r["supply_chain_risk"],
                    "sfdr_article": sfdr["sfdr_article"],
                }
            )
        rows.sort(key=lambda x: x["composite_esg"], reverse=True)
        return rows

    def company_profile(self, company_id: str) -> dict | None:
        comp = self.companies.get(company_id)
        if comp is None:
            return None
        risk = self.risk[company_id]
        sfdr = classify_sfdr(comp, risk["composite_esg"])
        evidence = self.retriever._expand_evidence(company_id)
        narrative = generate_narrative(evidence, risk)
        return {
            "company": comp,
            "risk": risk,
            "sfdr": sfdr,
            "narrative": narrative,
        }

    # -- GraphRAG query --------------------------------------------------------
    def query(self, question: str, top_k: int = 3) -> dict:
        retrieval = self.retriever.retrieve(question, top_k=top_k)
        answers = []
        for ev in retrieval["evidence"]:
            cid = ev["company_id"]
            answers.append(
                {
                    "company_id": cid,
                    "name": ev["name"],
                    "narrative": generate_narrative(ev, self.risk.get(cid)),
                }
            )
        return {
            "question": question,
            "retrieved_documents": [
                {"company_id": d["company_id"], "source": d["source"], "relevance": d["relevance"]}
                for d in retrieval["documents"]
            ],
            "answers": answers,
        }
