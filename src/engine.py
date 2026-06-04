"""ESG Intelligence engine — orchestrates the full GraphRAG pipeline.

This single object wires together every layer in the architecture diagram:

    crawlers (load) -> graph (build + propagate) -> graphrag (retrieve)
                    -> llm (narrate) -> compliance (SFDR)

The FastAPI app holds one shared :class:`ESGEngine` so the graph is built once.
"""
from __future__ import annotations

from typing import Any

from .compliance.sfdr import classify_sfdr
from .crawlers import aws_s3, loader
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

    # -- curated output artifacts ---------------------------------------------
    def build_outputs(self) -> dict[str, Any]:
        """Assemble the curated result artifacts produced by a full run.

        These are the platform's outputs (portfolio scores, per-company ESG
        profiles, knowledge-graph stats) — the things worth persisting to the
        data lake's curated layer.
        """
        profiles = [self.company_profile(cid) for cid in self.companies]
        return {
            "portfolio_scores.json": self.portfolio_scores(),
            "company_profiles.json": [p for p in profiles if p],
            "graph_stats.json": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
            },
        }

    def export_results_to_s3(self) -> list[str]:
        """Write the curated result artifacts back to the S3 data lake.

        Closes the AWS loop: raw data is read from ``s3://<bucket>/data/raw`` and
        results are written to ``s3://<bucket>/data/curated`` (override the prefix
        with ``ESG_S3_OUTPUT_PREFIX``).
        """
        return [aws_s3.write_json(name, data) for name, data in self.build_outputs().items()]

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
