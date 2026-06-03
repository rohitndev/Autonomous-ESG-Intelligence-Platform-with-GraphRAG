"""FastAPI application — the public face of the ESG Intelligence Platform.

Run with:  uvicorn api.main:app --reload
Docs at:   http://127.0.0.1:8000/docs

Endpoints
---------
GET  /                     service metadata + endpoint index
GET  /health               liveness probe
GET  /portfolio            ESG + SFDR scoreboard for portfolio companies
GET  /company/{id}         full ESG profile, risk, SFDR & narrative for one company
POST /query                GraphRAG natural-language ESG query
GET  /graph/stats          knowledge-graph size metrics
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Allow `uvicorn api.main:app` to import the sibling `src` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import ESGEngine  # noqa: E402

app = FastAPI(
    title="DS-03 Autonomous ESG Intelligence Platform",
    description="A free, GraphRAG-based alternative to institutional ESG ratings (prototype).",
    version="1.0.0",
)

# Build the knowledge graph once at startup and share it across requests.
engine = ESGEngine()


class QueryRequest(BaseModel):
    question: str = Field(..., examples=["Which portfolio companies have conflict-mineral exposure?"])
    top_k: int = Field(3, ge=1, le=10)


@app.get("/")
def root() -> dict:
    return {
        "service": "DS-03 Autonomous ESG Intelligence Platform",
        "version": "1.0.0",
        "endpoints": ["/health", "/portfolio", "/company/{id}", "/query", "/graph/stats", "/docs"],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "graph_nodes": engine.graph.number_of_nodes()}


@app.get("/portfolio")
def portfolio() -> dict:
    return {"portfolio": engine.portfolio_scores()}


@app.get("/company/{company_id}")
def company(company_id: str) -> dict:
    profile = engine.company_profile(company_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Unknown company id: {company_id}")
    return profile


@app.post("/query")
def query(req: QueryRequest) -> dict:
    return engine.query(req.question, top_k=req.top_k)


@app.get("/graph/stats")
def graph_stats() -> dict:
    g = engine.graph
    kinds: dict[str, int] = {}
    for _, attrs in g.nodes(data=True):
        kinds[attrs.get("kind", "unknown")] = kinds.get(attrs.get("kind", "unknown"), 0) + 1
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "node_kinds": kinds,
    }
