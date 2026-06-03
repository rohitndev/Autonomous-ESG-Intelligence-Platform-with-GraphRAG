"""ESG Knowledge Graph + supply-chain risk propagation.

Full platform:  Neo4j Community Edition stores Company / Supplier / ESGEvent nodes
and a PyTorch-Geometric GNN propagates ESG risk through supply-chain tiers.

Prototype:  the identical graph is built in-memory with **networkx** (a directed
multigraph) and risk is propagated with a transparent tier-decayed sum instead of a
learned GNN. The graph object exposes the same neighbourhood queries the GraphRAG
retriever needs, so the abstraction matches the architecture diagram in the PDF.
"""
from __future__ import annotations

import networkx as nx

from ..crawlers import loader

# How much of a supplier's controversy risk flows up to the buyer, per tier hop.
TIER_DECAY = {1: 0.60, 2: 0.30, 3: 0.15}
DEFAULT_DECAY = 0.10


def build_graph() -> nx.MultiDiGraph:
    """Construct the ESG knowledge graph from the raw collections."""
    g = nx.MultiDiGraph()
    data = loader.load_all()

    # Company / supplier nodes
    for c in data["companies"]:
        g.add_node(
            c["id"],
            kind="company",
            **{k: v for k, v in c.items() if k != "id"},
            base_esg=round(
                (c["environmental_score"] + c["social_score"] + c["governance_score"]) / 3,
                1,
            ),
        )

    # Controversy event nodes + edges
    for ev in data["controversies"]:
        g.add_node(ev["id"], kind="controversy", **{k: v for k, v in ev.items() if k != "id"})
        g.add_edge(ev["company_id"], ev["id"], relation="has_controversy")

    # Supply-chain edges (buyer -> supplier)
    for edge in data["supply_chain"]:
        g.add_edge(
            edge["buyer"],
            edge["supplier"],
            relation="sources_from",
            tier=edge["tier"],
            commodity=edge["commodity"],
        )

    return g


def _direct_controversy_risk(g: nx.MultiDiGraph, node: str) -> float:
    """Sum of severities of controversies attached directly to a node (0-100 scale)."""
    risk = 0.0
    for _, target, attrs in g.out_edges(node, data=True):
        if attrs.get("relation") == "has_controversy":
            risk += g.nodes[target]["severity"] * 4  # severity 1-5 -> 0-20 each
    return min(risk, 100.0)


def propagate_supply_chain_risk(g: nx.MultiDiGraph) -> dict[str, dict]:
    """Propagate controversy risk from suppliers up to buyers (GNN proxy).

    Returns a per-company dict with direct risk, inherited supply-chain risk and the
    contributing supplier chain — this is the "hidden portfolio exposure" the PDF
    describes the GNN surfacing.
    """
    result: dict[str, dict] = {}
    for node, attrs in g.nodes(data=True):
        if attrs.get("kind") != "company":
            continue

        direct = _direct_controversy_risk(g, node)
        inherited = 0.0
        contributors = []

        # Walk every supplier reachable from this company.
        for supplier in nx.descendants(g, node):
            if g.nodes[supplier].get("kind") != "company":
                continue
            # tier = shortest sourcing distance in hops
            try:
                hops = nx.shortest_path_length(g, node, supplier)
            except nx.NetworkXNoPath:
                continue
            decay = TIER_DECAY.get(hops, DEFAULT_DECAY)
            supplier_risk = _direct_controversy_risk(g, supplier)
            flowed = supplier_risk * decay
            if flowed > 0:
                inherited += flowed
                contributors.append(
                    {
                        "supplier_id": supplier,
                        "supplier_name": g.nodes[supplier]["name"],
                        "tier": hops,
                        "supplier_risk": round(supplier_risk, 1),
                        "propagated_risk": round(flowed, 1),
                    }
                )

        inherited = min(inherited, 100.0)
        contributors.sort(key=lambda x: x["propagated_risk"], reverse=True)
        result[node] = {
            "company_id": node,
            "name": attrs["name"],
            "base_esg": attrs["base_esg"],
            "direct_controversy_risk": round(direct, 1),
            "supply_chain_risk": round(inherited, 1),
            # Composite ESG: base score penalised by direct + inherited risk.
            "composite_esg": round(max(0.0, attrs["base_esg"] - 0.5 * direct - 0.4 * inherited), 1),
            "supplier_contributors": contributors,
        }
    return result
