"""Smoke + correctness tests for the ESG Intelligence pipeline.

Run with:  pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from src.compliance.sfdr import classify_sfdr  # noqa: E402
from src.engine import ESGEngine  # noqa: E402
from src.graph.knowledge_graph import build_graph, propagate_supply_chain_risk  # noqa: E402
from src.nlp.esg_classifier import classify_sentence  # noqa: E402


@pytest.fixture(scope="module")
def engine() -> ESGEngine:
    return ESGEngine()


def test_graph_builds_with_expected_node_kinds():
    g = build_graph()
    kinds = {a.get("kind") for _, a in g.nodes(data=True)}
    assert "company" in kinds
    assert "controversy" in kinds
    assert g.number_of_nodes() > 10


def test_nlp_classifier_detects_pillars():
    assert classify_sentence("The company exceeded permitted NOx emissions.")["label"] == "Environmental"
    assert classify_sentence("Cited for conflict minerals and labor rights abuses.")["label"] == "Social"
    assert classify_sentence("The board faces an antitrust governance review.")["label"] == "Governance"


def test_supply_chain_risk_propagates_to_buyer():
    g = build_graph()
    risk = propagate_supply_chain_risk(g)
    # Meridian (C003) sources from Kivu Minerals (S101) which has conflict-mineral events.
    assert risk["C003"]["supply_chain_risk"] > 0
    contributors = {c["supplier_id"] for c in risk["C003"]["supplier_contributors"]}
    assert "S101" in contributors


def test_sfdr_article9_for_green_company():
    green = {
        "environmental_score": 88, "social_score": 79, "governance_score": 82,
        "sustainable_investment_pct": 0.91, "fossil_fuel_exposure": False,
    }
    assert classify_sfdr(green, composite_esg=83)["sfdr_article"] == "Article 9"


def test_sfdr_article6_for_fossil_company():
    brown = {
        "environmental_score": 44, "social_score": 55, "governance_score": 60,
        "sustainable_investment_pct": 0.18, "fossil_fuel_exposure": True,
    }
    assert classify_sfdr(brown, composite_esg=40)["sfdr_article"] == "Article 6"


def test_graphrag_query_returns_grounded_answers(engine: ESGEngine):
    result = engine.query("conflict mineral supply chain exposure", top_k=3)
    assert result["answers"]
    assert all("narrative" in a for a in result["answers"])


def test_company_profile_has_full_payload(engine: ESGEngine):
    profile = engine.company_profile("C003")
    assert profile is not None
    assert "sfdr" in profile and "risk" in profile and "narrative" in profile
