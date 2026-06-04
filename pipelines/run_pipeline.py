"""End-to-end batch pipeline (Airflow `daily_crawl -> weekly_graph_refresh` proxy).

Runs the whole platform once and prints a portfolio ESG report to the console — handy
for a quick smoke test without starting the API.

    python -m pipelines.run_pipeline
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import ESGEngine  # noqa: E402


def main() -> None:
    print("=" * 70)
    print("AUTONOMOUS ESG INTELLIGENCE PLATFORM — BATCH PIPELINE")
    print("=" * 70)

    engine = ESGEngine()
    print(f"\n[1/3] Knowledge graph built: {engine.graph.number_of_nodes()} nodes, "
          f"{engine.graph.number_of_edges()} edges")

    print("\n[2/3] Portfolio ESG scoreboard (composite = base ESG - controversy/supply-chain risk):\n")
    header = f"{'Company':<26}{'Sector':<22}{'Base':>6}{'Comp':>6}{'SCRisk':>8}  SFDR"
    print(header)
    print("-" * len(header))
    for row in engine.portfolio_scores():
        print(
            f"{row['name']:<26}{row['sector']:<22}"
            f"{row['base_esg']:>6}{row['composite_esg']:>6}{row['supply_chain_risk']:>8}  "
            f"{row['sfdr_article']}"
        )

    print("\n[3/3] Sample GraphRAG query:\n")
    result = engine.query("Which portfolio companies have conflict-mineral supply-chain exposure?")
    print(f"Q: {result['question']}\n")
    for ans in result["answers"]:
        print(f"  • {ans['name']}")
        print(f"    {ans['narrative']}\n")

    print("=" * 70)
    print("Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
