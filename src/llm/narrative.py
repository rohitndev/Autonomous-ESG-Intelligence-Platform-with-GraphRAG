"""ESG narrative generator.

Full platform:  Llama 3.1 (8B) served locally via Ollama turns the retrieved
evidence chain into a grounded, cited ESG narrative.

Prototype:  a deterministic template generator composes the same style of grounded,
cited summary from the GraphRAG evidence. It is free, offline and reproducible — and
because it consumes the *same evidence structure* the retriever produces, replacing it
with a real Ollama call later is a drop-in change to :func:`generate_narrative`.
"""
from __future__ import annotations


def generate_narrative(evidence: dict, company_risk: dict | None = None) -> str:
    """Produce a short, evidence-cited ESG risk narrative for one company."""
    name = evidence["name"]
    parts: list[str] = []

    # Sentence 1 — headline ESG / risk position.
    if company_risk:
        parts.append(
            f"{name} carries a composite ESG score of {company_risk['composite_esg']}/100 "
            f"(base {company_risk['base_esg']}), with a direct controversy risk of "
            f"{company_risk['direct_controversy_risk']} and inherited supply-chain risk of "
            f"{company_risk['supply_chain_risk']}."
        )
    else:
        parts.append(f"ESG evidence summary for {name}.")

    # Sentence 2 — direct controversies.
    if evidence["controversies"]:
        joined = "; ".join(evidence["controversies"])
        parts.append(f"Direct controversies on record: {joined}.")
    else:
        parts.append("No direct controversies are on record for this entity.")

    # Sentence 3 — multi-hop supply-chain exposure (the GraphRAG differentiator).
    flagged = [s for s in evidence["supply_chain"] if s["controversies"]]
    if flagged:
        chains = []
        for s in flagged:
            chains.append(
                f"{s['supplier_name']} (Tier {s['tier']}: " + "; ".join(s["controversies"]) + ")"
            )
        parts.append("Upstream supply-chain exposure traces to " + " | ".join(chains) + ".")
    else:
        parts.append("No controversy exposure was found in the upstream supply chain.")

    # Sentence 4 — evidence provenance.
    parts.append(
        "This assessment is grounded in retrieved filing evidence and is traceable to "
        "the cited source documents."
    )
    return " ".join(parts)
