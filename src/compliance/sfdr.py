"""SFDR Article 8 / 9 auto-classifier.

Implements a transparent, clause-level rule engine mapping each company to its EU
SFDR (Sustainable Finance Disclosure Regulation) category, with the justification for
every decision — matching the "clause-level justification" requirement in the PDF.

  * Article 9  ("dark green")  — sustainable investment is the objective.
  * Article 8  ("light green") — promotes E/S characteristics, no harmful exposure.
  * Article 6  ("no claim")    — does not meet the Article 8 bar.

This is the rule-based "Custom Python SFDR checker" from the technology-stack table.
"""
from __future__ import annotations

# Decision thresholds (documented so the classification is auditable).
ART9_MIN_SUSTAINABLE = 0.80
ART9_MIN_ESG = 75
ART8_MIN_SUSTAINABLE = 0.30
ART8_MIN_ESG = 60


def classify_sfdr(company: dict, composite_esg: float | None = None) -> dict:
    """Classify one company record into an SFDR article with justifications."""
    base_esg = (
        company["environmental_score"]
        + company["social_score"]
        + company["governance_score"]
    ) / 3
    esg = composite_esg if composite_esg is not None else base_esg
    sustainable = company["sustainable_investment_pct"]
    fossil = company["fossil_fuel_exposure"]

    reasons: list[str] = []

    # ---- Article 9 test --------------------------------------------------
    if sustainable >= ART9_MIN_SUSTAINABLE and esg >= ART9_MIN_ESG and not fossil:
        reasons.append(
            f"Sustainable investment {sustainable:.0%} >= {ART9_MIN_SUSTAINABLE:.0%} "
            "(sustainable investment as the objective)."
        )
        reasons.append(f"Composite ESG {esg:.1f} >= {ART9_MIN_ESG} threshold.")
        reasons.append("No fossil-fuel exposure — consistent with EU Taxonomy alignment.")
        return _result("Article 9", "Sustainable investment objective", reasons, esg)

    # ---- Article 8 test --------------------------------------------------
    if sustainable >= ART8_MIN_SUSTAINABLE and esg >= ART8_MIN_ESG:
        reasons.append(
            f"Sustainable investment {sustainable:.0%} >= {ART8_MIN_SUSTAINABLE:.0%} "
            "(promotes E/S characteristics)."
        )
        reasons.append(f"Composite ESG {esg:.1f} >= {ART8_MIN_ESG} threshold.")
        if fossil:
            reasons.append("Note: residual fossil-fuel exposure caps the entity below Article 9.")
        else:
            reasons.append("No fossil-fuel exposure.")
        return _result("Article 8", "Promotes E/S characteristics", reasons, esg)

    # ---- Article 6 (fallback) -------------------------------------------
    if sustainable < ART8_MIN_SUSTAINABLE:
        reasons.append(
            f"Sustainable investment {sustainable:.0%} < {ART8_MIN_SUSTAINABLE:.0%} minimum."
        )
    if esg < ART8_MIN_ESG:
        reasons.append(f"Composite ESG {esg:.1f} < {ART8_MIN_ESG} threshold.")
    if fossil:
        reasons.append("Fossil-fuel exposure present.")
    return _result("Article 6", "No sustainability claim", reasons, esg)


def _result(article: str, label: str, reasons: list[str], esg: float) -> dict:
    return {
        "sfdr_article": article,
        "classification": label,
        "composite_esg_used": round(esg, 1),
        "justifications": reasons,
    }
