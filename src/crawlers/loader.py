"""Data loader.

In the full platform (see PDF architecture) this layer is a Scrapy + Playwright
crawler over SEC EDGAR / ESG reports with Kafka streaming controversy events.

For this prototype the same role is played by a deterministic loader that reads the
pre-collected JSON snapshots in ``data/raw``. The rest of the pipeline does not care
where the records came from, so the simplification is transparent.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _read(name: str) -> Any:
    path = DATA_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_companies() -> list[dict]:
    """Company + supplier master records (SEC EDGAR / ESG report stand-in)."""
    return _read("companies.json")


@lru_cache(maxsize=1)
def load_supply_chain() -> list[dict]:
    """Buyer -> supplier edges (OpenCorporates supply-chain stand-in)."""
    return _read("supply_chain.json")


@lru_cache(maxsize=1)
def load_controversies() -> list[dict]:
    """Streamed controversy events (Kafka controversy-monitor stand-in)."""
    return _read("controversies.json")


@lru_cache(maxsize=1)
def load_documents() -> list[dict]:
    """ESG narrative source documents used by the retrieval layer."""
    return _read("esg_documents.json")


def load_all() -> dict[str, list[dict]]:
    """Convenience bundle of every raw collection."""
    return {
        "companies": load_companies(),
        "supply_chain": load_supply_chain(),
        "controversies": load_controversies(),
        "documents": load_documents(),
    }
