# DS-03 · Autonomous ESG Intelligence Platform with GraphRAG

> A free, GraphRAG-based alternative to institutional ESG ratings — a runnable backend prototype.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![Tests](https://img.shields.io/badge/tests-7%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Project Overview

Asset managers must comply with **EU SFDR** and **SEC ESG** disclosure rules, yet
commercial ESG data (e.g. MSCI) costs **$50–200K/year** and manual analysis of a single
company's 10-K filings, sustainability reports and controversy databases takes **6–8 hours**.

This project is a **backend-only prototype** of an autonomous ESG intelligence platform
that does the same job at near-zero cost. It:

1. **Ingests** company, supply-chain and controversy data (SEC EDGAR / ESG report style).
2. **Builds a Knowledge Graph** of companies → suppliers → ESG events.
3. **Propagates supply-chain risk** so hidden upstream exposure (e.g. conflict minerals
   three tiers down) surfaces on the portfolio company.
4. Answers natural-language ESG questions with **GraphRAG** — multi-hop retrieval over the
   graph that standard vector RAG cannot do — returning grounded, evidence-cited narratives.
5. Auto-classifies each company under **SFDR Article 8 / 9 / 6** with clause-level
   justification.

Everything is exposed through a small **FastAPI** service. To stay a *college-level,
instantly-runnable prototype*, the heavyweight components from the reference architecture
are replaced by transparent, dependency-light stand-ins (see §4) **without changing the
architecture or the project structure** described in the source document.

---

## Table of Contents

1. **Project Structure**
   - 1.1 Directory Layout
   - 1.2 Layer-to-Architecture Mapping
2. **Architecture**
   - 2.1 High-Level Architecture Diagram
   - 2.2 Data Flow
   - 2.3 Technology Stack (Reference vs Prototype)
3. **Getting Started**
   - 3.1 Prerequisites
   - 3.2 Steps to Run This Project
   - 3.3 API Endpoints
4. **How It Works (Component Walkthrough)**
   - 4.1 Ingestion Layer
   - 4.2 NLP Layer
   - 4.3 Knowledge Graph + Risk Propagation
   - 4.4 GraphRAG Retrieval
   - 4.5 Narrative Generation
   - 4.6 SFDR Compliance Engine
5. **Program Output**
   - 5.1 Batch Pipeline Run
   - 5.2 API — Health & Graph Stats
   - 5.3 API — GraphRAG Query
   - 5.4 API — Company Profile
6. **Testing**
7. **Kaggle Notebook**

---

## 1. Project Structure

### 1.1 Directory Layout

This mirrors the GitHub repository structure defined in the portfolio document.

```
ds03-esg-intelligence/
├── data/raw/            # SEC EDGAR / ESG report & controversy snapshots (JSON)
├── src/crawlers/        # Data ingestion (Scrapy + Kafka stand-in)
├── src/nlp/             # ESG sentence classifier (FinBERT stand-in)
├── src/graph/           # Neo4j knowledge graph + GNN risk propagation (networkx)
├── src/graphrag/        # Microsoft GraphRAG retriever (TF-IDF + graph expansion)
├── src/llm/             # ESG narrative generator (Llama 3.1 / Ollama stand-in)
├── src/compliance/      # SFDR Article 8/9 classifier + EU Taxonomy mapping
├── src/engine.py        # Orchestrator wiring every layer together
├── api/                 # FastAPI ESG query + portfolio scoring service
├── pipelines/           # Batch pipeline (Airflow DAG stand-in)
├── dashboards/          # Power BI / Streamlit notes (backend serves the data)
├── notebooks/           # Exploration notebooks (pointer to Kaggle/)
├── tests/               # pytest suite
├── requirements.txt
└── README.md
```

**In brief:**
- **`data/raw/`** — the only input: four JSON files (companies, supply chain,
  controversies, documents).
- **`src/`** — one sub-package per architecture layer, so each layer can be swapped for
  its production tool independently.
- **`api/`** — the public HTTP surface; holds one shared graph built at startup.
- **`pipelines/` + `tests/`** — a console smoke-run and an automated test suite.

### 1.2 Layer-to-Architecture Mapping

| Folder            | Architecture layer            | Reference tool (PDF)            |
|-------------------|-------------------------------|---------------------------------|
| `src/crawlers/`   | Web crawling / streaming      | Scrapy + Playwright + Kafka     |
| `src/nlp/`        | NLP pipeline                  | HuggingFace FinBERT + SpaCy     |
| `src/graph/`      | Knowledge graph + Graph ML    | Neo4j + PyTorch Geometric GNN   |
| `src/graphrag/`   | RAG engine + Vector DB        | Microsoft GraphRAG + ChromaDB   |
| `src/llm/`        | Local LLM                     | Ollama + Llama 3.1 (8B)         |
| `src/compliance/` | Compliance engine             | Custom Python SFDR checker      |
| `api/`            | Serving                       | FastAPI                         |

---

## 2. Architecture

### 2.1 High-Level Architecture Diagram

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                       DS-03 ARCHITECTURE                             │
 └─────────────────────────────────────────────────────────────────────┘

   [Data Sources]            [NLP Processing]          [Knowledge Graph]
 ┌───────────────┐          ┌──────────────┐          ┌──────────────┐
 │ SEC EDGAR API │──Scrapy─▶│ FinBERT ESG  │────────▶ │ Neo4j Graph  │
 │ ESG Reports   │          │ Classification│         │  Companies   │
 │ News Feeds    │──Kafka──▶│ SpaCy NER    │────────▶ │  Suppliers   │
 │ Controversy DB│          │ Entity Extract│         │  ESG Events  │
 └───────────────┘          └──────────────┘          │ Controversies│
                                                       └──────┬───────┘
                                                              │
   [GNN Layer]              [GraphRAG + LLM]          [Output Layer]
 ┌───────────────┐          ┌──────────────┐          ┌──────────────┐
 │ PyTorch       │          │ MS GraphRAG  │──LLM───▶ │ ESG Score    │
 │ Geometric GNN │──risk──▶ │ + LangChain  │         │ SFDR Classify│
 │ Supply chain  │          │ Llama 3.1    │         │ Narratives   │
 │ propagation   │          │ (Ollama free)│         │ Power BI     │
 └───────────────┘          └──────────────┘          └──────────────┘
```

### 2.2 Data Flow

- Crawler fetches SEC EDGAR / ESG filings → stored as raw records (`data/raw/`).
- Controversy events stream in (Kafka in production; JSON snapshot here).
- NLP layer classifies each sentence as **Environmental / Social / Governance**.
- Knowledge graph is populated: `Company → has_controversy → ESGEvent` and
  `Company → sources_from → Supplier` edges.
- Graph ML **propagates ESG risk** from suppliers up to buyers across tiers (T1/T2/T3).
- **GraphRAG** retrieves the relevant documents *and* expands their supply-chain
  sub-graph for multi-hop reasoning.
- The LLM generates a **5-sentence, evidence-cited ESG narrative** per company.
- The compliance engine maps each company to its **SFDR article**.

### 2.3 Technology Stack (Reference vs Prototype)

| Layer            | Reference (PDF)                | This prototype                         |
|------------------|--------------------------------|----------------------------------------|
| Crawling/Stream  | Scrapy + Playwright + Kafka    | Deterministic JSON loader              |
| NLP              | FinBERT + SpaCy                | Keyword-weighted classifier            |
| Knowledge Graph  | Neo4j Community                | `networkx` `MultiDiGraph` (in-memory)  |
| Graph ML         | PyTorch Geometric GNN          | Tier-decayed risk propagation          |
| RAG + Vector DB  | Microsoft GraphRAG + ChromaDB  | TF-IDF retrieval + graph expansion     |
| LLM              | Ollama + Llama 3.1             | Template narrative generator           |
| Compliance       | Custom Python SFDR checker     | Same — rule engine (unchanged)         |
| Serving          | FastAPI                        | Same — FastAPI (unchanged)             |

> The architecture and folder structure are preserved exactly; only the *implementation*
> inside each layer is simplified so the project runs with no GPU, no Docker and no API keys.

---

## 3. Getting Started

### 3.1 Prerequisites

- **Python 3.10+** (developed and tested on 3.12)
- **pip**
- ~50 MB disk for dependencies
- No database, GPU, Docker or internet access required to run.

### 3.2 Steps to Run This Project

```bash
# 1. Move into the project
cd ds03-esg-intelligence

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4a. Run the batch pipeline (quick console smoke-test)
python -m pipelines.run_pipeline

# 4b. OR start the API server
uvicorn api.main:app --reload
# then open the interactive docs:
#   http://127.0.0.1:8000/docs

# 5. Run the test suite
pytest -q
```

### 3.3 API Endpoints

| Method | Path               | Purpose                                            |
|--------|--------------------|----------------------------------------------------|
| GET    | `/`                | Service metadata + endpoint index                  |
| GET    | `/health`          | Liveness probe                                     |
| GET    | `/portfolio`       | ESG + SFDR scoreboard for portfolio companies      |
| GET    | `/company/{id}`    | Full ESG profile, risk, SFDR & narrative           |
| POST   | `/query`           | GraphRAG natural-language ESG query                |
| GET    | `/graph/stats`     | Knowledge-graph size metrics                       |

---

## 4. How It Works (Component Walkthrough)

- **4.1 Ingestion Layer** (`src/crawlers/loader.py`) — reads the four raw JSON
  collections and caches them. This is the seam where Scrapy/Kafka would plug in.
- **4.2 NLP Layer** (`src/nlp/esg_classifier.py`) — labels each sentence E/S/G with a
  pseudo-confidence using weighted keyword lexicons (a FinBERT proxy with the same API).
- **4.3 Knowledge Graph + Risk Propagation** (`src/graph/knowledge_graph.py`) — builds a
  directed multigraph of companies, suppliers and controversies, then propagates each
  supplier's controversy risk upward with a **tier decay** (T1 60%, T2 30%, T3 15%).
- **4.4 GraphRAG Retrieval** (`src/graphrag/retriever.py`) — ranks documents by TF-IDF
  cosine similarity, then **expands the graph** around each hit so upstream suppliers and
  their controversies join the evidence — the multi-hop step plain vector RAG misses.
- **4.5 Narrative Generation** (`src/llm/narrative.py`) — turns the retrieved evidence
  into a short, grounded, cited ESG narrative (a deterministic Llama-3.1 stand-in).
- **4.6 SFDR Compliance Engine** (`src/compliance/sfdr.py`) — applies auditable thresholds
  on sustainable-investment %, composite ESG and fossil-fuel exposure to assign
  **Article 9 / 8 / 6** with a per-rule justification list.

---

## 5. Program Output

The output below is the **actual result of running this project** — copied verbatim from
the console and the live API.

### 5.1 Batch Pipeline — `python -m pipelines.run_pipeline`

```
======================================================================
DS-03 ESG INTELLIGENCE — BATCH PIPELINE
======================================================================

[1/3] Knowledge graph built: 17 nodes, 16 edges

[2/3] Portfolio ESG scoreboard (composite = base ESG - controversy/supply-chain risk):

Company                   Sector                  Base  Comp  SCRisk  SFDR
--------------------------------------------------------------------------
Helios Renewable Energy   Renewable Energy        83.0  80.1     7.2  Article 9
Verdant Foods Group       Consumer Staples        76.7  76.7     0.0  Article 8
Northwind Logistics       Transportation          68.3  59.5    12.0  Article 6
Meridian Semiconductors   Technology Hardware     65.0  53.5    28.8  Article 6
Atlas Heavy Industries    Industrials             53.0  33.5    28.8  Article 6

[3/3] Sample GraphRAG query:

Q: Which portfolio companies have conflict-mineral supply-chain exposure?

  • Meridian Semiconductors
    Meridian Semiconductors carries a composite ESG score of 53.5/100 (base 65.0), with a
    direct controversy risk of 0.0 and inherited supply-chain risk of 28.8. No direct
    controversies are on record for this entity. Upstream supply-chain exposure traces to
    Kivu Minerals Ltd (Tier 1: Kivu Minerals linked to conflict-mineral sourcing in
    artisanal mines; Tailings discharge contaminates local watershed near Kivu operations)
    | Pacific Circuit Assembly (Tier 1: Pacific Circuit Assembly cited for excessive
    overtime at two plants). This assessment is grounded in retrieved filing evidence and
    is traceable to the cited source documents.

  • Verdant Foods Group
    Verdant Foods Group carries a composite ESG score of 76.7/100 (base 76.7), with a
    direct controversy risk of 0.0 and inherited supply-chain risk of 0.0. No direct
    controversies are on record for this entity. No controversy exposure was found in the
    upstream supply chain. This assessment is grounded in retrieved filing evidence and is
    traceable to the cited source documents.

  • Atlas Heavy Industries
    Atlas Heavy Industries carries a composite ESG score of 33.5/100 (base 53.0), with a
    direct controversy risk of 16.0 and inherited supply-chain risk of 28.8. Direct
    controversies on record: Atlas Heavy Industries exceeds permitted NOx emissions at
    flagship plant. Upstream supply-chain exposure traces to Summit Steelworks (Tier 1:
    Summit Steelworks under antitrust review over regional price coordination) | Kivu
    Minerals Ltd (Tier 1: Kivu Minerals linked to conflict-mineral sourcing in artisanal
    mines; Tailings discharge contaminates local watershed near Kivu operations). This
    assessment is grounded in retrieved filing evidence and is traceable to the cited
    source documents.

======================================================================
Pipeline complete.
======================================================================
```

### 5.2 API — `GET /health` and `GET /graph/stats`

```json
{ "status": "ok", "graph_nodes": 17 }
```

```json
{
  "nodes": 17,
  "edges": 16,
  "node_kinds": { "company": 10, "controversy": 7 }
}
```

### 5.3 API — `POST /query`

Request body: `{"question": "Which portfolio companies have conflict-mineral exposure?", "top_k": 2}`

```json
{
  "question": "Which portfolio companies have conflict-mineral exposure?",
  "retrieved_documents": [
    { "company_id": "C003", "source": "SEC EDGAR 10-K / Supplier Disclosure", "relevance": 0.218 },
    { "company_id": "C005", "source": "SEC EDGAR 10-K / Regulator Filing", "relevance": 0.075 }
  ],
  "answers": [
    {
      "company_id": "C003",
      "name": "Meridian Semiconductors",
      "narrative": "Meridian Semiconductors carries a composite ESG score of 53.5/100 (base 65.0), with a direct controversy risk of 0.0 and inherited supply-chain risk of 28.8. No direct controversies are on record for this entity. Upstream supply-chain exposure traces to Pacific Circuit Assembly (Tier 1: Pacific Circuit Assembly cited for excessive overtime at two plants) | Kivu Minerals Ltd (Tier 1: Kivu Minerals linked to conflict-mineral sourcing in artisanal mines; Tailings discharge contaminates local watershed near Kivu operations). This assessment is grounded in retrieved filing evidence and is traceable to the cited source documents."
    },
    {
      "company_id": "C005",
      "name": "Atlas Heavy Industries",
      "narrative": "Atlas Heavy Industries carries a composite ESG score of 33.5/100 (base 53.0), with a direct controversy risk of 16.0 and inherited supply-chain risk of 28.8. Direct controversies on record: Atlas Heavy Industries exceeds permitted NOx emissions at flagship plant. Upstream supply-chain exposure traces to Kivu Minerals Ltd (Tier 1: Kivu Minerals linked to conflict-mineral sourcing in artisanal mines; Tailings discharge contaminates local watershed near Kivu operations) | Summit Steelworks (Tier 1: Summit Steelworks under antitrust review over regional price coordination). This assessment is grounded in retrieved filing evidence and is traceable to the cited source documents."
    }
  ]
}
```

### 5.4 API — `GET /company/C005`

```json
{
  "company": {
    "id": "C005",
    "name": "Atlas Heavy Industries",
    "ticker": "ATHI",
    "sector": "Industrials",
    "region": "United States",
    "in_portfolio": true,
    "environmental_score": 44,
    "social_score": 55,
    "governance_score": 60,
    "sustainable_investment_pct": 0.18,
    "fossil_fuel_exposure": true
  },
  "risk": {
    "company_id": "C005",
    "name": "Atlas Heavy Industries",
    "base_esg": 53.0,
    "direct_controversy_risk": 16.0,
    "supply_chain_risk": 28.8,
    "composite_esg": 33.5,
    "supplier_contributors": [
      { "supplier_id": "S101", "supplier_name": "Kivu Minerals Ltd", "tier": 1, "supplier_risk": 36.0, "propagated_risk": 21.6 },
      { "supplier_id": "S105", "supplier_name": "Summit Steelworks", "tier": 1, "supplier_risk": 12.0, "propagated_risk": 7.2 }
    ]
  },
  "sfdr": {
    "sfdr_article": "Article 6",
    "classification": "No sustainability claim",
    "composite_esg_used": 33.5,
    "justifications": [
      "Sustainable investment 18% < 30% minimum.",
      "Composite ESG 33.5 < 60 threshold.",
      "Fossil-fuel exposure present."
    ]
  },
  "narrative": "Atlas Heavy Industries carries a composite ESG score of 33.5/100 (base 53.0), with a direct controversy risk of 16.0 and inherited supply-chain risk of 28.8. Direct controversies on record: Atlas Heavy Industries exceeds permitted NOx emissions at flagship plant. Upstream supply-chain exposure traces to Kivu Minerals Ltd (Tier 1: Kivu Minerals linked to conflict-mineral sourcing in artisanal mines; Tailings discharge contaminates local watershed near Kivu operations) | Summit Steelworks (Tier 1: Summit Steelworks under antitrust review over regional price coordination). This assessment is grounded in retrieved filing evidence and is traceable to the cited source documents."
}
```

---

## 6. Testing

```bash
pytest -q
```

```
.......                                                                  [100%]
7 passed in 8.43s
```

The suite covers graph construction, the NLP classifier, supply-chain risk propagation,
SFDR classification (Article 9 and Article 6 paths), and the end-to-end GraphRAG query.

---

## 7. Kaggle Notebook

A presentation-ready notebook — **"Building a Free Alternative to MSCI ESG Ratings"** —
walks through the entire pipeline with charts and explanations:

```
Kaggle/esg-graphrag-intelligence.ipynb
```

It reproduces the knowledge graph, supply-chain risk propagation, GraphRAG query and SFDR
classification end-to-end, designed to be uploaded and run on Kaggle.

---

### License

MIT — free for educational and commercial use.
