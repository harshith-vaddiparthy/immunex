# IMMUNEX Architecture

## System Overview

IMMUNEX is a multi-agent AI system designed to systematically identify drug repurposing opportunities for innate immune modulation. The system processes data from 9 biomedical databases through 5 specialized agents to produce ranked, evidence-graded repurposing hypotheses.

## Design Principles

1. **Agent Specialization:** Each agent handles one stage of the pipeline. This enables independent testing, development, and scaling.

2. **Evidence Convergence:** Hypotheses are scored based on how many independent evidence types support them. A drug flagged by literature, molecular prediction, AND knowledge graph link prediction scores higher than one found by a single source.

3. **Conservative Confidence:** The system explicitly distinguishes between established facts, strong predictions, and speculative hypotheses. False positives are more damaging than false negatives in drug repurposing.

4. **Continuous Operation:** Unlike one-shot screening tools, IMMUNEX runs continuously, updating hypotheses as new literature and data are published.

## Agent Architecture

```
                    ┌─────────────────┐
                    │   Data Sources   │
                    │  PubMed, ChEMBL, │
                    │  DrugBank, FAERS  │
                    └────────┬────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
        ┌───────▼──────┐ ┌──▼────────┐ ┌─▼──────────────┐
        │  Literature   │ │ Molecular │ │  Knowledge     │
        │  Scanner      │ │ Reasoner  │ │  Graph Builder │
        │  (Agent 1)    │ │ (Agent 2) │ │  (Agent 3)     │
        └───────┬──────┘ └──┬────────┘ └─┬──────────────┘
                │            │            │
                └────────────┼────────────┘
                             │
                  ┌──────────▼──────────┐
                  │   Hypothesis        │
                  │   Generator         │
                  │   (Agent 4)         │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │   Safety            │
                  │   Validator         │
                  │   (Agent 5)         │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │  Ranked Candidates  │
                  │  + Evidence Chains  │
                  │  + Safety Reports   │
                  └─────────────────────┘
```

## Agent Details

### Agent 1: Literature Scanner
- **Input:** PubMed queries targeting innate immune pathways + drug repurposing
- **Processing:** LLM-powered extraction of drug-pathway-effect triples
- **Output:** Structured associations with confidence scores and PMID citations
- **Technology:** NCBI E-utilities API + GPT-4 with structured output
- **Rate:** Can process ~100 articles per pathway per run

### Agent 2: Molecular Reasoner
- **Input:** Drug molecular fingerprints + protein sequence features
- **Processing:** Graph neural network predicts binding affinity
- **Output:** Ranked drug-target interaction scores
- **Technology:** PyTorch Geometric GNN, Morgan fingerprints (RDKit)
- **Training data:** ChEMBL (~2M labeled drug-target pairs)

### Agent 3: Knowledge Graph Builder
- **Input:** Multi-source biomedical data (DrugBank, ChEMBL, Reactome, InnateDB, literature)
- **Processing:** Heterogeneous graph construction + TransE/RotatE link prediction
- **Output:** Biomedical knowledge graph with predicted novel edges
- **Technology:** NetworkX (construction), PyTorch (embeddings)
- **Scale:** Target 5-10M nodes, 50-100M edges at full scale

### Agent 4: Hypothesis Generator
- **Input:** Outputs from Agents 1-3
- **Processing:** LLM-based multi-source evidence synthesis
- **Output:** Ranked hypotheses with mechanistic explanations and confidence tiers
- **Technology:** GPT-4 with structured JSON output, retrieval-augmented prompting
- **Quality control:** Three-tier confidence (high/medium/speculative)

### Agent 5: Safety Validator
- **Input:** Drug names from hypothesis generator
- **Processing:** FAERS adverse event querying + drug interaction checking
- **Output:** Safety reports with risk classification and flags
- **Technology:** openFDA REST API + rule-based flagging

## Data Flow

1. **Ingestion:** Literature Scanner and data fetching scripts pull from external sources
2. **Integration:** Knowledge Graph Builder merges all data into unified graph
3. **Analysis:** Molecular Reasoner scores drug-target pairs, KG embeddings predict novel links
4. **Synthesis:** Hypothesis Generator combines evidence from all sources
5. **Validation:** Safety Validator screens candidates for safety concerns
6. **Output:** Ranked candidates with full evidence chains, accessible via API and dashboard

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| ML Framework | PyTorch, PyTorch Geometric |
| Knowledge Graph | NetworkX + custom embeddings |
| LLM Integration | OpenAI API (GPT-4) |
| Molecular | RDKit, ChEMBL web client |
| API | FastAPI + Uvicorn |
| Data Sources | NCBI E-utilities, ChEMBL, openFDA, Reactome |

## Confidence Scoring

Hypotheses are assigned confidence tiers based on evidence convergence:

- **HIGH:** Supported by 2+ independent evidence types (e.g., literature + molecular prediction + KG link)
- **MEDIUM:** Supported by 1 strong source or 2+ weak sources
- **SPECULATIVE:** Computational prediction only, no literature support

Numeric confidence scores (0.0-1.0) provide finer granularity within each tier.
