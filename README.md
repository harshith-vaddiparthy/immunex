<div align="center">

# IMMUNEX

**Autonomous Multi-Agent AI System for Drug Repurposing in Innate Immunity**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Encode Fellowship](https://img.shields.io/badge/Encode-AI%20for%20Science-purple.svg)](https://encode.pillar.vc)

[Architecture](docs/ARCHITECTURE.md) | [API Docs](#api) | [Data Sources](#data-sources) | [Landing Page](https://harshith.com/immunex)

</div>

---

90% of drugs fail clinical trials. Meanwhile, **35% of FDA-designated "transformative" drugs are repurposed products**. The innate immune system - our first line of defense against infection, inflammation, and chronic disease - remains massively underexplored therapeutically. Hundreds of approved drugs have documented but unstudied effects on innate immune pathways.

**IMMUNEX makes drug repurposing systematic instead of serendipitous.**

It continuously scans biomedical literature, molecular databases, and clinical evidence through five specialized AI agents to produce ranked, evidence-graded repurposing hypotheses for innate immune modulation.

## How It Works

IMMUNEX orchestrates five specialized agents in a pipeline:

| # | Agent | What It Does | Status |
|---|-------|-------------|--------|
| 1 | **Literature Scanner** | Mines PubMed (36M+ articles) for drug-innate immunity associations using LLM-powered NLP | ✅ Operational |
| 2 | **Molecular Reasoner** | GNN predicts drug-target binding for innate immune targets (NLRP3, STING, TLR4, etc.) | 🔄 Training |
| 3 | **Knowledge Graph** | Builds heterogeneous biomedical graph from 9 data sources; link prediction via TransE/RotatE | ✅ Operational |
| 4 | **Hypothesis Generator** | LLM-based evidence synthesis into ranked hypotheses with confidence tiers | ✅ Operational |
| 5 | **Safety Validator** | Checks candidates against FDA FAERS (20M+ adverse event reports) and drug interactions | ✅ Operational |

```
PubMed/bioRxiv ──→ Literature Scanner ──┐
                                        │
ChEMBL/BindingDB ──→ Molecular Reasoner ──→ Hypothesis Generator ──→ Safety Validator ──→ Ranked Candidates
                                        │
DrugBank/Reactome ──→ Knowledge Graph ──┘
```

## Current Results

The system has already processed real biomedical data:

| Metric | Count |
|--------|-------|
| PubMed articles scanned | 78 (across 7 pathways) |
| ChEMBL bioactivity records | 808 (for 11 innate immune targets) |
| Curated protein-protein interactions | 28 (with PMID citations) |
| Reference drugs with innate immune evidence | 15 |
| Reactome pathway participants | 2,369 |
| Innate immune targets tracked | 24 |
| Pathways covered | 7 |

## Target Pathways

| Pathway | Key Targets | Therapeutic Relevance |
|---------|------------|----------------------|
| TLR Signaling | TLR2, TLR4, TLR7, TLR9 | Pathogen recognition, inflammatory initiation |
| Inflammasome | NLRP3, NLRC4, AIM2, Caspase-1 | IL-1β/IL-18 processing, pyroptosis |
| cGAS-STING | cGAS, STING, TBK1 | DNA sensing, type I interferon response |
| NF-κB | IKKβ, RelA, NEMO | Master inflammatory transcription regulator |
| JAK-STAT | JAK1, JAK2, TYK2, STAT1 | Cytokine and interferon signal transduction |
| Trained Immunity | mTOR, HIF-1α, KDM5 | Epigenetic reprogramming of innate immune cells |
| Complement | C3, C5, Factor D | Innate opsonization and inflammatory cascading |

## Quick Start

```bash
# Clone
git clone https://github.com/harshith-vaddiparthy/immunex.git
cd immunex

# Install
pip install -r requirements.txt

# Fetch biomedical data (ChEMBL, Reactome, InnateDB)
python scripts/fetch_data.py

# Scan PubMed for a specific pathway
python -m src.agents.literature_scanner --pathway inflammasome --max-results 50

# Run the full pipeline
python -m src.pipeline --pathway inflammasome --output results/

# Start the API dashboard
python -m src.api.server
# Open http://localhost:8051
```

## Data Sources

All primary data sources are publicly available:

| Source | Records | Type | Access |
|--------|---------|------|--------|
| PubMed/MEDLINE | 36M+ articles | Biomedical literature | Free API |
| ChEMBL | 2.4M compounds | Drug bioactivity | Open access |
| DrugBank | 14,000+ drugs | Drug information | Academic license |
| BindingDB | 2.9M data points | Binding affinity | Open access |
| Reactome | 2,600+ pathways | Biological pathways | Open access |
| InnateDB | 32,000+ interactions | Innate immunity | Open access |
| KEGG | 500+ pathways | Metabolic/signaling | Open access |
| ClinicalTrials.gov | 500K+ studies | Clinical trials | Public |
| FDA FAERS | 20M+ reports | Adverse events | Public |

## Project Structure

```
immunex/
├── src/
│   ├── agents/
│   │   ├── literature_scanner.py     # PubMed mining + LLM extraction
│   │   ├── molecular_reasoner.py     # GNN drug-target prediction
│   │   ├── hypothesis_generator.py   # Evidence synthesis + ranking
│   │   └── safety_validator.py       # FAERS + interaction checking
│   ├── knowledge_graph/
│   │   └── builder.py                # Multi-source graph construction
│   ├── models/
│   │   ├── gnn.py                    # Drug-target GNN architecture
│   │   └── embeddings.py             # TransE/RotatE for link prediction
│   ├── api/
│   │   └── server.py                 # FastAPI dashboard + REST API
│   ├── utils/                        # Data source clients
│   └── pipeline.py                   # Full pipeline orchestration
├── data/                             # Fetched biomedical data
├── results/                          # Pipeline output + scan results
├── scripts/
│   └── fetch_data.py                 # Data fetching pipeline
├── docs/
│   └── ARCHITECTURE.md               # System architecture documentation
└── tests/                            # Test suite
```

## API

Start the dashboard and API:

```bash
python -m src.api.server
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Interactive dashboard |
| `GET` | `/health` | System health + KG statistics |
| `GET` | `/targets` | All 24 innate immune targets |
| `GET` | `/pathways` | All 7 tracked pathways |
| `GET` | `/candidates` | Ranked repurposing candidates |
| `GET` | `/kg/stats` | Knowledge graph statistics |

## ARIA Alignment

This project aligns with [ARIA's Sculpting Innate Immunity](https://aria.org.uk/opportunity-spaces/sculpting-innate-immunity/) opportunity space by systematically identifying existing drugs that can modulate innate immune pathways with precision. Directly applicable to ARIA's Sustained Viral Resilience programme.

## References

1. Pushpakom, S. et al. "Drug repurposing: progress, challenges and recommendations." *Nature Reviews Drug Discovery* 18, 41-58 (2019). [DOI: 10.1038/nrd.2018.168](https://doi.org/10.1038/nrd.2018.168)
2. Stokes, J.M. et al. "A Deep Learning Approach to Antibiotic Discovery." *Cell* 180, 688-702 (2020). [DOI: 10.1016/j.cell.2020.01.021](https://doi.org/10.1016/j.cell.2020.01.021)
3. Netea, M.G. et al. "Trained immunity: a program of innate immune memory in health and disease." *Science* 352, aaf1098 (2016). [DOI: 10.1126/science.aaf1098](https://doi.org/10.1126/science.aaf1098)

## License

MIT

---

<div align="center">

Built by [Harshith Vaddiparthy](https://harshith.com) for the [Encode: AI for Science Fellowship](https://encode.pillar.vc) (Pillar VC x ARIA)

[LinkedIn](https://linkedin.com/in/harshith-vaddiparthy) | [Twitter/X](https://x.com/harshith) | [Website](https://harshith.com/immunex)

</div>
