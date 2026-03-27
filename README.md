# IMMUNEX

**Autonomous Multi-Agent AI System for Drug Repurposing in Innate Immunity**

IMMUNEX identifies existing approved drugs that can be repurposed as innate immune modulators. It continuously scans biomedical literature, molecular databases, and clinical evidence to generate ranked repurposing hypotheses with full mechanistic explanations.

> 90% of drugs fail clinical trials. Meanwhile, 35% of FDA-designated "transformative" drugs are repurposed products. IMMUNEX makes drug repurposing systematic instead of serendipitous.

## Why Innate Immunity?

The adaptive immune system gave us vaccines and checkpoint inhibitors. The innate immune system - our first line of defense against infection, inflammation, and chronic disease - remains an underexplored therapeutic frontier. Hundreds of approved drugs have documented but unstudied effects on innate immune pathways (toll-like receptors, inflammasomes, trained immunity, interferon signaling). IMMUNEX finds them.

## Architecture

IMMUNEX is not a single model. It's an orchestrated system of five specialized AI agents:

```
┌─────────────────────────────────────────────────────────┐
│                    IMMUNEX Pipeline                      │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │
│  │  Literature   │   │  Molecular   │   │ Knowledge  │  │
│  │   Scanner     │   │  Reasoner    │   │   Graph    │  │
│  │              │   │              │   │  Builder   │  │
│  │  PubMed +    │   │  GNN-based   │   │            │  │
│  │  bioRxiv     │   │  drug-target │   │ Heterogen. │  │
│  │  extraction  │   │  prediction  │   │ biomedical │  │
│  └──────┬───────┘   └──────┬───────┘   └─────┬──────┘  │
│         │                  │                  │         │
│         └──────────┬───────┴──────────────────┘         │
│                    │                                    │
│         ┌──────────▼──────────┐                         │
│         │    Hypothesis       │                         │
│         │    Generator        │                         │
│         │                    │                         │
│         │  LLM reasoning +   │                         │
│         │  evidence synthesis │                         │
│         └──────────┬──────────┘                         │
│                    │                                    │
│         ┌──────────▼──────────┐                         │
│         │     Safety          │                         │
│         │     Validator       │                         │
│         │                    │                         │
│         │  FAERS + drug      │                         │
│         │  interactions      │                         │
│         └─────────────────────┘                         │
│                    │                                    │
│                    ▼                                    │
│         Ranked Repurposing Candidates                   │
│         + Mechanistic Hypotheses                        │
└─────────────────────────────────────────────────────────┘
```

### Agent 1: Literature Scanner
Continuously ingests PubMed (36M+ articles) and bioRxiv preprints. Extracts structured drug-pathway-effect triples for innate immune pathways using LLM-powered NLP. Outputs confidence-scored associations with source citations.

### Agent 2: Molecular Reasoner
Graph neural network trained on drug-target interaction data from ChEMBL (2.4M compounds) and BindingDB. Predicts binding affinity between drugs and innate immune targets (NLRP3, STING, cGAS, TLR4, etc.). Performs off-target analysis to flag unexpected innate immune effects.

### Agent 3: Knowledge Graph Builder
Constructs a heterogeneous biomedical knowledge graph linking drugs, targets, pathways, diseases, and clinical outcomes. Uses knowledge graph embeddings (TransE/RotatE) for link prediction to identify undocumented drug-pathway associations.

### Agent 4: Hypothesis Generator
LLM-based reasoning agent that synthesizes outputs from Agents 1-3 into ranked repurposing hypotheses. Each hypothesis includes the drug, proposed mechanism, target condition, supporting evidence chain, confidence tier, and suggested validation experiments.

### Agent 5: Safety Validator
Cross-references candidates against FDA FAERS adverse event database, known drug interactions, and contraindications. Flags problematic safety signals before candidates reach human review.

## Data Sources

| Source | Type | Size | Access |
|--------|------|------|--------|
| PubMed/MEDLINE | Biomedical literature | 36M+ articles | Free API |
| ChEMBL | Drug bioactivity | 2.4M compounds, 20M measurements | Open |
| DrugBank | Drug information | 14,000+ drugs | Academic license |
| BindingDB | Binding data | 2.9M data points | Open |
| Reactome | Pathway database | 2,600+ pathways | Open |
| InnateDB | Innate immunity interactions | 32,000+ interactions | Open |
| KEGG | Pathway maps | 500+ pathways | Open |
| ClinicalTrials.gov | Clinical trials | 500K+ studies | Public |
| FDA FAERS | Adverse events | 20M+ reports | Public |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/harshith-vaddiparthy/immunex.git
cd immunex

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your API keys (NCBI, OpenAI/Azure)

# Run the literature scanner
python -m src.agents.literature_scanner --query "innate immunity drug repurposing" --max-results 100

# Build the knowledge graph
python -m src.knowledge_graph.builder --sources drugbank,chembl,reactome,innatedb

# Run the full pipeline
python -m src.pipeline --target-pathway "TLR signaling" --output results/
```

## Project Structure

```
immunex/
├── src/
│   ├── agents/
│   │   ├── literature_scanner.py    # PubMed/bioRxiv mining
│   │   ├── molecular_reasoner.py    # GNN drug-target prediction
│   │   ├── kg_builder.py            # Knowledge graph construction
│   │   ├── hypothesis_generator.py  # LLM reasoning + ranking
│   │   └── safety_validator.py      # FAERS + interaction checking
│   ├── knowledge_graph/
│   │   ├── builder.py               # Graph construction
│   │   ├── embeddings.py            # TransE/RotatE training
│   │   └── query.py                 # Graph querying + link prediction
│   ├── models/
│   │   ├── gnn.py                   # Graph neural network
│   │   └── embeddings.py            # KG embedding models
│   ├── api/
│   │   ├── server.py                # REST API
│   │   └── dashboard.py             # Interactive web dashboard
│   └── utils/
│       ├── pubmed.py                # PubMed API client
│       ├── chembl.py                # ChEMBL data loader
│       ├── drugbank.py              # DrugBank parser
│       └── faers.py                 # FAERS data loader
├── data/                            # Raw and processed data
├── docs/                            # Documentation
├── tests/                           # Test suite
├── scripts/                         # Utility scripts
├── requirements.txt
├── .env.example
└── pyproject.toml
```

## Innate Immune Targets

IMMUNEX focuses on key innate immune pathways and their druggable targets:

| Pathway | Key Targets | Relevance |
|---------|------------|-----------|
| TLR Signaling | TLR2, TLR4, TLR7, TLR9 | Pathogen recognition, inflammation |
| Inflammasome | NLRP3, NLRC4, AIM2 | IL-1β/IL-18 processing, pyroptosis |
| cGAS-STING | cGAS, STING, TBK1 | DNA sensing, type I IFN response |
| NF-κB | IKKβ, RelA, NEMO | Master inflammatory regulator |
| JAK-STAT | JAK1, JAK2, TYK2, STAT1 | Cytokine signaling |
| Trained Immunity | mTOR, HIF-1α, KDM5 | Epigenetic reprogramming of monocytes |
| Complement | C3, C5, Factor D | Innate immune opsonization |

## Alignment

This project aligns with [ARIA's Sculpting Innate Immunity](https://aria.org.uk/opportunity-spaces/sculpting-innate-immunity/) opportunity space and their Sustained Viral Resilience programme. By identifying existing drugs that modulate innate immune pathways, IMMUNEX accelerates the development of broad-spectrum therapeutic approaches.

Built as part of the [Encode: AI for Science Fellowship](https://encode.pillar.vc/) (Pillar VC x ARIA).

## References

1. Pushpakom, S. et al. "Drug repurposing: progress, challenges and recommendations." *Nature Reviews Drug Discovery* 18, 41-58 (2019)
2. Stokes, J.M. et al. "A Deep Learning Approach to Antibiotic Discovery." *Cell* 180, 688-702 (2020)
3. Netea, M.G. et al. "Trained immunity: a program of innate immune memory in health and disease." *Science* 352, aaf1098 (2016)

## License

MIT

## Author

**Harshith Vaddiparthy** - AI Systems Architect
- [LinkedIn](https://linkedin.com/in/harshith-vaddiparthy)
- [Twitter/X](https://x.com/harshith)
- Forbes Technology Council Member
