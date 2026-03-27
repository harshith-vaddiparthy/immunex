"""
Knowledge Graph Builder

Constructs a heterogeneous biomedical knowledge graph linking drugs, targets,
pathways, diseases, and clinical outcomes from multiple data sources.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import networkx as nx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    DRUG = "drug"
    TARGET = "target"  # protein/gene
    PATHWAY = "pathway"
    DISEASE = "disease"
    ADVERSE_EVENT = "adverse_event"
    CLINICAL_TRIAL = "clinical_trial"


class EdgeType(str, Enum):
    TARGETS = "targets"  # drug -> target
    INHIBITS = "inhibits"  # drug -> target
    ACTIVATES = "activates"  # drug -> target
    MODULATES = "modulates"  # drug -> pathway
    PARTICIPATES_IN = "participates_in"  # target -> pathway
    TREATS = "treats"  # drug -> disease
    INDICATED_FOR = "indicated_for"  # drug -> disease
    ASSOCIATED_WITH = "associated_with"  # drug -> adverse_event
    TESTED_IN = "tested_in"  # drug -> clinical_trial
    IMPLICATED_IN = "implicated_in"  # target -> disease
    PREDICTED = "predicted"  # link prediction output


class KGNode(BaseModel):
    id: str
    name: str
    node_type: NodeType
    properties: dict = {}
    sources: list[str] = []


class KGEdge(BaseModel):
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    confidence: float = 1.0
    properties: dict = {}
    sources: list[str] = []


# Known innate immune targets for IMMUNEX focus
INNATE_IMMUNE_TARGETS = {
    # TLR signaling
    "TLR2": {"name": "Toll-like receptor 2", "uniprot": "O60603", "pathway": "tlr_signaling"},
    "TLR4": {"name": "Toll-like receptor 4", "uniprot": "O00206", "pathway": "tlr_signaling"},
    "TLR7": {"name": "Toll-like receptor 7", "uniprot": "Q9NYK1", "pathway": "tlr_signaling"},
    "TLR9": {"name": "Toll-like receptor 9", "uniprot": "Q9NR96", "pathway": "tlr_signaling"},
    "MYD88": {"name": "Myeloid differentiation factor 88", "uniprot": "Q99836", "pathway": "tlr_signaling"},
    # Inflammasome
    "NLRP3": {"name": "NACHT, LRR and PYD domains-containing protein 3", "uniprot": "Q96P20", "pathway": "inflammasome"},
    "NLRC4": {"name": "NLR family CARD domain-containing protein 4", "uniprot": "Q9NPP4", "pathway": "inflammasome"},
    "CASP1": {"name": "Caspase-1", "uniprot": "P29466", "pathway": "inflammasome"},
    "PYCARD": {"name": "ASC/PYCARD", "uniprot": "Q9ULZ3", "pathway": "inflammasome"},
    # cGAS-STING
    "CGAS": {"name": "Cyclic GMP-AMP synthase", "uniprot": "Q8N884", "pathway": "cgas_sting"},
    "STING1": {"name": "Stimulator of interferon genes", "uniprot": "Q86WV6", "pathway": "cgas_sting"},
    "TBK1": {"name": "TANK-binding kinase 1", "uniprot": "Q9UHD2", "pathway": "cgas_sting"},
    # NF-kB
    "NFKB1": {"name": "Nuclear factor NF-kappa-B p105", "uniprot": "P19838", "pathway": "nfkb"},
    "RELA": {"name": "Transcription factor p65/RelA", "uniprot": "Q04206", "pathway": "nfkb"},
    "IKBKB": {"name": "IKK-beta", "uniprot": "O14920", "pathway": "nfkb"},
    # JAK-STAT
    "JAK1": {"name": "Tyrosine-protein kinase JAK1", "uniprot": "P23458", "pathway": "jak_stat"},
    "JAK2": {"name": "Tyrosine-protein kinase JAK2", "uniprot": "O60674", "pathway": "jak_stat"},
    "TYK2": {"name": "Non-receptor tyrosine-protein kinase TYK2", "uniprot": "P29597", "pathway": "jak_stat"},
    "STAT1": {"name": "Signal transducer and activator of transcription 1", "uniprot": "P42224", "pathway": "jak_stat"},
    # Trained immunity
    "MTOR": {"name": "Serine/threonine-protein kinase mTOR", "uniprot": "P42345", "pathway": "trained_immunity"},
    "HIF1A": {"name": "Hypoxia-inducible factor 1-alpha", "uniprot": "Q16665", "pathway": "trained_immunity"},
    # Complement
    "C3": {"name": "Complement C3", "uniprot": "P01024", "pathway": "complement"},
    "C5": {"name": "Complement C5", "uniprot": "P01031", "pathway": "complement"},
}


class KnowledgeGraphBuilder:
    """
    Agent 3: Knowledge Graph Builder

    Constructs a heterogeneous biomedical knowledge graph from multiple sources:
    - DrugBank (drug-target interactions)
    - ChEMBL (bioactivity data)
    - Reactome/KEGG (pathways)
    - InnateDB (innate immunity interactions)
    - Literature Scanner output (drug-pathway associations)
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes: dict[str, KGNode] = {}
        self.edges: list[KGEdge] = []
        self._stats = {"nodes": 0, "edges": 0, "sources_loaded": []}

    @property
    def stats(self) -> dict:
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": self._count_node_types(),
            "edge_types": self._count_edge_types(),
            "sources_loaded": self._stats["sources_loaded"],
        }

    def _count_node_types(self) -> dict[str, int]:
        counts = {}
        for _, data in self.graph.nodes(data=True):
            nt = data.get("node_type", "unknown")
            counts[nt] = counts.get(nt, 0) + 1
        return counts

    def _count_edge_types(self) -> dict[str, int]:
        counts = {}
        for _, _, data in self.graph.edges(data=True):
            et = data.get("edge_type", "unknown")
            counts[et] = counts.get(et, 0) + 1
        return counts

    def add_node(self, node: KGNode) -> None:
        """Add a node to the knowledge graph."""
        self.graph.add_node(
            node.id,
            name=node.name,
            node_type=node.node_type.value,
            **node.properties,
        )
        self.nodes[node.id] = node

    def add_edge(self, edge: KGEdge) -> None:
        """Add an edge to the knowledge graph."""
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type.value,
            weight=edge.weight,
            confidence=edge.confidence,
            **edge.properties,
        )
        self.edges.append(edge)

    def load_innate_immune_targets(self) -> None:
        """Load the curated set of innate immune targets as nodes."""
        logger.info("Loading innate immune targets...")

        # Add pathway nodes
        pathways = set()
        for target_id, info in INNATE_IMMUNE_TARGETS.items():
            pathway_id = f"pathway:{info['pathway']}"
            if pathway_id not in pathways:
                self.add_node(KGNode(
                    id=pathway_id,
                    name=info["pathway"].replace("_", " ").title(),
                    node_type=NodeType.PATHWAY,
                    properties={"pathway_key": info["pathway"]},
                    sources=["immunex_curated"],
                ))
                pathways.add(pathway_id)

            # Add target node
            self.add_node(KGNode(
                id=f"target:{target_id}",
                name=info["name"],
                node_type=NodeType.TARGET,
                properties={"gene_symbol": target_id, "uniprot": info.get("uniprot", "")},
                sources=["immunex_curated"],
            ))

            # Add target -> pathway edge
            self.add_edge(KGEdge(
                source_id=f"target:{target_id}",
                target_id=pathway_id,
                edge_type=EdgeType.PARTICIPATES_IN,
                confidence=1.0,
                sources=["immunex_curated"],
            ))

        logger.info(f"Loaded {len(INNATE_IMMUNE_TARGETS)} targets across {len(pathways)} pathways")
        self._stats["sources_loaded"].append("innate_immune_targets")

    def load_from_chembl(self, chembl_data: list[dict]) -> None:
        """
        Load drug-target interaction data from ChEMBL.

        Expected format: list of dicts with keys:
        - molecule_chembl_id, pref_name (drug)
        - target_chembl_id, target_pref_name (target)
        - standard_type, standard_value, standard_units (activity)
        """
        logger.info(f"Loading {len(chembl_data)} ChEMBL interactions...")
        drugs_added = set()
        edges_added = 0

        for record in chembl_data:
            drug_id = f"drug:{record.get('molecule_chembl_id', '')}"
            drug_name = record.get("pref_name", "")
            target_gene = record.get("target_gene_symbol", "")

            if not drug_name or not target_gene:
                continue

            # Add drug node
            if drug_id not in drugs_added:
                self.add_node(KGNode(
                    id=drug_id,
                    name=drug_name,
                    node_type=NodeType.DRUG,
                    properties={
                        "chembl_id": record.get("molecule_chembl_id", ""),
                        "max_phase": record.get("max_phase", 0),
                    },
                    sources=["chembl"],
                ))
                drugs_added.add(drug_id)

            # Add edge if target is an innate immune target
            target_node_id = f"target:{target_gene}"
            if target_gene in INNATE_IMMUNE_TARGETS:
                activity_type = record.get("standard_type", "")
                activity_value = record.get("standard_value")
                self.add_edge(KGEdge(
                    source_id=drug_id,
                    target_id=target_node_id,
                    edge_type=EdgeType.TARGETS,
                    confidence=0.8,
                    properties={
                        "activity_type": activity_type,
                        "activity_value": activity_value,
                        "activity_units": record.get("standard_units", ""),
                    },
                    sources=["chembl"],
                ))
                edges_added += 1

        logger.info(f"Added {len(drugs_added)} drugs, {edges_added} edges from ChEMBL")
        self._stats["sources_loaded"].append("chembl")

    def load_literature_associations(self, associations: list[dict]) -> None:
        """Load drug-pathway associations extracted by the Literature Scanner."""
        logger.info(f"Loading {len(associations)} literature associations...")

        for assoc in associations:
            drug_name = assoc.get("drug_name", "")
            pathway = assoc.get("pathway", "")
            effect = assoc.get("effect", "unknown")

            if not drug_name or not pathway:
                continue

            drug_id = f"drug:{drug_name.lower().replace(' ', '_')}"
            pathway_id = f"pathway:{pathway}"

            # Add drug node if not exists
            if drug_id not in self.nodes:
                self.add_node(KGNode(
                    id=drug_id,
                    name=drug_name,
                    node_type=NodeType.DRUG,
                    sources=["literature"],
                ))

            # Map effect to edge type
            edge_type_map = {
                "inhibitor": EdgeType.INHIBITS,
                "activator": EdgeType.ACTIVATES,
                "modulator": EdgeType.MODULATES,
                "unknown": EdgeType.MODULATES,
            }
            edge_type = edge_type_map.get(effect, EdgeType.MODULATES)

            self.add_edge(KGEdge(
                source_id=drug_id,
                target_id=pathway_id,
                edge_type=edge_type,
                confidence=assoc.get("confidence", 0.5),
                properties={
                    "source_pmid": assoc.get("source_pmid", ""),
                    "target_protein": assoc.get("target_protein", ""),
                },
                sources=["literature"],
            ))

        self._stats["sources_loaded"].append("literature")

    def find_repurposing_candidates(
        self,
        target_pathway: Optional[str] = None,
        min_confidence: float = 0.5,
    ) -> list[dict]:
        """
        Find drugs that interact with innate immune targets/pathways.
        Returns ranked candidates with evidence summaries.
        """
        candidates = {}

        for drug_id, data in self.graph.nodes(data=True):
            if data.get("node_type") != "drug":
                continue

            # Get all edges from this drug to immune targets/pathways
            drug_edges = []
            for _, target, edge_data in self.graph.out_edges(drug_id, data=True):
                target_data = self.graph.nodes.get(target, {})
                target_type = target_data.get("node_type", "")

                if target_type in ("target", "pathway"):
                    if target_pathway:
                        # Check if target belongs to specified pathway
                        if target_type == "pathway" and target != f"pathway:{target_pathway}":
                            continue
                        if target_type == "target":
                            gene = target_data.get("gene_symbol", "")
                            if gene in INNATE_IMMUNE_TARGETS:
                                if INNATE_IMMUNE_TARGETS[gene]["pathway"] != target_pathway:
                                    continue

                    if edge_data.get("confidence", 0) >= min_confidence:
                        drug_edges.append({
                            "target": target,
                            "target_name": target_data.get("name", ""),
                            "edge_type": edge_data.get("edge_type", ""),
                            "confidence": edge_data.get("confidence", 0),
                            "sources": edge_data.get("sources", []),
                        })

            if drug_edges:
                # Score: number of connections * average confidence
                avg_conf = sum(e["confidence"] for e in drug_edges) / len(drug_edges)
                score = len(drug_edges) * avg_conf

                candidates[drug_id] = {
                    "drug_id": drug_id,
                    "drug_name": data.get("name", ""),
                    "score": round(score, 3),
                    "num_connections": len(drug_edges),
                    "avg_confidence": round(avg_conf, 3),
                    "interactions": drug_edges,
                }

        # Sort by score
        ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
        return ranked

    def export_json(self, filepath: str) -> None:
        """Export the knowledge graph to JSON."""
        data = {
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "stats": self.stats,
            },
            "nodes": [
                {"id": n, **d} for n, d in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in self.graph.edges(data=True)
            ],
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Exported KG to {filepath}")

    def export_graphml(self, filepath: str) -> None:
        """Export to GraphML format for visualization tools."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self.graph, filepath)
        logger.info(f"Exported KG to {filepath}")
