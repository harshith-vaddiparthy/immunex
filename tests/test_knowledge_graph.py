"""
Tests for the Knowledge Graph Builder.
"""

from src.knowledge_graph.builder import (
    KnowledgeGraphBuilder,
    KGNode,
    KGEdge,
    NodeType,
    EdgeType,
    INNATE_IMMUNE_TARGETS,
)


def test_load_innate_immune_targets():
    """Test loading curated innate immune targets."""
    kg = KnowledgeGraphBuilder()
    kg.load_innate_immune_targets()

    stats = kg.stats
    assert stats["total_nodes"] > 0
    assert stats["node_types"].get("target", 0) == len(INNATE_IMMUNE_TARGETS)
    assert stats["node_types"].get("pathway", 0) == 7  # 7 pathways


def test_add_custom_nodes_and_edges():
    """Test adding custom nodes and edges."""
    kg = KnowledgeGraphBuilder()

    kg.add_node(KGNode(
        id="drug:metformin",
        name="Metformin",
        node_type=NodeType.DRUG,
    ))
    kg.add_node(KGNode(
        id="target:NLRP3",
        name="NLRP3",
        node_type=NodeType.TARGET,
    ))
    kg.add_edge(KGEdge(
        source_id="drug:metformin",
        target_id="target:NLRP3",
        edge_type=EdgeType.INHIBITS,
        confidence=0.85,
    ))

    assert kg.stats["total_nodes"] == 2
    assert kg.stats["total_edges"] == 1


def test_find_repurposing_candidates():
    """Test candidate finding from knowledge graph."""
    kg = KnowledgeGraphBuilder()
    kg.load_innate_immune_targets()

    # Add a drug with connections
    kg.add_node(KGNode(
        id="drug:colchicine",
        name="Colchicine",
        node_type=NodeType.DRUG,
    ))
    kg.add_edge(KGEdge(
        source_id="drug:colchicine",
        target_id="pathway:inflammasome",
        edge_type=EdgeType.INHIBITS,
        confidence=0.9,
    ))

    candidates = kg.find_repurposing_candidates()
    assert len(candidates) >= 1
    assert candidates[0]["drug_name"] == "Colchicine"


def test_load_literature_associations():
    """Test loading associations from literature scanner output."""
    kg = KnowledgeGraphBuilder()
    kg.load_innate_immune_targets()

    associations = [
        {
            "drug_name": "Metformin",
            "pathway": "inflammasome",
            "effect": "inhibitor",
            "confidence": 0.8,
            "source_pmid": "12345678",
        },
        {
            "drug_name": "Hydroxychloroquine",
            "pathway": "tlr_signaling",
            "effect": "inhibitor",
            "confidence": 0.75,
            "source_pmid": "87654321",
        },
    ]

    kg.load_literature_associations(associations)
    assert kg.stats["total_nodes"] > len(INNATE_IMMUNE_TARGETS) + 7  # targets + pathways + new drugs
