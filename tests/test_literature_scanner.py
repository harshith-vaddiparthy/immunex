"""
Tests for the Literature Scanner agent.
"""

import asyncio
import pytest


def test_innate_immune_pathways_defined():
    """Verify all expected innate immune pathways are defined."""
    from src.agents.literature_scanner import INNATE_IMMUNE_PATHWAYS

    expected = [
        "tlr_signaling", "inflammasome", "cgas_sting",
        "nfkb", "jak_stat", "trained_immunity", "complement",
    ]
    for pathway in expected:
        assert pathway in INNATE_IMMUNE_PATHWAYS
        assert "name" in INNATE_IMMUNE_PATHWAYS[pathway]
        assert "targets" in INNATE_IMMUNE_PATHWAYS[pathway]
        assert "search_terms" in INNATE_IMMUNE_PATHWAYS[pathway]
        assert len(INNATE_IMMUNE_PATHWAYS[pathway]["search_terms"]) > 0


def test_build_repurposing_query():
    """Test PubMed query construction."""
    from src.agents.literature_scanner import LiteratureScanner

    scanner = LiteratureScanner()
    query = scanner._build_repurposing_query("inflammasome")

    assert "NLRP3" in query or "inflammasome" in query
    assert "drug repurposing" in query or "drug repositioning" in query
    assert "innate immun" in query


@pytest.mark.asyncio
async def test_pubmed_search_returns_results():
    """Integration test: verify PubMed search works."""
    scanner = LiteratureScanner()
    async with scanner:
        pmids = await scanner.search_pubmed(
            '"NLRP3 inflammasome" AND "drug repurposing"',
            max_results=5,
        )
    assert isinstance(pmids, list)
    # PubMed should return some results for this well-studied topic
    assert len(pmids) > 0


@pytest.mark.asyncio
async def test_fetch_abstracts():
    """Integration test: verify abstract fetching works."""
    scanner = LiteratureScanner()
    async with scanner:
        # Use a known PMID (Pushpakom et al. drug repurposing review)
        articles = await scanner.fetch_abstracts(["30310233"])

    assert len(articles) == 1
    assert articles[0].pmid == "30310233"
    assert "repurposing" in articles[0].title.lower() or "repositioning" in articles[0].title.lower()
    assert len(articles[0].abstract) > 100
