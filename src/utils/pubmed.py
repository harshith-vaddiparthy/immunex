"""
PubMed API client utilities.
"""

from typing import Optional

# NCBI E-utilities base URLs
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"
ESUMMARY_URL = f"{EUTILS_BASE}/esummary.fcgi"


def build_drug_repurposing_query(
    drug_name: Optional[str] = None,
    pathway: Optional[str] = None,
    target: Optional[str] = None,
    additional_terms: Optional[list[str]] = None,
) -> str:
    """Build a PubMed search query for drug repurposing studies."""
    parts = []

    if drug_name:
        parts.append(f'"{drug_name}"[Title/Abstract]')

    if pathway:
        parts.append(f'"{pathway}"[Title/Abstract]')

    if target:
        parts.append(f'"{target}"[Title/Abstract]')

    if additional_terms:
        for term in additional_terms:
            parts.append(f'"{term}"[Title/Abstract]')

    # Always include repurposing context
    repurposing = '("drug repurposing" OR "drug repositioning" OR "off-target" OR "repurposed")'
    innate = '("innate immunity" OR "innate immune" OR "inflammasome" OR "toll-like receptor")'

    query_parts = parts + [repurposing, innate]
    return " AND ".join(query_parts)
