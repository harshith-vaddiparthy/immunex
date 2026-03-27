"""
ChEMBL data loader utilities.

Fetches drug-target interaction data from ChEMBL for innate immune targets.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def fetch_target_activities(
    target_chembl_id: str,
    activity_type: str = "IC50",
    max_results: int = 1000,
) -> list[dict]:
    """
    Fetch bioactivity data for a target from ChEMBL.

    Uses the chembl_webresource_client library.
    """
    try:
        from chembl_webresource_client.new_client import new_client

        activity = new_client.activity
        results = (
            activity.filter(
                target_chembl_id=target_chembl_id,
                standard_type=activity_type,
            )
            .only(
                "molecule_chembl_id",
                "molecule_pref_name",
                "target_chembl_id",
                "target_pref_name",
                "standard_type",
                "standard_value",
                "standard_units",
            )
            [:max_results]
        )

        return [dict(r) for r in results]

    except ImportError:
        logger.warning("chembl_webresource_client not installed")
        return []
    except Exception as e:
        logger.error(f"ChEMBL query failed: {e}")
        return []


# ChEMBL IDs for key innate immune targets
INNATE_IMMUNE_CHEMBL_TARGETS = {
    "TLR4": "CHEMBL5863",
    "TLR7": "CHEMBL6164",
    "TLR9": "CHEMBL6160",
    "NLRP3": "CHEMBL6135",
    "STING1": "CHEMBL4523577",
    "CASP1": "CHEMBL3563",
    "JAK1": "CHEMBL2835",
    "JAK2": "CHEMBL2971",
    "TYK2": "CHEMBL3952",
    "IKBKB": "CHEMBL3476",
    "MTOR": "CHEMBL2842",
    "C5": "CHEMBL2176836",
}


def fetch_all_innate_immune_activities(max_per_target: int = 500) -> dict[str, list[dict]]:
    """Fetch activities for all curated innate immune targets."""
    all_data = {}
    for gene, chembl_id in INNATE_IMMUNE_CHEMBL_TARGETS.items():
        logger.info(f"Fetching ChEMBL data for {gene} ({chembl_id})...")
        data = fetch_target_activities(chembl_id, max_results=max_per_target)
        all_data[gene] = data
        logger.info(f"  Found {len(data)} activity records")
    return all_data
