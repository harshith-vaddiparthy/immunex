"""
FDA FAERS data loader utilities.

Provides access to the FDA Adverse Event Reporting System via the openFDA API.
"""

import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

OPENFDA_BASE = "https://api.fda.gov"


async def search_drug_adverse_events(
    drug_name: str,
    limit: int = 10,
    api_key: Optional[str] = None,
) -> dict:
    """
    Search FDA FAERS for adverse events associated with a drug.

    Returns top adverse reactions by frequency.
    """
    url = f"{OPENFDA_BASE}/drug/event.json"
    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_name}"',
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": limit,
    }
    if api_key:
        params["api_key"] = api_key

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {
                    "drug": drug_name,
                    "total_reports": data.get("meta", {}).get("results", {}).get("total", 0),
                    "top_reactions": [
                        {"reaction": r["term"], "count": r["count"]}
                        for r in data.get("results", [])
                    ],
                }
            elif resp.status == 404:
                return {"drug": drug_name, "total_reports": 0, "top_reactions": []}
            else:
                return {"drug": drug_name, "error": f"HTTP {resp.status}"}


async def get_drug_label_info(
    drug_name: str,
    api_key: Optional[str] = None,
) -> dict:
    """
    Fetch drug labeling information from openFDA.

    Includes indications, contraindications, warnings, interactions.
    """
    url = f"{OPENFDA_BASE}/drug/label.json"
    params = {
        "search": f'openfda.generic_name:"{drug_name}"',
        "limit": 1,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                results = data.get("results", [])
                if results:
                    label = results[0]
                    return {
                        "drug": drug_name,
                        "brand_name": label.get("openfda", {}).get("brand_name", []),
                        "generic_name": label.get("openfda", {}).get("generic_name", []),
                        "indications": label.get("indications_and_usage", []),
                        "contraindications": label.get("contraindications", []),
                        "warnings": label.get("warnings", []),
                        "drug_interactions": label.get("drug_interactions", []),
                        "mechanism_of_action": label.get("mechanism_of_action", []),
                    }
            return {"drug": drug_name, "error": "Not found"}
