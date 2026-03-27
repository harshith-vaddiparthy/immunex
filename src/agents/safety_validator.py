"""
Safety Validator Agent

Cross-references drug repurposing candidates against FDA FAERS adverse event
database, known drug interactions, and contraindications.
"""

import logging
from typing import Optional

import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SafetyReport(BaseModel):
    """Safety assessment for a drug repurposing candidate."""

    drug_name: str
    overall_risk: str  # "low", "moderate", "high"

    # FAERS data
    total_adverse_events: int = 0
    serious_events: int = 0
    top_adverse_events: list[dict] = []

    # Drug interactions
    known_interactions: list[dict] = []

    # Contraindications
    contraindications: list[str] = []

    # Safety flags
    flags: list[str] = []
    recommendation: str = ""


class SafetyValidator:
    """
    Agent 5: Safety Validator

    Validates drug repurposing candidates against safety databases:
    - FDA FAERS (adverse event reports)
    - openFDA API for drug labeling and interactions
    """

    OPENFDA_BASE = "https://api.fda.gov"

    def __init__(self, openfda_api_key: Optional[str] = None):
        self.api_key = openfda_api_key
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def query_faers(self, drug_name: str, limit: int = 10) -> dict:
        """
        Query FDA FAERS for adverse event reports for a drug.

        Uses the openFDA drug/event endpoint.
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.OPENFDA_BASE}/drug/event.json"
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": limit,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "total_results": data.get("meta", {}).get("results", {}).get("total", 0),
                        "top_reactions": data.get("results", []),
                    }
                elif resp.status == 404:
                    return {"total_results": 0, "top_reactions": []}
                else:
                    logger.warning(f"FAERS query failed with status {resp.status}")
                    return {"total_results": 0, "top_reactions": [], "error": f"HTTP {resp.status}"}
        except Exception as e:
            logger.error(f"FAERS query error: {e}")
            return {"total_results": 0, "top_reactions": [], "error": str(e)}

    async def query_drug_interactions(self, drug_name: str) -> list[dict]:
        """Query openFDA for known drug interactions from labeling."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.OPENFDA_BASE}/drug/label.json"
        params = {
            "search": f'openfda.generic_name:"{drug_name}"',
            "limit": 1,
        }

        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("results", [])
                    if results:
                        label = results[0]
                        interactions = label.get("drug_interactions", [])
                        contraindications = label.get("contraindications", [])
                        warnings = label.get("warnings", [])
                        return {
                            "interactions": interactions,
                            "contraindications": contraindications,
                            "warnings": warnings,
                        }
                return {"interactions": [], "contraindications": [], "warnings": []}
        except Exception as e:
            logger.error(f"Drug interaction query error: {e}")
            return {"interactions": [], "contraindications": [], "warnings": []}

    async def validate_candidate(self, drug_name: str) -> SafetyReport:
        """
        Run full safety validation on a drug repurposing candidate.
        """
        logger.info(f"Validating safety for: {drug_name}")

        # Query FAERS
        faers_data = await self.query_faers(drug_name)
        total_events = faers_data.get("total_results", 0)
        top_reactions = faers_data.get("top_reactions", [])

        # Query drug interactions/labeling
        label_data = await self.query_drug_interactions(drug_name)

        # Analyze risk level
        flags = []
        risk = "low"

        # High adverse event count
        if total_events > 10000:
            flags.append(f"High FAERS report volume ({total_events:,} reports)")
            risk = "moderate"

        if total_events > 100000:
            risk = "high"

        # Check for serious reactions in top events
        serious_keywords = [
            "death", "cardiac", "hepatic", "renal failure", "anaphylaxis",
            "stevens-johnson", "toxic epidermal", "agranulocytosis",
        ]
        for reaction in top_reactions:
            term = reaction.get("term", "").lower()
            if any(kw in term for kw in serious_keywords):
                flags.append(f"Serious adverse event reported: {reaction.get('term', '')}")
                if risk == "low":
                    risk = "moderate"

        # Generate recommendation
        if risk == "low":
            recommendation = "Candidate has an acceptable safety profile for repurposing evaluation. Proceed with standard preclinical validation."
        elif risk == "moderate":
            recommendation = "Candidate has moderate safety signals. Careful dose-response evaluation and enhanced monitoring recommended."
        else:
            recommendation = "Candidate has significant safety concerns. Repurposing should only proceed if therapeutic benefit clearly outweighs risks."

        return SafetyReport(
            drug_name=drug_name,
            overall_risk=risk,
            total_adverse_events=total_events,
            top_adverse_events=[
                {"reaction": r.get("term", ""), "count": r.get("count", 0)}
                for r in top_reactions[:10]
            ],
            contraindications=label_data.get("contraindications", [])[:5],
            flags=flags,
            recommendation=recommendation,
        )

    async def validate_batch(self, drug_names: list[str]) -> list[SafetyReport]:
        """Validate multiple candidates."""
        reports = []
        for name in drug_names:
            report = await self.validate_candidate(name)
            reports.append(report)
        return reports
