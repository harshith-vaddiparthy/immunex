"""
IMMUNEX Pipeline

Orchestrates all five agents to produce ranked drug repurposing candidates
for innate immune modulation.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.agents.literature_scanner import LiteratureScanner, INNATE_IMMUNE_PATHWAYS
from src.agents.hypothesis_generator import HypothesisGenerator
from src.agents.safety_validator import SafetyValidator
from src.knowledge_graph.builder import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)


class ImmunexPipeline:
    """
    Main IMMUNEX orchestration pipeline.

    Coordinates:
    1. Literature Scanner - mines PubMed for drug-innate immunity associations
    2. Molecular Reasoner - predicts drug-target binding (requires trained model)
    3. Knowledge Graph Builder - integrates multi-source data
    4. Hypothesis Generator - synthesizes evidence into ranked hypotheses
    5. Safety Validator - checks candidates against FAERS
    """

    def __init__(
        self,
        ncbi_api_key: Optional[str] = None,
        ncbi_email: Optional[str] = None,
        llm_client=None,
        output_dir: str = "results",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.literature_scanner = LiteratureScanner(
            ncbi_api_key=ncbi_api_key,
            ncbi_email=ncbi_email,
            llm_client=llm_client,
        )
        self.kg_builder = KnowledgeGraphBuilder()
        self.hypothesis_generator = HypothesisGenerator(llm_client=llm_client)
        self.safety_validator = SafetyValidator()

    async def run(
        self,
        pathways: Optional[list[str]] = None,
        max_articles_per_pathway: int = 50,
        min_date: str = "2020/01/01",
        extract_associations: bool = True,
        validate_safety: bool = True,
    ) -> dict:
        """
        Run the full IMMUNEX pipeline.

        Args:
            pathways: List of pathway keys to scan (None = all)
            max_articles_per_pathway: Max PubMed results per pathway
            min_date: Earliest publication date to include
            extract_associations: Whether to use LLM for association extraction
            validate_safety: Whether to run safety validation on candidates

        Returns:
            Complete pipeline results with hypotheses and safety reports
        """
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting IMMUNEX pipeline run: {run_id}")
        results = {
            "run_id": run_id,
            "started_at": datetime.utcnow().isoformat(),
            "config": {
                "pathways": pathways or list(INNATE_IMMUNE_PATHWAYS.keys()),
                "max_articles_per_pathway": max_articles_per_pathway,
                "min_date": min_date,
            },
        }

        # Step 1: Literature scanning
        logger.info("Step 1/5: Literature scanning...")
        target_pathways = pathways or list(INNATE_IMMUNE_PATHWAYS.keys())
        all_associations = []

        async with self.literature_scanner:
            for pathway_key in target_pathways:
                scan_result = await self.literature_scanner.scan_pathway(
                    pathway_key,
                    max_results=max_articles_per_pathway,
                    min_date=min_date,
                    extract=extract_associations,
                )
                all_associations.extend(scan_result.get("associations", []))

                # Save per-pathway results
                with open(run_dir / f"literature_{pathway_key}.json", "w") as f:
                    json.dump(scan_result, f, indent=2)

                await asyncio.sleep(1)

        results["literature"] = {
            "total_associations": len(all_associations),
            "pathways_scanned": len(target_pathways),
        }

        # Step 2: Knowledge graph construction
        logger.info("Step 2/5: Building knowledge graph...")
        self.kg_builder.load_innate_immune_targets()
        self.kg_builder.load_literature_associations(all_associations)
        self.kg_builder.export_json(str(run_dir / "knowledge_graph.json"))
        results["knowledge_graph"] = self.kg_builder.stats

        # Step 3: Find candidates from KG
        logger.info("Step 3/5: Finding repurposing candidates...")
        kg_candidates = self.kg_builder.find_repurposing_candidates(min_confidence=0.3)
        results["kg_candidates"] = len(kg_candidates)

        with open(run_dir / "kg_candidates.json", "w") as f:
            json.dump(kg_candidates, f, indent=2)

        # Step 4: Generate hypotheses
        logger.info("Step 4/5: Generating hypotheses...")
        hypotheses = await self.hypothesis_generator.generate_hypotheses(
            literature_associations=all_associations,
            molecular_predictions=[],  # TODO: integrate molecular reasoner
            kg_candidates=kg_candidates,
        )

        hypotheses_data = [h.model_dump() for h in hypotheses]
        with open(run_dir / "hypotheses.json", "w") as f:
            json.dump(hypotheses_data, f, indent=2, default=str)

        results["hypotheses"] = {
            "total": len(hypotheses),
            "by_confidence": {
                "high": sum(1 for h in hypotheses if h.confidence_tier.value == "high"),
                "medium": sum(1 for h in hypotheses if h.confidence_tier.value == "medium"),
                "speculative": sum(1 for h in hypotheses if h.confidence_tier.value == "speculative"),
            },
        }

        # Step 5: Safety validation
        if validate_safety and hypotheses:
            logger.info("Step 5/5: Safety validation...")
            drug_names = list(set(h.drug_name for h in hypotheses))

            async with self.safety_validator:
                safety_reports = await self.safety_validator.validate_batch(drug_names)

            safety_data = [r.model_dump() for r in safety_reports]
            with open(run_dir / "safety_reports.json", "w") as f:
                json.dump(safety_data, f, indent=2)

            results["safety"] = {
                "drugs_validated": len(safety_reports),
                "by_risk": {
                    "low": sum(1 for r in safety_reports if r.overall_risk == "low"),
                    "moderate": sum(1 for r in safety_reports if r.overall_risk == "moderate"),
                    "high": sum(1 for r in safety_reports if r.overall_risk == "high"),
                },
            }

        results["completed_at"] = datetime.utcnow().isoformat()

        # Save full results
        with open(run_dir / "pipeline_results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Pipeline complete. Results saved to {run_dir}")
        return results


async def main():
    """Run IMMUNEX pipeline from command line."""
    import argparse
    from dotenv import load_dotenv
    import os

    load_dotenv()

    parser = argparse.ArgumentParser(description="IMMUNEX Pipeline")
    parser.add_argument("--pathway", type=str, nargs="+",
                        choices=list(INNATE_IMMUNE_PATHWAYS.keys()),
                        help="Pathways to scan (default: all)")
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--min-date", type=str, default="2020/01/01")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip LLM extraction (just fetch articles)")
    parser.add_argument("--no-safety", action="store_true",
                        help="Skip safety validation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize LLM client if keys available
    llm_client = None
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from openai import AsyncOpenAI
        llm_client = AsyncOpenAI(api_key=openai_key)

    pipeline = ImmunexPipeline(
        ncbi_api_key=os.getenv("NCBI_API_KEY"),
        ncbi_email=os.getenv("NCBI_EMAIL"),
        llm_client=llm_client,
        output_dir=args.output,
    )

    results = await pipeline.run(
        pathways=args.pathway,
        max_articles_per_pathway=args.max_results,
        min_date=args.min_date,
        extract_associations=not args.no_extract,
        validate_safety=not args.no_safety,
    )

    print("\n" + "=" * 60)
    print("IMMUNEX Pipeline Results")
    print("=" * 60)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
