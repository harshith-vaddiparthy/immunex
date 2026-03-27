"""
Hypothesis Generator Agent

LLM-based reasoning agent that synthesizes outputs from the Literature Scanner,
Molecular Reasoner, and Knowledge Graph to generate ranked drug repurposing hypotheses.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ConfidenceTier(str, Enum):
    HIGH = "high"  # Multiple independent evidence sources
    MEDIUM = "medium"  # Single strong source or multiple weak sources
    SPECULATIVE = "speculative"  # Computational prediction only


class RepurposingHypothesis(BaseModel):
    """A complete drug repurposing hypothesis with evidence chain."""

    drug_name: str
    drug_id: Optional[str] = None

    # Target pathway/mechanism
    pathway: str
    target_protein: Optional[str] = None
    proposed_mechanism: str

    # Disease/condition
    target_condition: str

    # Effect
    predicted_effect: str  # "activator", "inhibitor", "modulator"

    # Evidence
    confidence_tier: ConfidenceTier
    confidence_score: float  # 0-1
    evidence_chain: list[dict] = []

    # Suggested validation
    suggested_experiments: list[str] = []

    # Safety
    safety_flags: list[str] = []
    known_contraindications: list[str] = []

    # Metadata
    generated_at: str = ""
    sources: list[str] = []


HYPOTHESIS_SYSTEM_PROMPT = """You are IMMUNEX Hypothesis Generator, a biomedical AI reasoning system.

Your role is to synthesize evidence from multiple sources to generate drug repurposing hypotheses for innate immune modulation.

You receive:
1. Literature evidence: drug-pathway associations extracted from PubMed
2. Molecular predictions: GNN-predicted binding scores for drug-target pairs
3. Knowledge graph context: network relationships between drugs, targets, and pathways

For each candidate, you must:
- Propose a specific mechanistic hypothesis (HOW the drug modulates innate immunity)
- Assess confidence based on convergence of evidence sources
- Identify the target condition(s) where this modulation would be therapeutic
- Flag any safety concerns
- Suggest validation experiments

Be scientifically rigorous. Clearly distinguish between:
- Established facts (from literature with PMID citations)
- Strong predictions (multiple evidence types converging)
- Speculative hypotheses (single weak signal)

Never overstate confidence. A speculative hypothesis clearly labeled as such is more valuable than a false HIGH confidence claim."""


HYPOTHESIS_GENERATION_PROMPT = """Based on the following evidence, generate drug repurposing hypotheses for innate immune modulation.

## Literature Evidence
{literature_evidence}

## Molecular Predictions (GNN binding scores)
{molecular_evidence}

## Knowledge Graph Context
{kg_context}

For each viable candidate, provide a JSON object with:
{{
  "drug_name": "...",
  "pathway": "which innate immune pathway",
  "target_protein": "specific target if known",
  "proposed_mechanism": "detailed mechanistic hypothesis",
  "target_condition": "disease/condition this would treat",
  "predicted_effect": "activator|inhibitor|modulator",
  "confidence_tier": "high|medium|speculative",
  "confidence_score": 0.0-1.0,
  "evidence_chain": [
    {{"source": "literature|molecular|kg", "detail": "what evidence says", "strength": "strong|moderate|weak"}}
  ],
  "suggested_experiments": ["experiment 1", "experiment 2"],
  "safety_flags": ["any concerns"]
}}

Return a JSON object with key "hypotheses" containing an array of hypothesis objects.
Only include candidates where you have genuine reason to believe the drug could modulate innate immunity.
Quality over quantity - 3 well-supported hypotheses are better than 10 speculative ones."""


class HypothesisGenerator:
    """
    Agent 4: Hypothesis Generator

    Synthesizes evidence from all other agents to produce ranked
    drug repurposing hypotheses with full mechanistic explanations.
    """

    def __init__(self, llm_client=None, model: str = "gpt-4o"):
        self.llm_client = llm_client
        self.model = model

    def _format_literature_evidence(self, associations: list[dict]) -> str:
        """Format literature scanner output for the prompt."""
        if not associations:
            return "No literature evidence available."

        lines = []
        for a in associations[:20]:  # Limit to top 20
            lines.append(
                f"- {a.get('drug_name', '?')} -> {a.get('pathway', '?')} "
                f"({a.get('effect', '?')}, confidence: {a.get('confidence', 0):.2f}) "
                f"[PMID: {a.get('source_pmid', 'N/A')}]"
            )
        return "\n".join(lines)

    def _format_molecular_evidence(self, predictions: list[dict]) -> str:
        """Format molecular reasoner output for the prompt."""
        if not predictions:
            return "No molecular predictions available."

        lines = []
        for p in predictions[:20]:
            lines.append(
                f"- {p.get('drug', '?')} -> {p.get('target', '?')} "
                f"(predicted binding score: {p.get('predicted_score', 0):.4f})"
            )
        return "\n".join(lines)

    def _format_kg_context(self, candidates: list[dict]) -> str:
        """Format knowledge graph candidates for the prompt."""
        if not candidates:
            return "No knowledge graph context available."

        lines = []
        for c in candidates[:15]:
            interactions = ", ".join(
                f"{i['target_name']} ({i['edge_type']})"
                for i in c.get("interactions", [])[:5]
            )
            lines.append(
                f"- {c.get('drug_name', '?')} (score: {c.get('score', 0):.3f}, "
                f"{c.get('num_connections', 0)} connections): {interactions}"
            )
        return "\n".join(lines)

    async def generate_hypotheses(
        self,
        literature_associations: list[dict],
        molecular_predictions: list[dict],
        kg_candidates: list[dict],
    ) -> list[RepurposingHypothesis]:
        """
        Generate ranked repurposing hypotheses from multi-source evidence.
        """
        if not self.llm_client:
            logger.error("No LLM client configured")
            return []

        prompt = HYPOTHESIS_GENERATION_PROMPT.format(
            literature_evidence=self._format_literature_evidence(literature_associations),
            molecular_evidence=self._format_molecular_evidence(molecular_predictions),
            kg_context=self._format_kg_context(kg_candidates),
        )

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": HYPOTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=4000,
            )

            content = response.choices[0].message.content
            data = json.loads(content)
            hypotheses_data = data.get("hypotheses", [])

            hypotheses = []
            for h in hypotheses_data:
                hypothesis = RepurposingHypothesis(
                    drug_name=h.get("drug_name", ""),
                    pathway=h.get("pathway", ""),
                    target_protein=h.get("target_protein"),
                    proposed_mechanism=h.get("proposed_mechanism", ""),
                    target_condition=h.get("target_condition", ""),
                    predicted_effect=h.get("predicted_effect", "modulator"),
                    confidence_tier=ConfidenceTier(h.get("confidence_tier", "speculative")),
                    confidence_score=float(h.get("confidence_score", 0.5)),
                    evidence_chain=h.get("evidence_chain", []),
                    suggested_experiments=h.get("suggested_experiments", []),
                    safety_flags=h.get("safety_flags", []),
                    generated_at=datetime.utcnow().isoformat(),
                    sources=["literature", "molecular", "kg"],
                )
                hypotheses.append(hypothesis)

            # Sort by confidence
            hypotheses.sort(key=lambda x: x.confidence_score, reverse=True)
            logger.info(f"Generated {len(hypotheses)} hypotheses")
            return hypotheses

        except Exception as e:
            logger.error(f"Hypothesis generation failed: {e}")
            return []
