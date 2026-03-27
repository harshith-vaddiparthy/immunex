"""
DrugBank data utilities.

Parsers and helpers for working with DrugBank data.
DrugBank requires an academic license for full XML dataset.
This module works with the open subset and curated reference data.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def parse_drugbank_vocabulary(filepath: str) -> list[dict]:
    """
    Parse DrugBank vocabulary CSV (freely available).

    Downloads from: https://go.drugbank.com/releases/latest#open-data
    Contains DrugBank IDs, names, CAS numbers, and UNII codes.
    """
    drugs = []
    path = Path(filepath)

    if not path.exists():
        logger.warning(f"DrugBank vocabulary file not found: {filepath}")
        return drugs

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            drugs.append({
                "drugbank_id": row.get("DrugBank ID", ""),
                "accession_numbers": row.get("Accession Numbers", ""),
                "common_name": row.get("Common name", ""),
                "cas": row.get("CAS", ""),
                "unii": row.get("UNII", ""),
                "synonyms": row.get("Synonyms", "").split(" | ") if row.get("Synonyms") else [],
                "standard_inchi_key": row.get("Standard InChI Key", ""),
            })

    logger.info(f"Parsed {len(drugs)} drugs from DrugBank vocabulary")
    return drugs


def parse_drugbank_drug_targets(filepath: str) -> list[dict]:
    """
    Parse DrugBank drug-target associations CSV.

    Downloads from: https://go.drugbank.com/releases/latest#open-data
    """
    targets = []
    path = Path(filepath)

    if not path.exists():
        logger.warning(f"DrugBank targets file not found: {filepath}")
        return targets

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets.append({
                "drugbank_id": row.get("Drug IDs", ""),
                "drug_name": row.get("Name", ""),
                "gene_name": row.get("Gene Name", ""),
                "uniprot_id": row.get("UniProt ID", ""),
                "actions": row.get("Actions", "").split("; ") if row.get("Actions") else [],
                "pharmacological_action": row.get("Pharmacological Action", ""),
            })

    logger.info(f"Parsed {len(targets)} drug-target associations")
    return targets


# Mapping of DrugBank drug groups for filtering
DRUG_GROUPS = {
    "approved": "FDA-approved drugs (primary repurposing candidates)",
    "investigational": "Drugs in clinical trials",
    "experimental": "Pre-clinical compounds",
    "withdrawn": "Withdrawn drugs (potential safety concerns)",
    "nutraceutical": "Natural health products",
}
