"""
Molecular Reasoner Agent

Wraps the GNN model and provides high-level drug screening against
innate immune targets.
"""

import logging
from typing import Optional

import numpy as np

from src.models.gnn import DrugTargetGNN, MolecularReasoner
from src.knowledge_graph.builder import INNATE_IMMUNE_TARGETS

logger = logging.getLogger(__name__)


def get_drug_fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024) -> Optional[np.ndarray]:
    """
    Compute Morgan fingerprint for a drug from its SMILES string.

    Requires rdkit.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)

    except ImportError:
        logger.error("rdkit not installed - cannot compute fingerprints")
        return None


# Known SMILES for common drugs studied in innate immunity context
REFERENCE_DRUGS = {
    "Metformin": "CN(C)C(=N)NC(=N)N",
    "Colchicine": "COc1cc2c(c(OC)c1OC)-c1ccc(OC)c(=O)cc1[C@@H](NC(C)=O)CC2",
    "Hydroxychloroquine": "CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
    "Dexamethasone": "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO",
    "Baricitinib": "CCS(=O)(=O)N1CC(CC#N)(n2cc(-c3nncn3C)cn2)C1",
    "Dimethyl fumarate": "COC(=O)/C=C/C(=O)OC",
    "Anakinra": None,  # Biologic - no SMILES
    "Canakinumab": None,  # Biologic
    "Tofacitinib": "CC1CCN(C(=O)CC#N)CC1N1c2nccnc2NC1=O",
    "MCC950": None,  # NLRP3 inhibitor - research compound
}
