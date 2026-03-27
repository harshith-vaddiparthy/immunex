"""
Molecular Reasoner Agent

Graph neural network for predicting drug-target interactions,
specifically for innate immune targets.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DrugTargetGNN(nn.Module):
    """
    Graph Neural Network for drug-target interaction prediction.

    Uses a bipartite graph structure where:
    - Drug nodes have molecular fingerprint features
    - Target nodes have protein sequence/structure features
    - Edges represent known interactions (with activity values)

    The model predicts binding affinity between drug-target pairs.
    """

    def __init__(
        self,
        drug_feature_dim: int = 1024,  # Morgan fingerprint dimension
        target_feature_dim: int = 512,  # Protein embedding dimension
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.drug_feature_dim = drug_feature_dim
        self.target_feature_dim = target_feature_dim
        self.hidden_dim = hidden_dim

        # Drug encoder
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Target encoder
        self.target_encoder = nn.Sequential(
            nn.Linear(target_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Message passing layers
        self.message_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.message_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

        # Interaction predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode_drugs(self, drug_features: torch.Tensor) -> torch.Tensor:
        """Encode drug molecular fingerprints to embedding space."""
        return self.drug_encoder(drug_features)

    def encode_targets(self, target_features: torch.Tensor) -> torch.Tensor:
        """Encode protein features to embedding space."""
        return self.target_encoder(target_features)

    def predict_interaction(
        self, drug_embedding: torch.Tensor, target_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Predict interaction score between drug and target embeddings."""
        combined = torch.cat([drug_embedding, target_embedding], dim=-1)
        return self.predictor(combined)

    def forward(
        self,
        drug_features: torch.Tensor,
        target_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            drug_features: (N_drugs, drug_feature_dim)
            target_features: (N_targets, target_feature_dim)
            edge_index: (2, N_edges) known interaction edges for message passing

        Returns:
            interaction_scores: (N_drugs, N_targets) predicted binding scores
        """
        drug_emb = self.encode_drugs(drug_features)
        target_emb = self.encode_targets(target_features)

        # Message passing (if edge_index provided)
        if edge_index is not None and edge_index.shape[1] > 0:
            for layer in self.message_layers:
                # Aggregate messages from connected nodes
                src, dst = edge_index
                messages = torch.cat([drug_emb[src], target_emb[dst]], dim=-1)
                updates = layer(messages)

                # Update drug embeddings with aggregated target messages
                drug_updates = torch.zeros_like(drug_emb)
                drug_updates.index_add_(0, src, updates)
                counts = torch.zeros(drug_emb.shape[0], 1, device=drug_emb.device)
                counts.index_add_(0, src, torch.ones_like(updates[:, :1]))
                counts = counts.clamp(min=1)
                drug_emb = drug_emb + drug_updates / counts

        # Predict all pairwise interactions
        # drug_emb: (N_drugs, hidden_dim), target_emb: (N_targets, hidden_dim)
        n_drugs = drug_emb.shape[0]
        n_targets = target_emb.shape[0]

        drug_expanded = drug_emb.unsqueeze(1).expand(-1, n_targets, -1)
        target_expanded = target_emb.unsqueeze(0).expand(n_drugs, -1, -1)

        combined = torch.cat([drug_expanded, target_expanded], dim=-1)
        scores = self.predictor(combined).squeeze(-1)

        return scores


class MolecularReasoner:
    """
    Agent 2: Molecular Reasoner

    Uses a trained GNN to predict drug-target interactions for innate immune targets.
    Also performs off-target analysis to find drugs with unexpected innate immune effects.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = DrugTargetGNN().to(self.device)

        if model_path:
            self.load_model(model_path)

        self.model.eval()

    def load_model(self, path: str) -> None:
        """Load a trained model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model from {path}")

    def predict_binding(
        self,
        drug_fingerprint: np.ndarray,
        target_features: np.ndarray,
    ) -> float:
        """
        Predict binding affinity between a single drug and target.

        Args:
            drug_fingerprint: Morgan fingerprint (1024-dim)
            target_features: Protein embedding (512-dim)

        Returns:
            Predicted binding score (higher = stronger interaction)
        """
        with torch.no_grad():
            drug_tensor = torch.FloatTensor(drug_fingerprint).unsqueeze(0).to(self.device)
            target_tensor = torch.FloatTensor(target_features).unsqueeze(0).to(self.device)

            drug_emb = self.model.encode_drugs(drug_tensor)
            target_emb = self.model.encode_targets(target_tensor)
            score = self.model.predict_interaction(drug_emb, target_emb)

            return score.item()

    def screen_drug_against_immune_targets(
        self,
        drug_fingerprint: np.ndarray,
        target_features_dict: dict[str, np.ndarray],
    ) -> list[dict]:
        """
        Screen a single drug against all innate immune targets.

        Returns sorted list of (target, score) pairs.
        """
        results = []
        for target_name, target_feat in target_features_dict.items():
            score = self.predict_binding(drug_fingerprint, target_feat)
            results.append({
                "target": target_name,
                "predicted_score": round(score, 4),
            })

        results.sort(key=lambda x: x["predicted_score"], reverse=True)
        return results

    def batch_screen(
        self,
        drug_fingerprints: np.ndarray,
        target_features: np.ndarray,
        drug_names: list[str],
        target_names: list[str],
    ) -> list[dict]:
        """
        Screen multiple drugs against multiple targets in batch.

        Returns ranked list of all drug-target pairs by predicted binding.
        """
        with torch.no_grad():
            drug_tensor = torch.FloatTensor(drug_fingerprints).to(self.device)
            target_tensor = torch.FloatTensor(target_features).to(self.device)

            scores = self.model(drug_tensor, target_tensor)
            scores_np = scores.cpu().numpy()

        results = []
        for i, drug_name in enumerate(drug_names):
            for j, target_name in enumerate(target_names):
                results.append({
                    "drug": drug_name,
                    "target": target_name,
                    "predicted_score": round(float(scores_np[i, j]), 4),
                })

        results.sort(key=lambda x: x["predicted_score"], reverse=True)
        return results
