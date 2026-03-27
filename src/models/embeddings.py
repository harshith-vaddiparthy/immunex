"""
Knowledge Graph embedding models for link prediction.

Implements TransE and RotatE for predicting undocumented
drug-pathway associations in the biomedical knowledge graph.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TransE(nn.Module):
    """
    TransE knowledge graph embedding model.

    Models relationships as translations: h + r ≈ t
    where h is head entity, r is relation, t is tail entity.

    Used for link prediction: given a drug (h) and relation type (r),
    predict which targets (t) it might interact with.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 128,
        margin: float = 1.0,
        norm: int = 2,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.margin = margin
        self.norm = norm

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Initialize
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # Normalize relation embeddings
        with torch.no_grad():
            self.relation_embeddings.weight.data = F.normalize(
                self.relation_embeddings.weight.data, p=2, dim=1
            )

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TransE score: -||h + r - t||"""
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)
        return -torch.norm(h + r - t, p=self.norm, dim=-1)

    def forward(
        self,
        positive_triples: torch.Tensor,
        negative_triples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute margin-based ranking loss.

        Args:
            positive_triples: (batch, 3) - [head, relation, tail] indices
            negative_triples: (batch, 3) - corrupted triples
        """
        pos_scores = self.score(
            positive_triples[:, 0],
            positive_triples[:, 1],
            positive_triples[:, 2],
        )
        neg_scores = self.score(
            negative_triples[:, 0],
            negative_triples[:, 1],
            negative_triples[:, 2],
        )

        # Margin ranking loss
        loss = torch.relu(self.margin + neg_scores - pos_scores).mean()
        return loss

    def predict_tails(
        self,
        head_idx: int,
        relation_idx: int,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Given a head entity and relation, predict the most likely tail entities.

        Returns list of (entity_idx, score) tuples.
        """
        with torch.no_grad():
            h = self.entity_embeddings.weight[head_idx]
            r = self.relation_embeddings.weight[relation_idx]

            # Score all possible tails
            all_tails = self.entity_embeddings.weight
            scores = -torch.norm(h + r - all_tails, p=self.norm, dim=-1)

            top_scores, top_indices = torch.topk(scores, top_k)
            return [
                (idx.item(), score.item())
                for idx, score in zip(top_indices, top_scores)
            ]


class RotatE(nn.Module):
    """
    RotatE knowledge graph embedding model.

    Models relationships as rotations in complex space: t = h ∘ r
    where ∘ is the Hadamard product in complex space.

    Generally outperforms TransE on complex relation patterns.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 128,
        margin: float = 6.0,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.margin = margin

        # Complex embeddings (real + imaginary parts)
        self.entity_re = nn.Embedding(num_entities, embedding_dim)
        self.entity_im = nn.Embedding(num_entities, embedding_dim)
        self.relation_phase = nn.Embedding(num_relations, embedding_dim)

        # Initialize
        nn.init.xavier_uniform_(self.entity_re.weight)
        nn.init.xavier_uniform_(self.entity_im.weight)
        nn.init.uniform_(self.relation_phase.weight, -np.pi, np.pi)

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Compute RotatE score."""
        h_re = self.entity_re(head)
        h_im = self.entity_im(head)
        t_re = self.entity_re(tail)
        t_im = self.entity_im(tail)

        r_phase = self.relation_phase(relation)
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)

        # Complex multiplication: (h_re + i*h_im) * (r_re + i*r_im)
        rot_re = h_re * r_re - h_im * r_im
        rot_im = h_re * r_im + h_im * r_re

        # Score = -||h∘r - t||
        diff_re = rot_re - t_re
        diff_im = rot_im - t_im
        score = -torch.sqrt(diff_re ** 2 + diff_im ** 2 + 1e-9).sum(dim=-1)

        return score

    def forward(
        self,
        positive_triples: torch.Tensor,
        negative_triples: torch.Tensor,
    ) -> torch.Tensor:
        """Margin ranking loss."""
        pos_scores = self.score(
            positive_triples[:, 0],
            positive_triples[:, 1],
            positive_triples[:, 2],
        )
        neg_scores = self.score(
            negative_triples[:, 0],
            negative_triples[:, 1],
            negative_triples[:, 2],
        )

        loss = torch.relu(self.margin + neg_scores - pos_scores).mean()
        return loss
