"""
Implementation of BLP model.

Daniel Daza, Michael Cochez, and Paul Groth. 2020.
Inductive entity representations from text via link prediction.

PDF: https://arxiv.org/abs/2010.03496
Code: https://github.com/dfdazac/blp
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pykeen.losses import Loss
from torch_geometric.data import Data

from .base import QualifierModel, get_in_canonical_shape
from ..data.statement_factory import StatementsFactory

__all__ = [
    "BLP",
]


class BLP(QualifierModel):
    """The BLP model."""

    def __init__(
        self,
        statement_factory: StatementsFactory,
        loss: Loss,
        scoring_fct_name: str = 'transe',
        embedding_dim: int = 200,
        device: torch.device = None,
    ):
        super().__init__(statement_factory=statement_factory, device=device, loss=loss)
        self.embedding_dim = embedding_dim
        self.relation_embs = nn.Embedding(self.statement_factory.num_relations, self.embedding_dim)
        self.data_geometric = self.statement_factory.create_data_object()
        self.projection = nn.Linear(self.data_geometric.x[0].shape[-1], self.embedding_dim, bias=False)

        if scoring_fct_name == 'transe':
            self.scoring_fct = transe_score
            self.normalize_embs = True
        elif scoring_fct_name == 'distmult':
            self.scoring_fct = distmult_score
        elif scoring_fct_name == 'complex':
            self.scoring_fct = complex_score
        elif scoring_fct_name == 'simple':
            self.scoring_fct = simple_score
        else:
            raise ValueError(f'Unknown relational model {scoring_fct_name}.')

    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
        data_geometric: Optional[Data] = None,
        qualifier_indices: torch.LongTensor = None,
    ) -> torch.FloatTensor:
        # Get embeddings

        if data_geometric is None:
            data_geometric = self.data_geometric
        data_geometric = data_geometric.to(self.device)

        batch_size = r_indices.shape[0]

        h = self.projection(get_in_canonical_shape(emb=data_geometric.x, indices=h_indices))
        r = self.relation_embs(r_indices).unsqueeze(dim=1)
        t = self.projection(get_in_canonical_shape(emb=data_geometric.x, indices=t_indices))

        if self.normalize_embs:
            h = F.normalize(h, dim=-1)
            t = F.normalize(t, dim=-1)

        return self.scoring_fct(heads=h, rels=r, tails=t).view(batch_size, -1)


def transe_score(heads, tails, rels) -> torch.FloatTensor:
    return -torch.norm(heads + rels - tails, dim=-1, p=1, keepdim=True)


def distmult_score(heads, tails, rels) -> torch.FloatTensor:
    return torch.sum(heads * rels * tails, dim=-1)


def complex_score(heads, tails, rels) -> torch.FloatTensor:
    heads_re, heads_im = torch.chunk(heads, chunks=2, dim=-1)
    tails_re, tails_im = torch.chunk(tails, chunks=2, dim=-1)
    rels_re, rels_im = torch.chunk(rels, chunks=2, dim=-1)

    return torch.sum(rels_re * heads_re * tails_re +
                     rels_re * heads_im * tails_im +
                     rels_im * heads_re * tails_im -
                     rels_im * heads_im * tails_re,
                     dim=-1)


def simple_score(heads, tails, rels) -> torch.FloatTensor:
    heads_h, heads_t = torch.chunk(heads, chunks=2, dim=-1)
    tails_h, tails_t = torch.chunk(tails, chunks=2, dim=-1)
    rel_a, rel_b = torch.chunk(rels, chunks=2, dim=-1)

    return torch.sum(heads_h * rel_a * tails_t +
                     tails_h * rel_b * heads_t, dim=-1) / 2


def margin_loss(pos_scores, neg_scores) -> torch.FloatTensor:
    loss = 1 - pos_scores + neg_scores
    loss[loss < 0] = 0
    return loss.mean()


def l2_regularization(heads, tails, rels) -> torch.FloatTensor:
    reg_loss = 0.0
    for tensor in (heads, tails, rels):
        reg_loss += torch.mean(tensor ** 2)

    return reg_loss / 3.0


def nll_loss(pos_scores, neg_scores) -> torch.FloatTensor:
    return (F.softplus(-pos_scores).mean() + F.softplus(neg_scores).mean()) / 2
