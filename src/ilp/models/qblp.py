"""The QBLP model."""
from typing import Optional

import torch
from pykeen.losses import Loss
from torch import nn
from torch_geometric.data import Data

from .base import BaseQualifierModel
from ..data.statement_factory import StatementsFactory

__all__ = [
    "QBLP",
]


class QBLP(BaseQualifierModel):
    """BLP variant making use of qualifiers."""

    def __init__(
        self,
        statement_factory: StatementsFactory,
        loss: Loss,
        embedding_dim: int = 500,
        transformer_drop: float = 0.1,
        num_transformer_heads: int = 8,
        num_transformer_layers: int = 2,
        dim_transformer_hidden: int = 2048,
        affine_transformation: bool = False,
    ):
        super().__init__(
            statement_factory=statement_factory,
            loss=loss,
            embedding_dim=embedding_dim,
            transformer_drop=transformer_drop,
            num_transformer_heads=num_transformer_heads,
            num_transformer_layers=num_transformer_layers,
            dim_transformer_hidden=dim_transformer_hidden,
            affine_transformation=affine_transformation,
        )
        self.relation_embeddings = nn.Embedding(
            statement_factory.num_relations,
            embedding_dim,
        )
        self.projection = nn.Linear(self.statement_factory.feature_dim, embedding_dim)

    def _get_entity_representations(
        self,
        indices: torch.LongTensor,
        data_geometric: Optional[Data],
    ) -> torch.FloatTensor:
        if data_geometric is None:
            data_geometric = self.data_geometric
        x = data_geometric.x.to(self.device)
        return self.projection(x[indices])

    def _get_relation_representations(
        self,
        indices: torch.LongTensor,
        data_geometric: Optional[Data],
    ) -> torch.FloatTensor:
        return self.relation_embeddings(indices)
