# -*- coding: utf-8 -*-

"""Implementation of the StarE model."""
from typing import Callable, Optional

import torch
from pykeen.losses import Loss
from torch_geometric.data import Data

from .base import BaseQualifierModel
from .stare_representation import StarEEncoder, StarERepresentation
from ..data.statement_factory import StatementsFactory

__all__ = [
    "StarE",
]


class StarE(BaseQualifierModel):

    def __init__(
        self,
        statement_factory: StatementsFactory,
        embedding_dim: int = 500,
        num_layers: int = 2,
        use_bias: bool = True,
        feature_dim: int = None,
        hid_drop: float = 0.3,
        gcn_drop: float = 0.1,
        transformer_drop: float = 0.1,
        composition_function: Callable = None,
        qualifier_aggregation: str = "mul",
        qualifier_comp_function: Callable = None,
        use_attention: bool = False,
        num_attention_heads: Optional[int] = None,
        num_transformer_heads: int = 8,
        num_transformer_layers: int = 2,
        dim_transformer_hidden: int = 2048,
        attention_slope: Optional[float] = None,
        attention_drop: Optional[float] = None,
        triple_qual_weight: Optional[float] = None,
        use_learnable_x=True,
        affine_transformation: bool = False,
        loss: Optional[Loss] = None,
        device: torch.device = None,
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
            device=device,
        )
        self.star_e_encoder = StarEEncoder(
            data_geometric=self.data_geometric,
            statement_factory=statement_factory,
            feature_dim=feature_dim,
            use_learnable_x=use_learnable_x,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            hid_drop=hid_drop,
            use_bias=use_bias,
            gcn_drop=gcn_drop,
            composition_function=composition_function,
            qualifier_aggregation=qualifier_aggregation,
            qualifier_comp_function=qualifier_comp_function,
            use_attention=use_attention,
            num_attention_heads=num_attention_heads,
            attention_slope=attention_slope,
            attention_drop=attention_drop,
            triple_qual_weight=triple_qual_weight,
            device=self.device,
        )
        self.entity_representations, self.relation_representations = [
            StarERepresentation(
                base=self.star_e_encoder,
                component_index=i,
            )
            for i in (0, 1)
        ]

    def _get_relation_representations(
        self,
        indices: torch.LongTensor,
        data_geometric: Optional[Data],
    ) -> torch.FloatTensor:
        return self.relation_representations.forward(indices=indices, data_geometric=data_geometric)

    def _get_entity_representations(
        self,
        indices: Optional[torch.LongTensor],
        data_geometric: Optional[Data],
    ) -> torch.FloatTensor:
        return self.entity_representations.forward(indices=indices, data_geometric=data_geometric)
