# -*- coding: utf-8 -*-

"""Implementation of the StarE encoder."""
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.data import Data

from .layers import StarEConvLayer, mult
from ..data.statement_factory import StatementsFactory
from ..utils import get_parameter

__all__ = [
    "StarEEncoder",
    "StarERepresentation",
]


class StarEEncoder(nn.Module):
    """Representations enriched by StarE."""

    #: The buffered embeddings
    enriched_embeddings: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]

    def __init__(
        self,
        data_geometric: Data,
        statement_factory: StatementsFactory,
        feature_dim: Optional[int] = 1024,
        embedding_dim: int = 200,
        num_layers: int = 2,
        hid_drop: float = 0.3,
        use_bias: bool = False,
        gcn_drop: float = 0.2,
        composition_function: Callable = None,
        qualifier_aggregation: str = "mul",
        qualifier_comp_function: Callable = None,
        use_attention: bool = False,
        num_attention_heads: Optional[int] = None,
        attention_slope: Optional[float] = None,
        attention_drop: Optional[float] = None,
        triple_qual_weight: Optional[float] = None,
        device: torch.device = None,
        use_learnable_x: bool = True,
    ):
        super().__init__()

        # default
        composition_function = composition_function or mult
        qualifier_comp_function = qualifier_comp_function or composition_function

        self.statement_factory = statement_factory
        self.data_geometric = data_geometric
        self.use_learnable_x = use_learnable_x

        if use_learnable_x:
            self.node_features = Parameter(self.data_geometric.x)
            self.data_geometric.x = self.node_features
            self.data_geometric.x.data[0] = 0  # Padding

        self.device = device

        self.triple_mode = statement_factory.statement_length == 3
        self.drop1 = torch.nn.Dropout(hid_drop)
        self.num_rel = statement_factory.num_relations

        if feature_dim is None:
            feature_dim = statement_factory.feature_dim
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.initial_relation_embs = get_parameter((self.num_rel, embedding_dim))
        self.initial_relation_embs.data[0] = 0  # padding

        self.feature_reduction = nn.Linear(feature_dim, embedding_dim)

        self.composition_function = composition_function
        self.qualifier_comp_function = qualifier_comp_function

        self.convs = nn.ModuleList()
        self.convs.append(StarEConvLayer(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            embedding_dim=embedding_dim,
            statement_factory=statement_factory,
            composition_function=composition_function,
            qualifier_aggregation=qualifier_aggregation,
            qualifier_comp_function=qualifier_comp_function,
            gcn_drop=gcn_drop,
            use_bias=use_bias,
            use_attention=use_attention,
            num_attention_heads=num_attention_heads,
            attention_slope=attention_slope,
            attention_drop=attention_drop,
            triple_qual_weight=triple_qual_weight,
            device=device,
        ))

        for _ in range(self.num_layers - 1):
            self.convs.append(StarEConvLayer(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                embedding_dim=embedding_dim,
                statement_factory=statement_factory,
                composition_function=composition_function,
                qualifier_aggregation=qualifier_aggregation,
                qualifier_comp_function=qualifier_comp_function,
                gcn_drop=gcn_drop,
                use_bias=use_bias,
                use_attention=use_attention,
                num_attention_heads=num_attention_heads,
                attention_slope=attention_slope,
                attention_drop=attention_drop,
                triple_qual_weight=triple_qual_weight,
                device=device,
            ))
        self.enriched_embeddings = None
        self.reset_parameters()

    def post_parameter_update(self) -> None:  # noqa: D102
        # invalidate enriched embeddings
        self.enriched_embeddings = None

    def reset_parameters(self):
        if self.use_learnable_x:
            torch.nn.init.xavier_normal_(self.data_geometric.x)

        self.feature_reduction.reset_parameters()

        if self.composition_function.__name__ == 'rotate' or self.qualifier_comp_function.__name__ == 'rotate':
            phases = 2 * np.pi * torch.rand((self.num_rel - 1) // 2, self.embedding_dim // 2)
            self.initial_relation_embs = nn.Parameter(
                torch.cat(
                    [torch.zeros(1, self.embedding_dim), torch.cat(
                        [torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1),
                         torch.cat([torch.cos(phases), -torch.sin(phases)], dim=-1)], dim=-1).view(-1,
                                                                                                   self.embedding_dim)],
                    dim=0)
            )

            self.initial_relation_embs.data[0] = 0  # padding
        else:
            torch.nn.init.xavier_normal_(self.initial_relation_embs.data)

        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data_geometric: Data = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Calculate enriched entity and relation representations."""
        # use buffering
        if self.enriched_embeddings is not None:
            return self.enriched_embeddings

        if data_geometric is None:
            data_geometric = self.data_geometric.to(self.device)
        else:
            data_geometric = data_geometric.to(self.device)

        x = data_geometric.x
        edge_index = data_geometric.edge_index  # .to(self.device)
        edge_type = data_geometric.edge_type  # .to(self.device)
        qualifier_index = data_geometric.qualifier_index  # .to(self.device)

        relation_embeddings = self.initial_relation_embs  # let's not do any transe-specific tricks

        # this is only for the node features setup when feature_dim is 1024 from SBERT
        if self.feature_dim != self.embedding_dim:
            x = self.feature_reduction(x)

        # GNN forward prop
        for i, conv in enumerate(self.convs):
            x, relation_embeddings = conv(
                x=x,
                edge_index=edge_index,
                edge_type=edge_type,
                rel_embed=relation_embeddings,
                quals=qualifier_index,
            )
            x = self.drop1(x)

        self.enriched_embeddings = (x, relation_embeddings)
        return self.enriched_embeddings


class StarERepresentation:
    """Entity/Relation representations from StarE."""

    def __init__(self, base: StarEEncoder, component_index: int):
        # component_index: either 0 (for entities) or 1 (for relations)
        self.base = base
        self.component_index = component_index

    def get_in_canonical_shape(
        self,
        x: torch.FloatTensor,
        indices: Optional[torch.LongTensor],
    ) -> torch.FloatTensor:
        """Get representations in canonical shape: (batch_size?, num?, dim)."""
        if indices is None:
            return x
        return x[indices]

    def post_parameter_update(self):
        # delegate to base encoder
        self.base.post_parameter_update()

    def forward(self, indices: Optional[torch.LongTensor], data_geometric: Data) -> torch.FloatTensor:
        return self.get_in_canonical_shape(x=self.base(data_geometric)[self.component_index], indices=indices)
