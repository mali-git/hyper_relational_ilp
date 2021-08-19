"""Interface for models."""
from abc import abstractmethod
from typing import Optional

import torch
from pykeen.losses import Loss
from pykeen.utils import resolve_device
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.data import Data

from ..data.statement_factory import StatementsFactory

__all__ = [
    "get_in_canonical_shape",
    "QualifierModel",
    "BaseQualifierModel",
]


def get_in_canonical_shape(
    emb: torch.nn.Embedding,
    indices: Optional[torch.LongTensor],
) -> torch.FloatTensor:
    """Get representations in canonical shape: (batch_size?, num?, dim)."""
    if indices is None:
        return emb.unsqueeze(dim=0)
    x = emb[indices]
    if indices.ndim < 2:
        x = x.unsqueeze(dim=1)
    return x


class QualifierModel(nn.Module):
    """A model for link prediction with qualifiers."""

    def __init__(
        self,
        statement_factory: StatementsFactory,
        loss: Loss,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = resolve_device(device=device)
        self.loss = loss
        self.statement_factory = statement_factory
        # compatability with pykeen
        self._random_seed = 0

    def reset_parameters(self):
        # recursively initialize all submodules
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def post_parameter_update(self) -> None:
        # recursively update all submodules
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "post_parameter_update"):
                module.post_parameter_update()

    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
        qualifier_indices: Optional[torch.LongTensor],
        data_geometric: Optional[Data] = None,
    ) -> torch.FloatTensor:
        """
        :param h_indices: shape: (batch_size,)
            The head indices. None indicates to use all.
        :param r_indices: shape: (batch_size,)
            The relation indices. None indicates to use all.
        :param t_indices: shape: (batch_size,)
            The tail indices. None indicates to use all.
        :param qualifier_indices: shape: (batch_size, 2 * max_num_qualifier_pairs)
            The qualifier indices
        :param data_geometric:
            The data geometric object containing the graph.

        :return: shape: (batch_size, num_entities)
            The score for each triple.
        """
        raise NotImplementedError

    def score_t(
        self,
        hr_batch: torch.LongTensor,
        qualifiers: torch.LongTensor,
        data_geometric: Optional[Data] = None,
    ) -> torch.FloatTensor:
        h, r = hr_batch.t()
        return self.forward(
            h_indices=h,
            r_indices=r,
            t_indices=None,
            qualifier_indices=qualifiers,
            data_geometric=data_geometric,
        )

    def score_h(
        self,
        rt_batch: torch.LongTensor,
        qualifiers: torch.LongTensor,
        data_geometric: Optional[Data] = None,
    ) -> torch.FloatTensor:
        r, t = rt_batch.t()
        if self.statement_factory.create_inverse_triples:
            # todo refactor this to support inverses of inverses?
            r_inv = r + 1
            return self.forward(
                h_indices=t,
                r_indices=r_inv,
                t_indices=None,
                qualifier_indices=qualifiers,
                data_geometric=data_geometric,
            )
        return self.forward(
            h_indices=None,
            r_indices=r,
            t_indices=t,
            qualifier_indices=qualifiers,
            data_geometric=data_geometric,
        )

    # compatibility with pykeen
    def reset_parameters_(self):
        self.reset_parameters()

    # compatibility with pykeen
    def get_grad_params(self):
        return (p for p in self.parameters() if p.requires_grad)

    # compatibility with pykeen
    @property
    def num_entities(self) -> int:
        return self.statement_factory.num_entities

    # compatibility with pykeen
    def post_forward_pass(self):
        pass

    # compatibility with pykeen
    def _free_graph_and_cache(self):
        pass


class BaseQualifierModel(QualifierModel):
    """Common functionality."""

    def __init__(
        self,
        statement_factory: StatementsFactory,
        loss: Loss,
        embedding_dim: int = 500,
        transformer_drop: float = 0.1,
        num_transformer_heads: int = 8,
        num_transformer_layers: int = 2,
        dim_transformer_hidden: int = 2048,
        device: Optional[torch.device] = None,
        affine_transformation: bool = False,
    ):
        super().__init__(statement_factory=statement_factory, device=device, loss=loss)

        # Positional encoding for transformer
        self.position_embeddings = nn.Embedding(self.statement_factory.statement_length - 1, embedding_dim)

        # Transformer
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_transformer_heads,
            dim_feedforward=dim_transformer_hidden,
            dropout=transformer_drop,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_transformer_layers)
        self.linear_transformation = nn.Linear(embedding_dim, embedding_dim)

        self.data_geometric = statement_factory.create_data_object().to(self.device)
        if affine_transformation:
            # the scaling factor alpha is modelled to be positive to avoid the problem of "switching signs" and the
            # connected instability. Thus, we use a free parameter log_alpha, and compute alpha = exp(log_alpha)
            self.log_alpha = nn.Parameter(torch.zeros(1))
            self.beta = nn.Parameter(torch.zeros(1))
        else:
            self.log_alpha = self.beta = None

    def forward(
        self,
        h_indices: Optional[torch.LongTensor],
        r_indices: Optional[torch.LongTensor],
        t_indices: Optional[torch.LongTensor],
        qualifier_indices: Optional[torch.LongTensor] = None,
        data_geometric: Optional[Data] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        if h_indices is None:
            raise NotImplementedError("Please use inverse triples.")

        # Bind batch size to variable
        batch_size = h_indices.shape[0]

        # shape: (batch_size, dim)
        h = self._get_entity_representations(indices=h_indices, data_geometric=data_geometric)
        # shape: (batch_size, dim)
        r = self._get_relation_representations(indices=r_indices, data_geometric=data_geometric)

        # create empty qualifier indices if not provided.
        # use Tensor.new_empty to ensure the same device and dtype.
        if qualifier_indices is None:
            qualifier_indices = h_indices.new_empty(batch_size, 0)

        # get qualifier representations
        qual_ent_indices = qualifier_indices[:, 1::2]
        qual_rel_indices = qualifier_indices[:, 0::2]

        qual_ent_emb = self._get_entity_representations(indices=qual_ent_indices, data_geometric=data_geometric)
        qual_rel_emb = self._get_relation_representations(indices=qual_rel_indices, data_geometric=data_geometric)

        # concatenate: shape: (batch_size, 2 + max_len, dim)
        quals = torch.stack((qual_rel_emb, qual_ent_emb), 2).view(batch_size, 2 * qual_rel_emb.shape[1], qual_rel_emb.shape[2])

        if h_indices.ndimension() > 1:
            # negative batch and num_neg_per_pos > 1
            # repeat r & q -> larger batch size
            num_negatives = h_indices.shape[1]
            batch_size = batch_size * num_negatives
            h = h.view(batch_size, 1, -1)
            r = r.unsqueeze(dim=1).repeat(1, num_negatives, 1).view(batch_size, 1, -1)
            quals = quals.unsqueeze(dim=1).repeat(1, num_negatives, 1, 1).view(batch_size, -1, h.shape[-1])
            qual_ent_indices = qual_ent_indices.unsqueeze(dim=1).repeat(1, num_negatives, 1).view(batch_size, -1)
            qual_rel_indices = qual_rel_indices.unsqueeze(dim=1).repeat(1, num_negatives, 1).view(batch_size, -1)
        else:
            num_negatives = None
            h = h.unsqueeze(dim=1)
            r = r.unsqueeze(dim=1)
        stacked_inp = torch.cat([h, r, quals], dim=1)
        # transformer expects input in shape: (seq_len, batch_size, dim)
        stacked_inp = stacked_inp.transpose(0, 1)

        # Create mask with True where qualifier entities/relation are just padding
        seq_length = stacked_inp.shape[0]
        mask = stacked_inp.new_zeros(batch_size, seq_length, dtype=torch.bool)
        mask[:, 2::2] = qual_rel_indices == StatementsFactory.padding_idx
        mask[:, 3::2] = qual_ent_indices == StatementsFactory.padding_idx

        # get position embeddings, shape: (seq_len, dim)
        # Now we are position-dependent w.r.t qualifier pairs.
        positions = torch.arange(seq_length, dtype=torch.long, device=self.device)
        pos_embeddings = self.position_embeddings(positions)
        stacked_inp = stacked_inp + pos_embeddings.unsqueeze(dim=1)

        x = self.transformer_encoder(stacked_inp, src_key_padding_mask=mask)  # output shape: (seq_len, batch_size, dim)

        # average pooling over all the seq_len members
        # pool only over non-padding values
        mask = mask.t().unsqueeze(dim=-1)
        mask = (~mask).float()
        x = (mask * x).sum(dim=0) / mask.sum(dim=0).clamp_min(1.0)  # output shape: (batch_size, dim)

        x = self.linear_transformation(x)  # output shape: (batch_size', dim)
        if num_negatives is not None:
            batch_size = batch_size // num_negatives
            x = x.view(batch_size, num_negatives, -1)
        else:
            x = x.view(batch_size, 1, -1)
        # x.shape: (batch_size, num_negatives?, dim)

        if t_indices is None:
            # Ensure that in full inductive setting, entities are scored against unseen entities
            if data_geometric is None:
                data_geometric = self.data_geometric
            t_indices = data_geometric.entities.to(self.device)
        elif t_indices.ndimension() < 2:
            t_indices = t_indices.unsqueeze(dim=-1)
        # t_indices.shape: (batch_size, num_negatives/num_entities)
        t = self._get_entity_representations(indices=t_indices, data_geometric=data_geometric)
        # t.shape = (batch_size, num_negatives/num_entities, dim)

        x = (x @ t.transpose(-1, -2)).view(batch_size, -1)

        if self.log_alpha is not None:
            x = self.log_alpha.exp() * x + self.beta

        return x

    @abstractmethod
    def _get_entity_representations(
        self,
        indices: torch.LongTensor,
        data_geometric: Optional[Data],
    ) -> torch.FloatTensor:
        raise NotImplementedError

    @abstractmethod
    def _get_relation_representations(
        self,
        indices: torch.LongTensor,
        data_geometric: Optional[Data],
    ) -> torch.FloatTensor:
        raise NotImplementedError
