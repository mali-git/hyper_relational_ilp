# -*- coding: utf-8 -*-

"""Implementation of the StarE convolution layer."""
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

from ..data.statement_factory import StatementsFactory
from ..utils import get_parameter, softmax

__all__ = [
    "StarEConvLayer",
]


class StarEConvLayer(MessagePassing):
    """StarE's convolution layer with qualifiers """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        statement_factory: StatementsFactory,
        gcn_drop: float = 0.2,
        composition_function: Callable = None,
        qualifier_aggregation: str = None,
        qualifier_comp_function: Callable = None,
        use_bias: bool = True,
        use_attention: bool = False,
        embedding_dim: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        attention_slope: Optional[float] = None,
        attention_drop: Optional[float] = None,
        triple_qual_weight: Optional[float] = None,
        device: torch.device = None,
    ):
        super(self.__class__, self).__init__(flow='target_to_source', aggr='add')

        self.device = device

        self.create_inverse_triples = statement_factory.create_inverse_triples
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = statement_factory.num_relations
        self.act = torch.relu
        self.use_bias = use_bias
        self.use_attention = use_attention
        self.statement_len = statement_factory.statement_length
        self.triple_qual_weight = triple_qual_weight

        self.composition_function = composition_function
        self.qualifier_aggregation = qualifier_aggregation  # sum / concat / attn
        self.qualifier_scoring = qualifier_comp_function  # corr / sub / mul / rotate

        self.w_loop = get_parameter((in_channels, out_channels))
        self.w_in = get_parameter((in_channels, out_channels))
        self.w_out = get_parameter((in_channels, out_channels))
        self.w_rel = get_parameter((in_channels, out_channels))

        if self.statement_len != 3:
            self.w_q = get_parameter((in_channels, in_channels))

        self.loop_rel = get_parameter((1, in_channels))

        self.drop = torch.nn.Dropout(gcn_drop)
        self.bn = torch.nn.BatchNorm1d(out_channels)

        if self.use_attention:
            # assert gcn_dim == embedding_dim, "Current attn implementation requires those to be identical"
            assert embedding_dim % num_attention_heads == 0, "should be divisible"
            self.heads = num_attention_heads
            self.attn_dim = self.out_channels // self.heads
            self.negative_slope = attention_slope
            self.attn_drop = attention_drop
            self.att = get_parameter((1, self.heads, 2 * self.attn_dim))

        if self.qualifier_aggregation == "attn":
            # assert gcn_dim == embedding_dim, "Current attn implementation requires those to be identical"
            assert embedding_dim % num_attention_heads == 0, "should be divisible"
            if not self.use_attention:
                self.heads = num_attention_heads
                self.attn_dim = self.out_channels // self.heads
                self.negative_slope = attention_slope
                self.attn_drop = attention_drop
            self.att_qual = get_parameter((1, self.heads, 2 * self.attn_dim))

        if self.use_bias:
            self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.w_loop.data)
        torch.nn.init.xavier_normal_(self.w_in.data)
        torch.nn.init.xavier_normal_(self.w_out.data)
        torch.nn.init.xavier_normal_(self.w_rel.data)
        if self.statement_len != 3:
            torch.nn.init.xavier_normal_(self.w_q.data)
        torch.nn.init.xavier_normal_(self.loop_rel.data)
        self.bn.reset_parameters()
        if self.use_attention:
            torch.nn.init.xavier_normal_(self.att.data)
        if self.use_attention:
            torch.nn.init.xavier_normal_(self.att_qual.data)

    def forward(
        self,
        x,
        edge_index,
        edge_type,
        rel_embed,
        quals=None,
    ):
        """Forward pass through the convolution layer."""

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_triples = edge_index.size(1) // 2
        num_ent = x.size(0)

        if self.create_inverse_triples:
            self.in_index, self.out_index = edge_index[:, :num_triples], edge_index[:, num_triples:]
            self.in_type, self.out_type = edge_type[:num_triples], edge_type[num_triples:]
        else:
            pass
        if self.statement_len != 3:
            num_quals = quals.size(1) // 2
            self.in_index_qual_ent, self.out_index_qual_ent = quals[1, :num_quals], quals[1, num_quals:]
            self.in_index_qual_rel, self.out_index_qual_rel = quals[0, :num_quals], quals[0, num_quals:]
            self.quals_index_in, self.quals_index_out = quals[2, :num_quals], quals[2, num_quals:]

        # Self edges between all the nodes
        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1,
                                    dtype=torch.long).to(self.device)  # if rel meb is 500, the index of the self emb is
        # 499 .. which is just added here

        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)

        if self.statement_len != 3:
            in_res = self.propagate(
                self.in_index,
                x=x,
                edge_type=self.in_type,
                rel_embed=rel_embed,
                edge_norm=self.in_norm,
                mode='in',
                ent_embed=x,
                qualifier_ent=self.in_index_qual_ent,
                qualifier_rel=self.in_index_qual_rel,
                qual_index=self.quals_index_in,
                source_index=self.in_index[0],
            )

            loop_res = self.propagate(
                self.loop_index,
                x=x,
                edge_type=self.loop_type,
                rel_embed=rel_embed,
                edge_norm=None,
                mode='loop',
                ent_embed=None,
                qualifier_ent=None,
                qualifier_rel=None,
                qual_index=None,
                source_index=None,
            )

            out_res = self.propagate(
                self.out_index,
                x=x,
                edge_type=self.out_type,
                rel_embed=rel_embed,
                edge_norm=self.out_norm,
                mode='out',
                ent_embed=x,
                qualifier_ent=self.out_index_qual_ent,
                qualifier_rel=self.out_index_qual_rel,
                qual_index=self.quals_index_out,
                source_index=self.out_index[0],
            )
        else:
            in_res = self.propagate(
                self.in_index,
                x=x,
                edge_type=self.in_type,
                rel_embed=rel_embed,
                edge_norm=self.in_norm,
                mode='in',
                ent_embed=x,
                qualifier_ent=None,
                qualifier_rel=None,
                qual_index=None,
                source_index=self.in_index[0],
            )

            loop_res = self.propagate(
                self.loop_index,
                x=x,
                edge_type=self.loop_type,
                rel_embed=rel_embed,
                edge_norm=None, mode='loop',
                ent_embed=x,
                qualifier_ent=None,
                qualifier_rel=None,
                qual_index=None,
                source_index=None,
            )

            out_res = self.propagate(
                self.out_index,
                x=x,
                edge_type=self.out_type,
                rel_embed=rel_embed,
                edge_norm=self.out_norm,
                mode='out',
                ent_embed=x,
                qualifier_ent=None,
                qualifier_rel=None,
                qual_index=None,
                source_index=self.out_index[0],
            )

        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)

        if self.use_bias:
            out = out + self.bias

        out = self.bn(out)

        # Ignoring the self loop inserted, return.
        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]

    def update_rel_emb_with_qualifier(
        self,
        ent_embed,
        rel_embed,
        qualifier_ent,
        qualifier_rel,
        edge_type,
        qual_index=None,
    ):
        """."""
        # Step 1: embedding
        qualifier_emb_rel = rel_embed[qualifier_rel]
        qualifier_emb_ent = ent_embed[qualifier_ent]

        rel_part_emb = rel_embed[edge_type]

        # Step 2: pass it through qual_transform
        qualifier_emb = self.transform_qualifiers(qualifier_ent=qualifier_emb_ent, qualifier_rel=qualifier_emb_rel)

        # Pass it through a aggregate layer
        return self.aggregate_qualifiers(
            qualifier_emb,
            rel_part_emb,
            alpha=self.triple_qual_weight,
            qual_index=qual_index,
        )

    def transform_relations(self, ent_embed, rel_embed):
        if self.composition_function.__name__ == 'ccorr':
            transformed_relations = ccorr(ent_embed, rel_embed)
        elif self.composition_function.__name__ == 'sub':
            transformed_relations = sub(ent_embed, rel_embed)
        elif self.composition_function.__name__ == 'mult':
            transformed_relations = mult(ent_embed, rel_embed)
        elif self.composition_function.__name__ == 'complex_interaction':
            transformed_relations = mult(ent_embed, rel_embed)
        elif self.composition_function.__name__ == 'rotate':
            transformed_relations = rotate(ent_embed, rel_embed)
        else:
            raise NotImplementedError

        return transformed_relations

    def transform_qualifiers(self, qualifier_ent, qualifier_rel):
        """
        :return:
        """
        if self.qualifier_scoring.__name__ == 'ccorr':
            transformed_qualifiers = ccorr(qualifier_ent, qualifier_rel)
        elif self.qualifier_scoring.__name__ == 'sub':
            transformed_qualifiers = sub(qualifier_ent, qualifier_rel)
        elif self.qualifier_scoring.__name__ == 'mult':
            transformed_qualifiers = mult(qualifier_ent, qualifier_rel)
        elif self.qualifier_scoring.__name__ == 'rotate':
            transformed_qualifiers = rotate(qualifier_ent, qualifier_rel)
        else:
            raise NotImplementedError

        return transformed_qualifiers

    def aggregate_qualifiers(self, qualifier_emb, rel_part_emb, alpha=0.5, qual_index=None):
        if self.qualifier_aggregation == 'sum':
            qualifier_emb = torch.einsum(
                'ij,jk -> ik',
                self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0]),
                self.w_q,
            )
            return alpha * rel_part_emb + (1 - alpha) * qualifier_emb  # [N_EDGES / 2 x EMB_DIM]
        elif self.qualifier_aggregation == 'concat':
            qualifier_emb = self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0])
            agg_rel = torch.cat((rel_part_emb, qualifier_emb), dim=1)  # [N_EDGES / 2 x 2 * EMB_DIM]
            return torch.mm(agg_rel, self.w_q)  # [N_EDGES / 2 x EMB_DIM]
        elif self.qualifier_aggregation == 'mul':
            qualifier_emb = torch.mm(
                self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0], fill=1),
                self.w_q,
            )
            return rel_part_emb * qualifier_emb
        elif self.qualifier_aggregation == "attn":
            expanded_rels = torch.index_select(rel_part_emb, 0, qual_index)  # Nquals x D
            expanded_rels = expanded_rels.view(-1, self.heads, self.attn_dim)  # Nquals x heads x h_dim
            qualifier_emb = torch.mm(qualifier_emb, self.w_q).view(-1, self.heads,
                                                                   self.attn_dim)  # Nquals x heads x h_dim

            alpha_r = torch.einsum('bij,kij -> bi', [torch.cat([expanded_rels, qualifier_emb], dim=-1), self.att_qual])
            alpha_r = F.leaky_relu(alpha_r, self.negative_slope)  # Nquals x heads
            alpha_r = softmax(alpha_r, qual_index, rel_part_emb.size(0))  # Nquals x heads
            alpha_r = F.dropout(alpha_r, p=self.attn_drop)  # Nquals x heads
            expanded_rels = (expanded_rels * alpha_r.view(-1, self.heads, 1)).view(-1,
                                                                                   self.heads * self.attn_dim)  # Nquals x D
            single_rels = scatter_add(expanded_rels, qual_index, dim=0, dim_size=rel_part_emb.size(0))  # Nedges x D
            copy_mask = single_rels.sum(dim=1) != 0.0
            rel_part_emb[copy_mask] = single_rels[copy_mask]  # Nedges x D
            return rel_part_emb
        else:
            raise NotImplementedError

    # return qualifier_emb
    def message(
        self,
        x_j,
        x_i,
        edge_type,
        rel_embed,
        edge_norm,
        mode,
        ent_embed=None,
        qualifier_ent=None,
        qualifier_rel=None,
        qual_index=None,
        source_index=None,
    ):
        """."""
        weight = getattr(self, 'w_{}'.format(mode))

        if self.statement_len != 3:
            if mode != 'loop':
                rel_emb = self.update_rel_emb_with_qualifier(
                    ent_embed,
                    rel_embed,
                    qualifier_ent,
                    qualifier_rel,
                    edge_type,
                    qual_index,
                )
            else:
                rel_emb = torch.index_select(rel_embed, 0, edge_type)
        else:
            rel_emb = torch.index_select(rel_embed, 0, edge_type)

        # Use relations to transform entities
        # Note: Note that we generally refer to i as the central nodes that aggregates information, and refer to j
        # as the neighboring nodes, since this is the most common notation.
        xj_rel = self.transform_relations(x_j, rel_emb)
        out = torch.einsum('ij,jk->ik', xj_rel, weight)

        if self.use_attention and mode != 'loop':
            out = out.view(-1, self.heads, self.attn_dim)
            x_i = x_i.view(-1, self.heads, self.attn_dim)

            alpha = torch.einsum('bij,kij -> bi', [torch.cat([x_i, out], dim=-1), self.att])
            alpha = F.leaky_relu(alpha, self.negative_slope)
            # Computes a sparsely evaluated softmax
            alpha = softmax(alpha, source_index, ent_embed.size(0))
            alpha = F.dropout(alpha, p=self.attn_drop)
            return out * alpha.view(-1, self.heads, 1).view(-1, self.heads * self.attn_dim)
        else:
            return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, mode):
        if self.use_attention and mode != 'loop':
            aggr_out = aggr_out.view(-1, self.heads * self.attn_dim)

        return aggr_out

    def coalesce_quals(self,
                       qual_embeddings,
                       qual_index,
                       num_edges,
                       fill=0):
        output = scatter_add(qual_embeddings, qual_index, dim=0, dim_size=num_edges)

        if fill != 0:
            # by default scatter_ functions assign zeros to the output, so we assign them 1's for correct mult
            mask = output.sum(dim=-1) == 0
            output[mask] = fill

        return output

    @staticmethod
    def compute_norm(edge_index, num_ent):
        """."""
        row, col = edge_index
        edge_weight = torch.ones_like(
            row).float()  # Identity matrix where we know all entities are there
        deg = scatter_add(edge_weight, row, dim=0,
                          dim_size=num_ent)  # Summing number of weights of
        # the edges, D = A + I
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0  # for numerical stability
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # Norm parameter D^{-0.5} *

        return norm


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def cconv(a, b):
    return torch.irfft(
        com_mult(
            torch.rfft(a, 1),
            torch.rfft(b, 1),
        ),
        1,
        signal_sizes=(a.shape[-1],),
    )


def ccorr(a, b):
    return torch.irfft(
        com_mult(
            conj(torch.rfft(a, 1)),
            torch.rfft(b, 1)
        ),
        1,
        signal_sizes=(a.shape[-1],),
    )


def rotate(h, r):
    # re: first half, im: second half
    # assume embedding dim is the last dimension
    d = h.shape[-1]
    h_re, h_im = torch.split(h, d // 2, -1)
    r_re, r_im = torch.split(r, d // 2, -1)
    return torch.cat([h_re * r_re - h_im * r_im,
                      h_re * r_im + h_im * r_re], dim=-1)


def sub(a, b):
    return a - b


def mult(a, b):
    return a * b


def hole_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:  # noqa: D102
    """
    Evaluate the HolE interaction function.

    :param h: shape: (batch_size, num_heads, dim)
        The head representations.
    :param r: shape: (batch_size, num_relations, dim)
        The relation representations.
    :param t: shape: (batch_size, num_tails, dim)
        The tail representations.

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    # Circular correlation of entity embeddings
    a_fft = torch.fft.rfft(h, dim=-1)
    b_fft = torch.fft.rfft(t, dim=-1)

    # complex conjugate, shape = (b, h, d)
    a_fft = torch.conj(a_fft)

    # Hadamard product in frequency domain, shape: (b, h, t, d)
    p_fft = a_fft.unsqueeze(dim=2) * b_fft.unsqueeze(dim=1)

    # inverse real FFT, shape: (b, h, t, d)
    composite = torch.fft.irfft(p_fft, n=h.shape[-1], dim=-1)

    # inner product with relation embedding
    return torch.einsum("bhtd,brd->bhrt", composite, r)
