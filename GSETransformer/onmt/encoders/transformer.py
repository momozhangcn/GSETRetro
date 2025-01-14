"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GNN_block(nn.Module):
    def __init__(self,  d_model, heads, dropout=0.1):
        super(GNN_block, self).__init__()
        self.d_model = d_model
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.linear_before_gnn = nn.Linear(d_model, d_model)
        self.gnn_conv_1 = GATConv(d_model, d_model, heads=heads, dropout=dropout, concat=False)
        self.linear_after_gnn = nn.Linear(d_model, d_model)
        self.dropout_gnn = nn.Dropout(dropout)

    def forward(self, inputs, edge_index):
        input_norm = self.layer_norm_1(inputs)

        input_norm_resize = input_norm.view(-1, self.d_model)
        gnn_in = self.linear_before_gnn(input_norm_resize)
        gnn_out_1 = self.gnn_conv_1(gnn_in, edge_index)
        gnn_out_2 = self.linear_after_gnn(gnn_out_1)
        gnn_out_resize = gnn_out_2.view(input_norm.shape[0], -1, self.d_model)
        gnn_out = self.dropout_gnn(gnn_out_resize) + inputs
        return gnn_out
    def update_dropout(self, dropout):
        self.dropout_gnn.p = dropout
        self.gnn_conv_1.dropout = dropout

class GSETransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 max_relative_positions=0):
        super(GSETransformerEncoderLayer, self).__init__()
        self.d_model = d_model

        self.gnn_block = GNN_block(d_model, heads, dropout)

        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout,
            max_relative_positions=max_relative_positions)

        self.dropout = nn.Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, edge_index, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        # component_01, GNN_with_res
        gnn_out = self.gnn_block(inputs, edge_index)
        #component_02, multi_head_attention
        gnn_out_norm_2attn = self.layer_norm_2(gnn_out)
        context, _ = self.self_attn(gnn_out_norm_2attn, gnn_out_norm_2attn, gnn_out_norm_2attn,
                                    mask=mask, type="self")
        out = self.dropout(context) + gnn_out
        # component_03, feed_forward
        return self.feed_forward(out)

    def update_dropout(self, dropout):
        self.self_attn.update_dropout(dropout)
        self.feed_forward.update_dropout(dropout)
        self.gnn_block.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings,
                 max_relative_positions):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.GSETransformer = nn.ModuleList(
            [GSETransformerEncoderLayer(
                d_model, heads, d_ff, dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src_input, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        src = src_input[0]
        edge_index = src_input[1]
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
        # Run the forward pass of every layer of the tranformer.
        for layer in self.GSETransformer:
            out = layer(out, edge_index, mask)
        out = self.layer_norm(out)
        return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.GSETransformer:
            layer.update_dropout(dropout)
