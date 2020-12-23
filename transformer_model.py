"""
Implementation of the Transformer model as described in the 'Attention is All You Need' paper
I follow the implementation examples in the following resources:
- "The Annotated Transformer"
- "Aladdin Persons' Transformer from scratch YouTube video"

I try to balance between adhering to the variable names in the paper and using plain English for ease of comprehension
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadedAttention(nn.Module):
    """
    Implementation of the Multi-headed Attention sublayer in the Transformer block
    This sublayer is the same architecture in both the Encoder and Decoder blocks
    """

    def __init__(self, n_heads: int, d_model: int):
        """

        :type d_model: int
        :type n_heads: int
        :param n_heads: number of heads to split into (default = 8 in the paper)
        :param d_model: size of embedding (default = 512 in the paper)
        """
        super(MultiHeadedAttention).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert (self.n_heads * self.head_dim == d_model), "Model size (d_model) has to be divisible by n_heads"

        # these nn.Linear layers are the weight matrices that are multiplied with the input vectors
        # these are equivalent to W_v, W_k, W_q in the paper
        # head_dim are d_k = d_v = d_q = d_model / # heads as in the paper
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # the weight matrix that after concatenating the heads together makes the output from this sublayer
        # have compatible shape/size: d_model
        self.fc_layer = nn.Linear(self.n_heads * self.head_dim, self.d_model)

    def forward(self, value: Tensor, key: Tensor, query: Tensor, mask) -> Tensor:
        """

        :param value:
        :param key:
        :param query:
        :param mask:
        :return:
        """
        # value/key/query.shape: batch_size x sequence length
        # sequence length is either src or trg length depending on if this is being used in the Encoder or Decoder
        # N refers to number of samples/sentences in input batch
        N = query.shape[0]
        value_seq_len, key_seq_len, query_seq_len = value.shape[1], key.shape[1], query.shape[1]

        # splitting value, key, query embeddings into n_heads pieces by .reshaping (same as .view but .view always
        # need to be contiguous)
        value = value.reshape(N, value_seq_len, self.n_heads, self.head_dim)
        key = key.reshape(N, key_seq_len, self.n_heads, self.head_dim)
        query = query.reshape(N, query_seq_len, self.n_heads, self.head_dim)

        # query, key, value matrices from projection through multiplication with the W_q, W_k, W_v weight matrices
        # for calculating attention
        # q,k,v shapes from previous steps are: (N, query_seq_len or key_seq_len or value_seq_len, n_heads, head_dim)
        # bmm with weight matrices of shapes: (head_dim, head_dim)
        # resultant shapes: (N, query_seq_len or key_seq_len or value_seq_len, n_heads, head_dim)
        value = self.values(value)
        key = self.keys(key)
        query = self.queries(query)

        # this part corresponds to Q * K_transposed via bmm (batch matrix multiplication) in the Scaled Dot Product
        # Attention
        # einsum takes care of mapping Q_ij to K_ji so the transpose is taken care of under the hood
        # einsum maps the other dimensions automatically as well so no need to reshape or flatten N and n_heads
        # before multiplication
        q_bmm_kt = torch.einsum("nqhd, nkhd -> nhqk", [query, key])
        # query shape: (N := n, query_seq_len := q, n_heads := h, head_dim := d)
        # key shape: (N := n, key_seq_len := k, n_heads := h, head_dim := d)
        # q_bmm_kt desired shape: (N := n, n_heads := h, query_seq_len := q, key_seq_len := k)
        # because this will need to be bmm again with value, which is of shape: (N, value_seq_len, n_heads, head_dim)

        # wherever in the Tensor that's == 0, replace with a very small number (-1e20)
        if mask is not None:
            q_bmm_kt = q_bmm_kt.masked_fill(mask == 0, float("-1e20"))

        # scaled by * 1/sqrt of d_k before softmax, d_k = d_model // n_heads == head_dim
        # we want a probabilistic distribution across the source sentence (so softmax along the key_seq_len dimension)
        attention = torch.softmax(q_bmm_kt / (math.sqrt(self.head_dim)), dim=3)

        # batch mat mult attention with value via torch.einsum to map the dimensions correctly
        # kew_seq_len == value_seq_len := k in the einsum formula
        attention = torch.einsum("nhqk, nkhd -> nqhd", [attention, value])
        # attention shape: (N := n, n_heads := h, query_seq_len := q, key_seq_len := k)
        # value shape: (N := n, value_seq_len := k, n_heads := h, head_dim := d)
        # desired resultant shape: (N := n, query_seq_len := q, n_heads := h, head_dim := d)

        # concatenate these n_heads attentions together with reshape to flatten last two dim to end up with
        # output.shape: (N, query_seq_len, d_model == n_heads * head_dim)
        output = attention.reshape(N, query_seq_len, self.n_heads * self.head_dim)
        output: Tensor = self.fc_layer(output)

        return output  # output shape: (N, query_seq_len, d_model)


class FeedForward(nn.Module):
    """
    Implement the FeedForward sublayer in the TransformerBlock:
    FFN = Relu(input_x * W_1 + b_1) * W_2 + b_2
    """

    def __init__(self, d_model, d_ff):
        """

        :param d_model: embedding size or model size (default = 512 in the paper)
        :param d_ff: size of hidden layer in the FFN (default = 1024 in the paper)
        """
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        # self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, output_from_attention_sublayer: Tensor) -> Tensor:
        # shape of output_from_attention_sublayer: (N, query_seq_len, d_model)
        x_w_1 = F.relu(self.w_1(output_from_attention_sublayer))
        # shape of x_w_1: (N, query_seq_len, d_ff)
        # x_w_1 = self.dropout(x_w_1)

        # shape of output after going through self.w_2: (N, query_seq_len, d_model)
        output = self.w_2(x_w_1)

        return output


class TransformerBlock(nn.Module):
    """
    Implement the Transformer block with the attention sublayer and FFN sublayer
    """

    def __init__(self, d_model, n_heads, dropout_p, d_ff):
        """

        :param d_model: embedding size or model size (default = 512 in the paper)
        :param n_heads: number of heads to split into in the Multi-Headed Attention layer (default = 8 in the paper)
        :param dropout_p: dropout applied before output of each sublayer (default p=0.1)
        :param d_ff: size of hidden layer in the FFN (default = 1024 in the paper)
        """
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(d_model, n_heads)
        self.add_norm1 = nn.LayerNorm(d_model)
        self.add_norm2 = nn.LayerNorm(d_model)

        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, value, key, query, mask) -> Tensor:
        """
        1. attention sublayer
        2. add and normalize attention sublayer output with residual input from before the attention_sublayer
        3. apply dropout to the added and normed output of the attention sublayer
        4. output from attention sublayer goes through FFN
        5. add and normalize output_from_ffn with residual (output from attention sublayer) from before the ffn layer
        6. apply dropout to the added and normed output of the FFN sublayer

        :param value:
        :param key:
        :param query:
        :param mask:
        :return:
        """

        # output from attention sublayer shape: (N, query_seq_len, d_model)
        attention_sublayer = self.attention(value, key, query, mask)
        output_from_attention_sublayer = self.dropout(self.add_norm1(attention_sublayer + query))
        # should this be + query or + concat of query, key, value??
        # query shape needs to be also: (N, query_seq_len, d_model)??

        output_from_ffn = self.feed_forward(output_from_attention_sublayer)
        output = self.add_norm2(output_from_attention_sublayer + output_from_ffn)
        # output shape: (N, query_seq_len, d_model)
        output = self.dropout(output)

        return output


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout_p, max_length, device):
        super(Encoder, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.input_embedding = nn.Embedding(src_vocab_size, d_model)
        self.max_length = max_length
        self.dropout = nn.Dropout(p=dropout_p)
        self.device = device
