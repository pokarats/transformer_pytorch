"""
**transformer_model.py Module**

Implementation of the Transformer model as described in the 'Attention is All You Need' paper:
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017).
Attention Is All You Need. ArXiv:1706.03762 [Cs]. http://arxiv.org/abs/1706.03762


I followed the implementation examples in the following resources:
    - "The Annotated Transformer: https://nlp.seas.harvard.edu/2018/04/03/attention.html"
    - "Aladdin Perssons' Transformer from Scratch YouTube video: https://www.youtube.com/watch?v=U0s0f995w14&ab_channel=AladdinPersson"
    - "Ben Trevett's Language Translation with Transformer and Torchtext tutorial: https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html?highlight=transformer"


I tried to balance between adhering to the variable names in the paper and using plain English for ease of comprehension
"""
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


logger = logging.getLogger(__name__)


class MultiHeadedAttention(nn.Module):
    """
    Implementation of the Multi-headed Attention sublayer in the Transformer block
    This sublayer is the same architecture in both the Encoder and Decoder blocks
    The multi-headded attention sublayer comprises the scaled-dot product attention mechanism
    """

    def __init__(self, n_heads: int, d_model: int):
        """

        :type d_model: int
        :type n_heads: int
        :param n_heads: number of heads to split into (default = 8 in the paper)
        :param d_model: size of embedding (default = 512 in the paper)
        """
        super(MultiHeadedAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert (self.n_heads * self.head_dim == d_model), "Model size (d_model) has to be divisible by n_heads"

        # these nn.Linear layers are the weight matrices that are multiplied with the input vectors
        # these are equivalent to W_v, W_k, W_q in the paper
        # head_dim are d_k = d_v = d_q = d_model / # heads as in the paper
        self.fc_values = nn.Linear(self.d_model, self.d_model, bias=False)
        self.fc_keys = nn.Linear(self.d_model, self.d_model, bias=False)
        self.fc_queries = nn.Linear(self.d_model, self.d_model, bias=False)

        # the weight matrix that after concatenating the heads together makes the output from this sublayer
        # have compatible shape/size: d_model
        self.fc_out = nn.Linear(self.n_heads * self.head_dim, self.d_model)

    def forward(self, value: Tensor, key: Tensor, query: Tensor, mask: Tensor) -> Tensor:
        """

        :param value: value Tensor shape (N, sequence length, d_model)
        :param key: key Tensor shape (N, sequence length, d_model)
        :param query: query Tensor shape (N, sequence length, d_model)
        :param mask: src or trg masking Tensor shape src(N, 1, 1, src_seq_len) trg(N, 1, trg_seq_len, trg_seq_len)
        :return:
        """
        # value/key/query.shape: (N, sequence length, d_model)
        # sequence length is either src or trg length depending on if this is being used in the Encoder or Decoder
        # N refers to number of samples/sentences in input batch i.e. batch size
        N = query.shape[0]
        value_seq_len, key_seq_len, query_seq_len = value.shape[1], key.shape[1], query.shape[1]

        # query, key, value matrices from projection through multiplication with the W_q, W_k, W_v weight matrices
        # for calculating attention
        # transformed with weight matrices of shapes: (d_model, d_model)
        # resultant shapes: (N, query_seq_len or key_seq_len or value_seq_len, d_model)
        value = self.fc_values(value)
        key = self.fc_keys(key)
        query = self.fc_queries(query)

        # splitting value, key, query embeddings into n_heads pieces by .reshaping (same as .view but .view always
        # need to be contiguous)
        # q,k,v shapes from previous steps are: (N, query_seq_len or key_seq_len or value_seq_len, d_model)
        value = value.reshape(N, value_seq_len, self.n_heads, self.head_dim)
        key = key.reshape(N, key_seq_len, self.n_heads, self.head_dim)
        query = query.reshape(N, query_seq_len, self.n_heads, self.head_dim)
        # resultant q,k,v shapes from previous steps are:
        # (N, query_seq_len or key_seq_len or value_seq_len, n_heads, head_dim)

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

        return self.fc_out(output)  # output shape: (N, query_seq_len, d_model)


class FeedForward(nn.Module):
    """
    Implement the FeedForward sublayer in the TransformerBlock:
    FFN = Relu(input_x * W_1 + b_1) * W_2 + b_2
    """

    def __init__(self, d_model, d_ff, dropout_p):
        """

        :param d_model: embedding size or model size (default = 512 in the paper)
        :param d_ff: size of hidden layer in the FFN (default = 2048 in the paper)
        :param dropout_p: dropout applied after ReLu before final transform (default p=0.1)
        """
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, output_from_attention_sublayer: Tensor) -> Tensor:
        # shape of output_from_attention_sublayer: (N, query_seq_len, d_model)
        x_w_1 = F.relu(self.w_1(output_from_attention_sublayer))
        # shape of x_w_1: (N, query_seq_len, d_ff)
        x_w_1 = self.dropout(x_w_1)

        # shape of output after going through self.w_2: (N, query_seq_len, d_model)
        output = self.w_2(x_w_1)

        return output


class EncoderLayer(nn.Module):
    """
    Implement the Encoder block with the 2 sublayers: 1) attention sublayer 2) FFN sublayer.
    This layer is stacked n_layers times in the Encoder part of the Transformer architecture
    """

    def __init__(self, d_model, n_heads, d_ff, dropout_p):
        """

        :param d_model: embedding size or model size (default = 512 in the paper)
        :param n_heads: number of heads to split into in the Multi-Headed Attention layer (default = 8 in the paper)
        :param d_ff: size of hidden layer in the FFN (default = 2048 in the paper)
        :param dropout_p: dropout applied before output of each sublayer (default p=0.1)
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadedAttention(n_heads, d_model)
        self.add_norm_attention = nn.LayerNorm(d_model)
        self.add_norm_ffn = nn.LayerNorm(d_model)

        self.feed_forward = FeedForward(d_model, d_ff, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, src, src_mask) -> Tensor:
        """
        1. attention sublayer
            1.a in the EncoderLayer V,K,Q are all from the same x input
        2. add and normalize attention sublayer (with dropout applied) output with residual input from before the attention_sublayer
        3. output from attention sublayer goes through FFN
        4. add and normalize output_from_ffn (with dropout applied) with residual (output from attention sublayer) from before the ffn layer


        :param src: src shape (N, src_seq_len, d_model)
        :param src_mask: src_mask shape (N, 1, 1, src_seq_len)
        :return: output Tensor that will become input Tensor in the next EncoderLayer
        """

        # output from attention sublayer shape: (N, src_seq_len, d_model)
        attention_sublayer = self.attention(src, src, src, src_mask)
        output_from_attention_sublayer = self.add_norm_attention(src + self.dropout(attention_sublayer))

        output_from_ffn = self.feed_forward(output_from_attention_sublayer)
        output = self.add_norm_ffn(output_from_attention_sublayer + self.dropout(output_from_ffn))
        # output shape: (N, src_seq_len, d_model)

        return output


class PositionEmbedding(nn.Module):
    """
    TODO implement PositionEmbedding per paper
    """
    pass


class Encoder(nn.Module):
    """
    Implement the Encoder block of nx Encoder layers. From input token embeddings + position embeddings and
    Nx EncodingLayers to generating the output that will be taken up by the Decoder block
    """
    def __init__(self, src_vocab_size, d_model, nx_layers, n_heads, d_ff, dropout_p, max_length, device):
        """

        :param src_vocab_size:
        :param d_model:
        :param nx_layers:
        :param n_heads:
        :param d_ff:
        :param dropout_p:
        :param max_length:
        :param device:
        """
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.device = device

        self.input_embedding = nn.Embedding(src_vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout_p)
                                             for _ in range(nx_layers)])

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """

        :param src: src shape (N, src_seq_len)
        :param src_mask: src_mask shape (N, 1, 1, src_seq_len)
        :return: output from the last encoder layer in the stack
        """

        # N refers to number of samples/sentences in input batch i.e. batch size
        N, src_seq_len = src.shape

        # generate position indices from 0 to src_seq_len and expand dimension to cover all samples in N batch size
        # positions shape: (N, src_seq_len)
        positions = torch.arange(0, src_seq_len).expand(N, src_seq_len).to(self.device)

        # input embeddings is the element-wise sum between input token embedding and position embedding
        # input token embedding is scaled by a factor of sqrt(d_model) in the paper
        # dropout is applied to the embeddings after summing
        src_input_embeddings = self.dropout(self.input_embedding(src) * math.sqrt(self.d_model) +
                                            self.position_embedding(positions))

        # src_input_embeddings shape: (N, src_seq_len, d_model)
        # the ouput from each enc_layer becomes the input to the next enc_layer
        # only the ouput from the last enc_layer will be sent out as input to the decoder block
        for enc_layer in self.encoder_layers:
            src_input_embeddings = enc_layer(src_input_embeddings, src_mask)

        return src_input_embeddings


class DecoderLayer(nn.Module):
    """
    Implement the Decoder block with the 3 sublayers:
        1. decoder attention sublayer
        2. encoder attention sublayer
        3. FFN sublayer

    This layer is stacked n_layers times in the Decoder part of the Transformer architecture
    """

    def __init__(self, d_model, n_heads, d_ff, dropout_p):
        super(DecoderLayer, self).__init__()
        self.masked_decoder_attention = MultiHeadedAttention(n_heads, d_model)
        self.add_norm_dec_attention = nn.LayerNorm(d_model)
        self.encoder_attention = MultiHeadedAttention(n_heads, d_model)
        self.add_norm_enc_attention = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_p)
        self.add_norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, trg, src_encoder_output, trg_mask, src_mask) -> Tensor:
        """
        1. masked decoder attention sublayer
            1.a in the EncoderLayer V,K,Q are all from the same input trg
        2. apply dropout the decoder attention sublayer
        3. add and normalize attention sublayer output with residual input from before the decoder attention sublayer
        4. encoder attention sublayer
            4.a in the DecoderLayer Q is from output of previous decoder sublayer,
            V,K are from last encoder layer output
        5. apply dropout to the encoder attention sublayer
        6. add and normalize encoder attention sublayer output with residual from the output of step 3.
        7. output from encoder attention sublayer goes through FFN
        8. apply dropout to the output from FFN
        9. add and normalize output_from_ffn with residual (output from decoder attention sublayer) from before the ffn layer



        :param trg: shape (N, trg_seq_len, d_model)
        :param src_encoder_output: shape (N, src_seq_len, d_model)
        :param trg_mask: shape (batch N, 1, trg_seq_len, trg_seq_len)
        :param src_mask: shape (N, 1, 1, src_seq_len)
        :return:
        """

        # Steps 1, 2, 3
        decoder_attention_sublayer = self.masked_decoder_attention(trg, trg, trg, trg_mask)
        output_dec_attention_sublayer = self.add_norm_dec_attention(trg + self.dropout(decoder_attention_sublayer))
        # output_dec_attention_sublayer shape: (N, trg_seq_len, d_model)

        # Steps 4, 5, 6
        enc_attention_sublayer = self.encoder_attention(value=src_encoder_output, key=src_encoder_output,
                                                        query=output_dec_attention_sublayer, mask=src_mask)
        output_enc_attention_sublayer = self.add_norm_enc_attention(output_dec_attention_sublayer +
                                                                    self.dropout(enc_attention_sublayer))
        # output_enc_attention_sublayer shape: (N, trg_seq_len, d_model)

        # Steps 7, 8, 9
        output_from_ffn = self.feed_forward(output_enc_attention_sublayer)
        output = self.add_norm_ffn(output_enc_attention_sublayer + self.dropout(output_from_ffn))
        # output shape: (N, trg_seq_len, d_model)

        return output


class Decoder(nn.Module):
    """
    Implement the Decoder block of nx DecodingLayers. From input token embeddings + position embeddings and
    Nx DecodingLayers to generating the output that will be interpreted as the predicted translation
    """
    def __init__(self, trg_vocab_size, d_model, nx_layers, n_heads, d_ff, dropout_p, max_length, device):
        """

        :param trg_vocab_size:
        :param d_model:
        :param nx_layers:
        :param n_heads:
        :param d_ff:
        :param dropout_p:
        :param max_length:
        :param device:
        """
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.device = device

        self.input_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)  # TODO implement PositionEmbedding per paper

        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout_p)
                                             for _ in range(nx_layers)])

        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, trg: Tensor, src_encoder_output: Tensor, trg_mask: Tensor, src_mask: Tensor) -> Tensor:
        """

        :param trg: src shape (N, trg_seq_len)
        :param src_encoder_output: src_input_embeddings shape (N, src_seq_len, d_model)
        :param trg_mask: shape (batch N, 1, trg_seq_len, trg_seq_len)
        :param src_mask: shape (N, 1, 1, src_seq_len)
        :return: trg decoder ouput shape: (N, trg_seq_len, d_model)
        """
        # N refers to number of samples/sentences in input batch i.e. batch size
        N, trg_seq_len = trg.shape

        # generate position indices from 0 to trg_seq_len and expand dimension to cover all samples in N batch size
        # positions shape: (N, trg_seq_len)
        positions = torch.arange(0, trg_seq_len).expand(N, trg_seq_len).to(self.device)

        # input embeddings is the element-wise sum between input token embedding and position embedding
        # input token embedding is scaled by a factor of sqrt(d_model) in the paper
        # dropout is applied to the embeddings after summing
        trg_input_embeddings = self.dropout(self.input_embedding(trg) * math.sqrt(self.d_model) +
                                            self.position_embedding(positions))

        # trg_input_embeddings shape: (N, trg_seq_len, d_model)
        # the ouput from each dec_layer becomes the input to the next dec_layer
        # only the ouput from the last deco_layer will be sent out to the linear layer after the Decoder block
        for dec_layer in self.decoder_layers:
            trg_input_embeddings = dec_layer(trg_input_embeddings, src_encoder_output, trg_mask, src_mask)

        return self.fc_out(trg_input_embeddings)  # output shape: (N, trg_seq_len, trg_vocab_size)


class Generator(nn.Module):
    """
    Implement the last linear layer and the softmax of the Transformer architecture

    This is not necessary if using nn.CrossEntropyLoss as softmax is already built-in
    """
    def __init__(self, d_model, trg_vocab_size):
        super(Generator, self).__init__()
        self.fc_proj = nn.Linear(d_model, trg_vocab_size)

    def forward(self, output_from_decoder):
        """

        :param output_from_decoder: shape (N, trg_seq_len, d_model)
        :return: output_from_decoder: shape (N, trg_seq_len, trg_vocab_size)???
        """
        return F.softmax(self.fc_proj(output_from_decoder), dim=-1)


class Transformer(nn.Module):
    """
    Implement the Transformer architecture consisting of nx_layers of the encoder block and nx_layers of the decoder
    block.
    """
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, d_model, nx_layers, n_heads, d_ff,
                 dropout_p, max_length, device):
        """

        :param src_vocab_size: src language vocab size == len(src_field.vocab)
        :param trg_vocab_size: trg language vocab size == len(trg_field.vocab)
        :param src_pad_idx: index for src padding
        :param trg_pad_idx: index for trg padding
        :param d_model: model hidden size, 512 in the paper
        :param nx_layers: number of encoder/decoder layers, 6 in the paper
        :param n_heads: number of attention heads, 8 in the paper
        :param d_ff: number of hidden size for the FFN sublayer, 2084 in the paper or 4 * d_model
        :param dropout_p: p value for the dropout, paper uses p = 0.1
        :param max_length: maximum tokens in the sentence, paper uses 300; set to 150
        :param device: cpu or cuda
        """
        super(Transformer, self).__init__()
        logger.info(f'initializing Transformer model')
        self.encoder = Encoder(src_vocab_size, d_model, nx_layers, n_heads, d_ff, dropout_p, max_length, device)
        self.decoder = Decoder(trg_vocab_size, d_model, nx_layers, n_heads, d_ff, dropout_p, max_length, device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        # self.output_generator = Generator(d_model, trg_vocab_size)
        self.device = device

    def make_src_mask(self, src: Tensor):
        """
        Wherever src is not a pad idx, add dim of 1
        :param src: src shape (N, src_seq_len)
        :return: src_mask shape (N, 1, 1, src_seq_len)
        """

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask shape after unsqueeze: (N, 1, 1, src_seq_len)

        return src_mask

    def make_trg_mask(self, trg: Tensor):
        """

        :param trg: trg shape (N, trg_seq_len)
        :return: trg_mask shape (N, 1, trg_seq_len, trg_seq_len)
        """
        N, trg_seq_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_seq_len, trg_seq_len)))
        # trg_mask shape: (trg_seq_len, trg_seq_len)
        # where the bottom left of the diagonal is filled with ones and the upper corner is 0-filled

        trg_mask = trg_mask.expand(N, 1, trg_seq_len, trg_seq_len).to(self.device)
        # trg_mask shape: (N, 1, trg_seq_len, trg_seq_len)

        return trg_mask

    def forward(self, src: Tensor, trg: Tensor) -> Tensor:
        """

        :param src: src shape (N, src_seq_len)
        :param trg: trg shape (N, trg_seq_len)
        :return: output shape (N, trg_seq_len, d_model)
        """

        logger.debug(f'src shape: {src.shape}')
        logger.debug(f'trg shape: {trg.shape}')

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # src_mask shape: (N, 1, 1, src_seq_len]
        # trg_mask shape: (N, 1, trg_seq_len, trg_seq_len)

        src_encoder_ouput = self.encoder(src, src_mask)
        # src_encoder_ouput shape: (N, src_seq_len, d_model)

        output = self.decoder(trg, src_encoder_ouput, trg_mask, src_mask)
        # output shape: (N, trg_seq_len, trg_vocab_size)

        # last linear layer and softmax to generate probabilities
        # output_prob = self.output_generator(output)
        logger.debug(f'Transformer model output shape: {output.shape}')

        return output


if __name__ == "__main__":
    # dummy data test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    src = torch.tensor([[2, 5, 6, 4, 0, 9, 5, 3, 1], [2, 8, 7, 0, 4, 5, 6, 7, 3]]).to(device)
    trg = torch.tensor([[2, 7, 4, 0, 5, 9, 3, 1], [2, 5, 6, 2, 4, 7, 6, 3]]).to(device)

    # hyperparameters
    src_pad_idx = 1
    trg_pad_idx = 1
    src_vocab_size = 10  # 0 to 9, 1 == 'pad' 2 == 'sos' 3 == 'eos'
    trg_vocab_size = 10  # 0 to 9, 1 == 'pad' 2 == 'sos' 3 == 'eos'
    d_model_hidden_size = 512
    nx = 6
    num_attention_heads = 8
    d_ff_hidden_size = 2048
    drop_out = 0.1
    max_seq_length = 100  # 300 in paper

    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, d_model_hidden_size, nx,
                        num_attention_heads, d_ff_hidden_size, drop_out, max_seq_length, device=device).to(device)

    # predicted output without 'eos' token from trg
    print(f'src: {src}, src shape: {src.shape}')
    print(f'trg: {trg} trg shape: {trg.shape}')
    print(f'trg without last token {trg[:, :-1]}, trg without <eos> shape: {trg[:, :-1].shape}')
    out = model(src, trg[:, :-1])
    print(f'out.shape: {out.shape}')
    print(out)
    print(out.argmax(2)[:, -1])
