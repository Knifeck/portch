import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', self.pe)
    def forward(self, x):
        return x + self.pe[:x.size(0), :]
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
    def attention(self, q, k, v, d_k, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn
    def forward(self, x, mask=None):
        bs = x.size(0)
        # Linear projections
        q = self.q_linear(x).view(bs, -1, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_linear(x).view(bs, -1, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_linear(x).view(bs, -1, self.num_heads, self.head_dim).transpose(1,2)
        # Scaled Dot-Product Attention
        att_output, attn = self.attention(q, k, v, self.head_dim, mask=mask)
        # Concatenation of heads
        att_output = att_output.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        # Output projection
        output = self.out_linear(att_output)
        return output, attn
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        att_output, attn = self.multi_head_attention(x, mask=mask)
        x = x + self.dropout_1(self.layer_norm_1(att_output))
        ff_output = self.feed_forward(x)
        x = x + self.dropout_2(self.layer_norm_2(ff_output))
        return x, attn
class Encoder(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, d_model)
        self.pe = PositionalEncoding(d_model=d_model, max_seq_len=1000)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
    def forward(self, x, mask=None):
        bs = x.size(0)
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).repeat(bs, 1)
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x = self.pe(x)
        for layer in self.layers:
            x, attn = layer(x, mask=mask)
        return x
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention_1 = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attention_2 = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
    def forward(self, x, enc_output, src_mask=None, trg_mask=None):
        att_output_1, attn_1 = self.multi_head_attention_1(x, mask=trg_mask)
        x = x + self.dropout_1(self.layer_norm_1(att_output_1))
        att_output_2, attn_2 = self.multi_head_attention_2(x=enc_output, mask=src_mask)
        x = x + self.dropout_2(self.layer_norm_2(att_output_2))
        ff_output = self.feed_forward(x)
        x = x + self.dropout_3(self.layer_norm_3(ff_output))
        return x, attn_1, attn_2
class Decoder(nn.Module):
    def __init__(self, output_size, d_model, num_heads, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, d_model)
        self.pe = PositionalEncoding(d_model=d_model, max_seq_len=1000)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.out = nn.Linear(d_model, output_size)
    def forward(self, x, enc_output, src_mask=None, trg_mask=None):
        bs = x.size(0)
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).repeat(bs, 1)
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x = self.pe(x)
        for layer in self.layers:
            x, attn_1, attn_2 = layer(x, enc_output, src_mask=src_mask, trg_mask=trg_mask)
        out = self.out(x)
        return out, attn_1, attn_2
class Transformer(nn.Module):
    def __init__(self, input_size, output_size, d_model, num_heads, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.encoder = Encoder(input_size=input_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        self.decoder = Decoder(output_size=output_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
    def forward(self, src, trg, src_mask=None, trg_mask=None):
        enc_output = self.encoder(src, mask=src_mask)
        out, attn_1, attn_2 = self.decoder(trg, enc_output, src_mask=src_mask, trg_mask=trg_mask)
        return out
    def encode(self, src, src_mask=None):
        enc_output = self.encoder(src, mask=src_mask)
        return enc_output
    def decode(self, trg, enc_output, src_mask=None, trg_mask=None):
        out, attn_1, attn_2 = self.decoder(trg, enc_output, src_mask=src_mask, trg_mask=trg_mask)
        return out, attn_1, attn_2
# Example Usage
input_size = 5000  # vocabulary size of input language
output_size = 4000  # vocabulary size of output language
d_model = 512  # dimensionality of embedding layer and hidden layers
num_heads = 8  # number of attention heads
num_layers = 6  # number of encoder and decoder layers
dropout = 0.1  # dropout probability
model = Transformer(input_size, output_size, d_model, num_heads, num_layers, dropout)
src = torch.randn(10, 25).long()  # batch_size x seq_len
trg = torch.randn(10, 20).long()  # batch_size x seq_len
out = model(src, trg)
print(out.size())  # output is of shape batch_size x seq_len x output_size