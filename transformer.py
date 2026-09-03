""""
In this, Multi Headed Attention is implemented.

maths behind,
Q = X * Wq
K = X * Wk
V = X * Wv 

scores = (Q * K^T) / sqrt(dk)
attention = softmax(scores) * V

MultiHead(Q, K, V) = Concat(Attention(head_i, ..., head_h)) * Wo

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
            
class MultiHeadAttention(nn.Module):
    def __init__(self, dmodel=512, nheads=6):
        super(MultiHeadAttention, self).__init__()
        assert dmodel % nheads == 0
        self.dmodel = dmodel
        self.nheads = nheads
        self.dk = self.dmodel // self.nheads
        self.Wq = nn.Linear(dmodel, dmodel)
        self.Wk = nn.Linear(dmodel, dmodel)
        self.Wv = nn.Linear(dmodel, dmodel)
        self.Wo = nn.Linear(dmodel, dmodel)

    def ScaledDotProduct(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.dk)
        attention = torch.matmul(functional.softmax(scores, dim=-1), V)
        return attention

    def SplitHeads(self, x):
        batch_size, sequence_length, dmodel = x.size()
        return x.view(batch_size, sequence_length, self.nheads, self.dk).transpose(1,2)

    def CombineHeads(self, x):
        batch_size, _, sequence_length, dk = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, sequence_length, self.dmodel)

    def forward(self, x):
        Q = self.SplitHeads(self.Wq(x))
        K = self.SplitHeads(self.Wk(x))
        V = self.SplitHeads(self.Wv(x))
        attention = self.ScaledDotProduct(Q, K, V)
        y = self.CombineHeads(attention)
        y = self.Wo(y)
        return y


class FFN(nn.Module):
    def __init__(self, dmodel, dff):
        super(FFN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dmodel, dff),
            nn.ReLU(),
            nn.Linear(dff, dmodel)
        )

    def forward(self, x):
        y = self.network(x)
        return y

class Transformer(nn.Module):
    def __init__(self, dmodel=512, dff=2048, nheads=8, vocab_size=2048, max_seq_length=128):
        super(Transformer, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, dmodel)
        self.pos_encoddings = PositionalEncoding(dmodel, max_seq_length)
        self.MultiHeadAttention = MultiHeadAttention(dmodel, nheads)
        self.FFN = FFN(dmodel, dff)
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        self.projections = nn.Linear(dmodel, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.pos_encoddings(x)
        attention = self.MultiHeadAttention(x)
        y1 = self.norm1(x + attention)
        y2 = self.FFN(y1)
        y3 = self.norm2(y1 + y2)
        y4 = self.projections(y3)
        y = y4[:,-1,:]
        return y