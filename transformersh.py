"""
In this, Single Head Attention is implemented.

maths behind,
Q = X * Wq
K = X * Wk
V = X * Wv 

scores = (Q * K^T) / dk
attention = softmax(scores) * V

y = FFN(attention(x))

"""



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import math

class Attention(nn.Module):
    def __init__(self, input_dims, dmodel, dk):
        super(Attention, self).__init__()
        self.Wk = nn.Linear(input_dims, dk)
        self.Wv = nn.Linear(input_dims, dk)
        self.Wq = nn.Linear(input_dims, dmodel)
        self.dk = dk

    def forward(self, x):
        q = self.Wq(x) # [B * L * dk]
        k = self.Wk(x) # [B * L * dk]
        v = self.Wv(x) # [B * L * dmodel]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        weights = functional.softmax(scores, dim=-1)
        y = torch.matmul(weights, v)
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
    def __init__(self, dmodel=512, dff=2048, input_dims=None, dk=None, vocab_size=2048):
        super(Transformer, self).__init__()
        self.attention = Attention(input_dims, dmodel, dk)
        self.ffn1 = FFN(dmodel, dff)
        self.ffn2 = FFN(dmodel, dff)
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        self.projections = nn.Linear(dmodel, vocab_size)

    def forward(self, x):
        attention = self.attention(x)
        y1 = self.norm1(x + attention)
        y2 = self.ffn1(y1)
        y2 = self.norm2(y1 + y2)
        y3 = self.projections(y2)
        y = functional.softmax(y3, dim=-1)
        return y2        