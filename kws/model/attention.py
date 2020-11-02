import math
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super(Attention, self).__init__()
        attention_dim = embed_dim // num_heads

        self.WQ = nn.Linear(embed_dim, attention_dim, bias=False)
        self.WK = nn.Linear(embed_dim, attention_dim, bias=False)
        self.WV = nn.Linear(embed_dim, attention_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, length, embed_dim)
        # mask: (batch_size, length, length)

        Q = self.WQ(query)
        K = self.WK(key)
        V = self.WV(value)
        # Q, K, V: (batch_size, length, attention_dim)

        norm_factor = math.sqrt(Q.shape[-1])
        attention = torch.bmm(Q, K.transpose(1, 2)) / norm_factor
        # attention: (batch_size, length, length)

        if mask is not None:
            attention = attention.masked_fill(mask, -math.inf)

        attention_score = nn.functional.softmax(attention, dim=-1)
        attention_score = self.dropout(attention_score)
        soft_argmax = torch.bmm(attention_score, V)
        # soft_argmax: (batch_size, length, attention_dim)

        return soft_argmax


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0
        self.attention_heads = nn.ModuleList([Attention(embed_dim, num_heads, dropout)
                                              for _ in range(num_heads)])

        self.linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, length, embed_dim)
        # mask: (batch_size, length, length)

        soft_argmax = [attention(query, key, value, mask) for attention in self.attention_heads]
        soft_argmax = torch.cat(soft_argmax, dim=-1)
        # soft_argmax: (batch_size, length, embed_dim)

        outputs = self.linear(soft_argmax)
        outputs = self.dropout(outputs)
        # outputs: (batch_size, length, embed_dim)

        return outputs
