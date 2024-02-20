# This is a 3d DISCRETE transformer. It's a bit different from the usual transformer, as it operates on 2d discrete data.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Embedding2d(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding2d, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.weight = nn.Parameter(torch.rand(num_embeddings, embedding_dim))
        self.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
    def forward(self, inputs):
        # flatten and convert to one hot
        flat_input = rearrange(inputs, 'b h w t -> b t (h w)')
        one_hots = F.one_hot(flat_input, self._num_embeddings).float()
        embeddings = torch.matmul(one_hots, self.weight)
        # unflatten
        embeddings = rearrange(embeddings, 'b t (h w) c -> b c h w t', h=inputs.shape[1])
        return embeddings

class ToProbabilities(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(ToProbabilities, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self.linear = nn.Conv3d(embedding_dim, num_embeddings, 1)

    def forward(self, inputs):
        y = self.linear(inputs) # B LOGIT H W T
        return y

class Attention(nn.Module):
    def __init__(self, num_heads, embedding_dim):
        super(Attention, self).__init__()
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim
        self._head_dim = embedding_dim // num_heads
        self.q = nn.Conv3d(embedding_dim, embedding_dim, (3, 3, 1), padding=(1, 1, 0))
        self.k = nn.Conv3d(embedding_dim, embedding_dim, (3, 3, 1), padding=(1, 1, 0))
        self.v = nn.Conv3d(embedding_dim, embedding_dim, (3, 3, 1), padding=(1, 1, 0))
        self.r = nn.Conv3d(embedding_dim, embedding_dim, (3, 3, 1), padding=(1, 1, 0))
        self.to_out = nn.Conv3d(embedding_dim, embedding_dim, (3, 3, 1), padding=(1, 1, 0))

    def forward(self, x):
        b, c, h, w, t = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        r = self.r(x)
        q = rearrange(q, 'b (head c) h w t -> b head t (c h w)', head=self._num_heads)
        k = rearrange(k, 'b (head c) h w t -> b head t (c h w)', head=self._num_heads)
        v = rearrange(v, 'b (head c) h w t -> b head t (c h w)', head=self._num_heads)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attention = rearrange(attention, 'b head t (c h w) -> b (head c) h w t', h=h, w=w, head=self._num_heads)
        attention = attention * torch.sigmoid(r)
        return self.to_out(attention)

class FFN(nn.Module):
    def __init__(self, embedding_dim, expansion_factor=4):
        super(FFN, self).__init__()
        self._expansion_factor = expansion_factor
        self._embedding_dim = embedding_dim
        self.linear1 = nn.Conv3d(embedding_dim, expansion_factor * embedding_dim, 1)
        self.linear2 = nn.Conv3d(expansion_factor * embedding_dim // 2, embedding_dim, 1)
        self.relu = nn.GELU()

    def forward(self, x):
        y1, y2 = self.linear1(x).chunk(2, dim=1)
        y = self.relu(y1) * y2
        y = self.linear2(y)
        return y
        

class Norm(nn.Module):
    def __init__(self, embedding_dim):
        super(Norm, self).__init__()
        self._embedding_dim = embedding_dim
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        B, C, H, W, T = x.shape
        x = rearrange(x, 'b c h w t -> b h w t c')
        x = self.norm(x)
        x = rearrange(x, 'b h w t c -> b c h w t')
        return x

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, embedding_dim, expansion_factor=4):
        super(TransformerBlock, self).__init__()
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim
        self._expansion_factor = expansion_factor
        self.attention = Attention(num_heads, embedding_dim)
        self.ffn = FFN(embedding_dim, expansion_factor)
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)


    def forward(self, x):
        y = self.norm1(x)
        y = self.attention(y)
        y = self.dropout(y)
        y = x + y

        x = y
        y = self.norm2(y)
        y = self.ffn(y)
        y = self.dropout2(y)
        y = x + y
        return y       

class Transformer3d(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_heads, num_layers, expansion_factor=4):
        super(Transformer3d, self).__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._expansion_factor = expansion_factor
        self.embedding = Embedding2d(num_embeddings, embedding_dim)
        self.to_probabilities = ToProbabilities(num_embeddings, embedding_dim)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(num_heads, embedding_dim, expansion_factor) for _ in range(num_layers)])
    def forward(self, x):
        x = self.embedding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        return self.to_probabilities(x)