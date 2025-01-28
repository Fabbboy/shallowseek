import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_latents, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_latents = num_latents
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.W_o = nn.Linear(d_model, d_model)
        
        self.latents = nn.Parameter(torch.randn(num_latents, d_model))
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        
        L = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        L = self.split_heads(L)
        
        K = torch.cat([K, L], dim=2)
        V = torch.cat([V, L], dim=2)
        
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            ##scores = scores.masked_fill(mask == 0, -1e9) 
            #use a smaller value for fp16
            scores = scores.masked_fill(mask == 0, -1e4)
        
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        context = torch.matmul(attn_probs, V)
        
        context = self.combine_heads(context)
        
        output = self.W_o(context)
        
        return output, attn_probs
