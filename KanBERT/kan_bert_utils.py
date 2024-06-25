import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import ModuleList
import numpy as np
import math
import re
from typing import List
from torch import Tensor
from random import *
from .kan_linear import KANLinear

from .constants import *

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def accuracy(probs: List[Tensor], labels: List[Tensor]):
    preds = torch.max(probs, dim=1)
    print(preds)
    # assert (preds.shape == labels.shape)
    return torch.sum(preds==labels).item()

class KanBertEmbeddings(nn.Module):
    def __init__(self, 
                 hidden_dropout_prob=0.1, 
                 layer_norm_eps=1e-12, 
                 type_vocab_size=n_segments, 
                 vocabulary_size=30522, 
                 final_embed_size=d_model, 
                 max_pos_embeddings=maxlen) -> None:
        super(KanBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocabulary_size, final_embed_size) 
        self.pos_embeddings = nn.Embedding(max_pos_embeddings+1, final_embed_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, final_embed_size)
        self.layer_norm = nn.LayerNorm(final_embed_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids):
        input_shape = input_ids.size() 
        seq_length = input_shape[1]

        pos_ids = torch.arange(seq_length, dtype=torch.long).to(device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)        
        input_embeds = self.word_embeddings(input_ids)
        pos_embeds = self.pos_embeddings(pos_ids)
        
        embeddings = input_embeds + token_type_embeddings + pos_embeds
        
        embeddings = self.layer_norm(embeddings)
        embeddings  = self.dropout(embeddings)

        return embeddings

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)

    return pad_attn_mask.expand(batch_size, len_q, len_k)

class ScaledDotProductAttention(nn.Module):
   def __init__(self):
       super(ScaledDotProductAttention, self).__init__()

   def forward(self, Q, K, V, attn_mask):
       scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
       scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
       attn = nn.Softmax(dim=-1)(scores)
       context = torch.matmul(attn, V)
       return context, attn

class KanBertSelfAttention(nn.Module):
    def __init__(self, final_embed_size=d_model) -> None:
        super(KanBertSelfAttention, self).__init__()
        self.final_embed = final_embed_size
        self.W_Q = nn.Linear(final_embed_size, d_q* n_heads)
        self.W_K = nn.Linear(final_embed_size, d_k* n_heads)
        self.W_V = nn.Linear(final_embed_size, d_v* n_heads)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        # print(attn_mask.shape)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(n_heads * d_v, self.final_embed)(context)

        return nn.LayerNorm(self.final_embed)(output + residual), attn
    
class KanBertNetPos(nn.Module):
    def __init__(self):
        super(KanBertNetPos, self).__init__()
        self.fc1 = KANLinear(d_model, d_ff)
        self.fc2 = KANLinear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(gelu(self.fc1(x)))

class KanBertEncoder(nn.Module):
    def __init__(self):
        super(KanBertEncoder, self).__init__()
        self.enc_self_attn = KanBertSelfAttention()
        self.pos_ffn = KanBertNetPos()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

