import torch
import torch.nn as nn
import lightning as pL
from .kan_linear import KANLinear
from .kan_bert_utils import (
    KanBertSelfAttention,
    KanBertEmbeddings,
    KanBertEncoder,
    KanBertNetPos,
    get_attn_pad_mask,
    gelu
)

from .constants import *
device = 'cuda'
class KanBert(nn.Module):
    def __init__(self, vocab_size) -> None:
        super(KanBert, self).__init__()
        self.embedding = KanBertEmbeddings(vocabulary_size=vocab_size)
        self.layers = nn.ModuleList([KanBertEncoder() for _ in range(n_layers)])
        self.fc_kan_1 = KANLinear(d_model, d_model)
        self.activ_1 = nn.Tanh()
        self.fc_kan_2 = KANLinear(d_model, d_model)
        self.activ_2 = gelu
        self.norm = nn.LayerNorm(d_model)
        self.classify = nn.Linear(d_model, n_classify)

    def forward(self, input_ids, segment_ids, masked_pos=None):
        output = self.embedding(input_ids, segment_ids).to(device)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids).to(device)
        
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        
        h_kan = self.fc_kan_1(output[:, 0])
        h_pooled = self.activ_1(h_kan)
        logits_clsf = self.classify(h_pooled)

        return logits_clsf