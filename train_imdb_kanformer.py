from typing import (Any, List,
                    Dict)
from KanBERT.kan_bert_tokenizer import KanTokenizer
from KanBERT.kan_bert_utils import accuracy
from KanBERT.kan_bert import KanBert
import torch.nn as nn
from KANFormer.KANFormer import KANFormer
import torch
from KanBERT.constants import device, maxlen
from einops import rearrange
from dataset import IMDBDataset, load_imdb_data
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from lightning.pytorch.loggers import WandbLogger
import lightning as pl
from torch.utils.data import random_split

text, labels = load_imdb_data("./data/IMDB Dataset.csv")


tokenizer = KanTokenizer(max_length=maxlen)
tokenizer.ingest_vocab_batch(text=text)

dataset_data = IMDBDataset(text=text, labels=labels, tokenizer=tokenizer)

print(tokenizer.vocab_size)

model = KANFormer(vocabulary_size=tokenizer.vocab_size, hidden_size=32, num_heads=2, window_size=4, d_ff=64, num_experts=4, n_experts_per_token=2, n_blocks=6, max_seq_len=128, num_tags_ner=2)

lengths = [int(len(dataset_data)*0.8), int(len(dataset_data)*0.2)]
train_dataset, val_dataset = random_split(dataset=dataset_data, lengths=lengths)

train_dataloader = DataLoader(dataset=train_dataset, 
                            shuffle=True, 
                            batch_size=4)

val_dataloader = DataLoader(dataset=val_dataset, 
                            shuffle=True, 
                            batch_size=4)

for i, y in train_dataloader:
    print(i[0].shape ,y.shape)
    break