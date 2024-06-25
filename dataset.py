from torch.utils.data import DataLoader, Dataset
import torch
from KanBERT.kan_bert_tokenizer import KanTokenizer
import torch.nn as nn
import pandas as pd


def load_imdb_data(data_file):
    df = pd.read_csv(data_file)
    texts = df["review"].tolist()
    labels = [
        1 if sentiment == "positive" else 0 for sentiment in df["sentiment"].tolist()
    ]
    return texts, labels


class IMDBDataset(Dataset):
    def __init__(self, text: str, tokenizer: KanTokenizer, labels: str) -> None:
        super().__init__()
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(text)

        return {
            "input": encoding,
            "label": torch.IntTensor([label])
        }
